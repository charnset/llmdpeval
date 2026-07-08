from pathlib import Path
from pprint import pprint
from typing import Any

import yaml

from llama_index.core.schema import TextNode

from metadata import extract_document_section_records_from_file


class LlamaIndexRAGPreprocessor:
    def __init__(
        self,
        rag_corpus_path: str | Path = "rag_corpus",
        document_pattern: str = "*.txt",
    ) -> None:
        self.rag_corpus_path = Path(rag_corpus_path)
        self.document_pattern = document_pattern
        self.documents: list[Path] = []
        self.document_sections: list[dict[str, Any]] = []
        self.nodes: list[TextNode] = []

    def load_documents(self) -> list[Path]:
        """Find source document files in the RAG corpus."""

        self.documents = [
            document_path
            for document_path in sorted(self.rag_corpus_path.rglob(self.document_pattern))
            if document_path.is_file()
        ]
        return self.documents

    def extract_document_sections(self) -> list[dict[str, Any]]:
        """Extract document-section records from all loaded documents."""

        if not self.documents:
            self.load_documents()

        self.document_sections = []
        for document_path in self.documents:
            self.document_sections.extend(
                extract_document_section_records_from_file(document_path)
            )
        return self.document_sections

    def build_nodes(self) -> list[TextNode]:
        """Convert extracted document sections into LlamaIndex nodes."""

        if not self.document_sections:
            self.extract_document_sections()

        self.nodes = [
            self.document_section_record_to_node(document_section_record)
            for document_section_record in self.document_sections
        ]
        return self.nodes

    def preprocess(self) -> list[TextNode]:
        """Run the full source-document -> document-section -> node pipeline."""

        self.load_documents()
        self.extract_document_sections()
        return self.build_nodes()

    def document_section_record_to_node(
        self,
        document_section_record: dict[str, Any],
    ) -> TextNode:
        """Convert one document-section record into a LlamaIndex TextNode."""

        metadata = document_section_record["metadata"].copy()
        node_text = (
            f"Document title: {metadata['document_title']}\n"
            f"Section: {metadata['document_section']}"
        )
        return TextNode(
            text=node_text,
            metadata=metadata,
            id_=self._build_node_id(metadata),
        )

    def print_nodes(self, limit: int | None = 10) -> None:
        """Print a preview of the preprocessed corpus nodes."""

        if not self.nodes:
            self.preprocess()

        print_nodes(self.nodes, limit=limit)

    def print_document_nodes(
        self,
        document_name_or_path: str | Path,
        limit: int | None = None,
    ) -> list[TextNode]:
        """Print nodes for one corpus document and return them for debugging."""

        if not self.nodes:
            self.preprocess()

        document_name_or_path = str(document_name_or_path)
        nodes = [
            node
            for node in self.nodes
            if node.metadata["document_filepath"] == document_name_or_path
            or node.metadata["document_filename"] == document_name_or_path
        ]

        print(f"Document: {document_name_or_path}")
        print(f"Nodes: {len(nodes)}")
        print_nodes(nodes, limit=limit)
        return nodes

    def _build_node_id(self, metadata: dict[str, Any]) -> str:
        return (
            f"{metadata['document_filepath']}::"
            f"{metadata['document_section']}::"
            f"{metadata['document_section_index']}"
        )


def print_nodes(nodes: list[TextNode], limit: int | None = 10) -> None:
    nodes_to_print = nodes if limit is None else nodes[:limit]

    for index, node in enumerate(nodes_to_print, start=1):
        document_path = node.metadata["document_filepath"]
        document_section = node.metadata["document_section"]
        node_text = node.get_content(metadata_mode="none")

        print(f"\n--- Node {index} ---")
        print(f"Document: {document_path}")
        print(f"Section : {document_section}")
        print(f"Length  : {len(node_text):,} characters")
        print("Metadata:")
        pprint(node.metadata, indent=4, sort_dicts=False)
        print("Text:")
        print(node_text[:1_000])


def load_prompt(prompt_path: Path, prompt_value_path: Path) -> str:
    prompt_template = prompt_path.read_text(encoding="utf-8")
    prompt_values = load_prompt_values(prompt_value_path)
    return prompt_template.format(**prompt_values)


def load_prompt_values(prompt_value_path: Path) -> dict[str, Any]:
    return yaml.safe_load(prompt_value_path.read_text(encoding="utf-8"))


def load_retrieval_query(prompt_value_path: Path) -> str:
    prompt_values = load_prompt_values(prompt_value_path)
    return (
        f"Query type: {prompt_values['query_type']}\n"
        f"Query definition: {prompt_values['query_definition']}\n"
        f"Mechanism: {prompt_values['mechanism']}"
    )


def print_prompt(prompt: str, title: str = "Filled prompt") -> None:
    print(f"\n{title}")
    print("=" * 128)
    print(prompt)


def print_response(response) -> None:
    print("\nLLM response")
    print("=" * 128)
    print(response)


def print_generation_summary(generation: dict) -> None:
    print("\nGenerated code")
    print("=" * 128)
    print(f"Saved code path: {generation['save_code_file_path']}")
    print(generation["generated_code"])


def print_retrieved_nodes(retrieved_nodes, show_code: bool = False) -> None:
    for rank, node_with_score in enumerate(retrieved_nodes, start=1):
        node = node_with_score.node
        metadata = node.metadata
        score = node_with_score.score
        text = " ".join(node.get_content(metadata_mode="none").split())
        code_blocks = metadata.get("code") or []

        print(f"\n[{rank}] score={score:.3f}")
        print(f"Document: {metadata['document_filepath']}")
        print(f"Section : {metadata['document_section']}")
        print(
            f"Section position: "
            f"{metadata['document_section_index']} of {metadata['document_section_count']}"
        )
        print(f"Has code: {metadata['has_code']}")
        if show_code and code_blocks:
            print("Code:")
            for code_index, code in enumerate(code_blocks, start=1):
                print(f"--- code block {code_index} ---")
                print(code)
        print(f"Text preview: {text[:100]}")


class RAGQueryPostprocessor:
    def __init__(
        self,
        prompt: str,
        candidate_nodes,
        all_nodes=None,
        threshold: float = 0.035,
    ) -> None:
        self.prompt = prompt
        self.candidate_nodes = candidate_nodes
        self.all_nodes = list(all_nodes or [])
        self.threshold = threshold
        self.final_nodes = []
        self.final_document_paths = []
        self.context = ""
        self.final_prompt = ""

    def postprocess(self) -> str:
        self.final_nodes = [
            node_with_score
            for node_with_score in self.candidate_nodes
            if node_with_score.score >= self.threshold
            and node_with_score.node.metadata["has_code"]
        ]
        self.final_document_paths = self.unique_document_paths(self.final_nodes)
        self.context = self.build_context(self.final_document_paths)
        self.final_prompt = self.build_final_prompt()
        return self.final_prompt

    def unique_document_paths(self, nodes) -> list[str]:
        document_paths = []

        for node_with_score in nodes:
            document_path = node_with_score.node.metadata["document_filepath"]
            if document_path not in document_paths:
                document_paths.append(document_path)

        return document_paths

    def build_context(self, document_paths: list[str]) -> str:
        context_blocks = []

        for index, document_path in enumerate(document_paths, start=1):
            code_sections = self.code_nodes_for_document(document_path)
            if not code_sections:
                continue

            first_section = code_sections[0]
            block_lines = [
                f"[Context {index}]",
                f"Filepath: {document_path}",
                f"Title: {first_section.metadata['document_title']}",
            ]

            for node in code_sections:
                metadata = node.metadata
                code = "\n\n".join(metadata["code"])
                block_lines.extend(
                    [
                        f"Section: {metadata['document_section']}",
                        "",
                        code,
                    ]
                )

            context_blocks.append("\n".join(block_lines))

        return "\n\n".join(context_blocks)

    def code_nodes_for_document(self, document_path: str) -> list[TextNode]:
        return sorted(
            [
                node
                for node in self.all_nodes
                if node.metadata["document_filepath"] == document_path
                and node.metadata["has_code"]
            ],
            key=lambda node: node.metadata["document_section_index"],
        )

    def build_final_prompt(self) -> str:
        return (
            "Use the documentation context to answer the user prompt.\n\n"
            f"User prompt:\n{self.prompt}\n\n"
            f"Documentation context:\n{self.context}"
        )
