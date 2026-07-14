import os
import re
from pathlib import Path
from pprint import pprint
from typing import Any

from llama_index.core import Settings
from llama_index.core.schema import TextNode
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI

from metadata import extract_document_section_records_from_file

RAG_CORPUS_PATH = Path("rag_corpus")
PROMPT_DIR = Path("prompts")
DP_FRAMEWORK_REQUIREMENT_PATTERN = re.compile(
    r"^4\. Use the following DP framework: .+\.$",
    flags=re.MULTILINE,
)
API_KEY_PATHS = {
    "anthropic": Path("anthropic_key.txt"),
    "gemini": Path("gemini_key.txt"),
    "openai": Path("openai_key.txt"),
}
API_KEY_ENV_VARS = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "gemini": ("GOOGLE_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
}
RETRIEVAL_TOP_K = 6
SIMILARITY_THRESHOLD = 0.35
MAX_OUTPUT_TOKENS = 4096

LOCAL_EMBED_MODEL = "nomic-embed-text"
LOCAL_LLM_MODEL = "llama3.2"

EMBED_MODELS = {
    "nomic-embed-text": {
        "provider": "ollama",
        "persist_dir": Path("storage/llamaindex_ollama_nomic"),
    },
    "text-embedding-3-large": {
        "provider": "openai",
        "persist_dir": Path("storage/llamaindex_openai"),
    },
}

LLM_MODELS = {
    "llama3.2": "ollama",
    "gpt-5.4": "openai",
    "gpt-5.4-nano": "openai",
    "gemini-3.1-flash-lite": "gemini",
    "claude-haiku-4-5": "anthropic",
    "qwen2.5-coder:7b": "ollama",
    "codellama:7b": "ollama",
}


def selected_embed_model(args) -> str:
    embed_model_name = args.embed or LOCAL_EMBED_MODEL
    if embed_model_name not in EMBED_MODELS:
        raise ValueError(
            f"Unsupported embedding model: {embed_model_name}. "
            f"Choose from: {', '.join(EMBED_MODELS)}"
        )
    return embed_model_name


def selected_llm_model(args) -> str:
    llm_model_name = args.llm or LOCAL_LLM_MODEL
    if llm_model_name not in LLM_MODELS:
        raise ValueError(
            f"Unsupported LLM model: {llm_model_name}. "
            f"Choose from: {', '.join(LLM_MODELS)}"
        )
    return llm_model_name


def configure_models(
    embed_model_name: str | None = None,
    llm_model_name: str | None = None,
) -> None:
    embed_provider = (
        EMBED_MODELS[embed_model_name]["provider"] if embed_model_name else None
    )
    llm_provider = LLM_MODELS[llm_model_name] if llm_model_name else None

    for provider in {embed_provider, llm_provider}:
        load_api_key(provider)

    embed_model_builders = {
        "ollama": lambda model_name: OllamaEmbedding(model_name=model_name),
        "openai": lambda model_name: OpenAIEmbedding(model=model_name),
    }
    llm_builders = {
        "anthropic": build_anthropic_llm,
        "gemini": build_gemini_llm,
        "ollama": build_ollama_llm,
        "openai": build_openai_llm,
    }

    if embed_model_name:
        Settings.embed_model = embed_model_builders[embed_provider](embed_model_name)

    if llm_model_name:
        Settings.llm = llm_builders[llm_provider](llm_model_name)

    if embed_model_name:
        print(f"Using {embed_provider} embedding model: {embed_model_name}")
    if llm_model_name:
        print(f"Using {llm_provider} LLM model: {llm_model_name}")


def load_api_key(provider: str | None) -> None:
    if provider not in API_KEY_PATHS:
        return

    api_key = API_KEY_PATHS[provider].read_text(encoding="utf-8").strip()
    for env_var in API_KEY_ENV_VARS[provider]:
        os.environ[env_var] = api_key


def build_openai_llm(model_name: str):
    return OpenAI(model=model_name, temperature=0.0, max_tokens=MAX_OUTPUT_TOKENS)


def build_ollama_llm(model_name: str):
    return Ollama(
        model=model_name,
        temperature=0.0,
        request_timeout=300.0,
        additional_kwargs={"num_predict": MAX_OUTPUT_TOKENS},
    )


def build_gemini_llm(model_name: str):
    return GoogleGenAI(
        model=model_name,
        temperature=0.0,
        max_tokens=MAX_OUTPUT_TOKENS,
    )


def build_anthropic_llm(model_name: str):
    return Anthropic(model=model_name, temperature=0.0, max_tokens=MAX_OUTPUT_TOKENS)


def persist_dir_for_embedding(embed_model_name: str) -> Path:
    return EMBED_MODELS[embed_model_name]["persist_dir"]


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
        section_text = document_section_record["text"].strip()
        node_text = (
            f"Document title: {metadata['document_title']}\n"
            f"Section: {metadata['document_section']}\n\n"
            f"{section_text}"
        )
        return TextNode(
            text=node_text,
            metadata=metadata,
            id_=self._build_node_id(metadata),
            excluded_embed_metadata_keys=sorted(metadata),
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


def load_prompt(task: str, framework: str = "OpenDP") -> str:
    prompt_path = PROMPT_DIR / f"{task}.txt"
    if not prompt_path.is_file():
        raise ValueError(f"Unsupported task: {task}. Choose from: {available_prompts()}")

    prompt = prompt_path.read_text(encoding="utf-8")
    return set_prompt_framework(prompt, framework)


def set_prompt_framework(prompt: str, framework: str) -> str:
    prompt, replacement_count = DP_FRAMEWORK_REQUIREMENT_PATTERN.subn(
        f"4. Use the following DP framework: {framework}.",
        prompt,
        count=1,
    )
    if replacement_count == 0:
        raise ValueError("Prompt is missing requirement 4 for the DP framework.")

    return prompt


def available_prompts() -> list[str]:
    return sorted(prompt_path.stem for prompt_path in PROMPT_DIR.glob("*.txt"))


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
