import argparse
import os
from pathlib import Path

from helper import (
    LlamaIndexRAGPreprocessor,
    RAGQueryPostprocessor,
    load_prompt,
    load_prompt_values,
    load_retrieval_query,
    print_generation_summary,
    print_prompt,
    print_retrieved_nodes,
)
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core import load_index_from_storage
from weave_trace import (
    DEFAULT_WEAVE_PROJECT,
    init_weave_tracing,
    retrieved_node_trace_records,
    trace_code_generation,
)


RAG_CORPUS_PATH = Path("rag_corpus")
KEY_PATH = Path("openai_key.txt")
RETRIEVAL_TOP_K = 6
SIMILARITY_THRESHOLD = 0.35

LOCAL_EMBED_MODEL = "nomic-embed-text"
LOCAL_LLM_MODEL = "llama3.2"

EMBED_MODELS = {
    "nomic-embed-text": {
        "provider": "ollama",
        "persist_dir": Path("storage/llamaindex_ollama_nomic_title_section"),
    },
    "text-embedding-3-large": {
        "provider": "openai",
        "persist_dir": Path("storage/llamaindex_openai"),
    },
}

LLM_MODELS = {
    "llama3.2": "ollama",
    "gpt-5.4": "openai",
}

TASKS = {
    "stat_count": {
        "prompt_path": Path("prompt_template/prompt_stat_function.txt"),
        "prompt_value_path": Path("prompt_value/prompt_stat_count_function_value.yaml"),
    },
    "selection_split": {
        "prompt_path": Path("prompt_template/prompt_selection_function.txt"),
        "prompt_value_path": Path("prompt_value/prompt_selection_split_function_value.yaml"),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indexing",
        action="store_true",
        help="Run the indexing pipeline: preprocess documents, build/load the index, and persist it.",
    )
    parser.add_argument(
        "--querying",
        action="store_true",
        help="Run the querying pipeline.",
    )
    parser.add_argument(
        "--embed",
        default=None,
        help="Embedding model to use. Examples: nomic-embed-text, text-embedding-3-large.",
    )
    parser.add_argument(
        "--llm",
        default=None,
        help="LLM model to use. Examples: llama3.2, gpt-5.4.",
    )
    parser.add_argument(
        "--task",
        default="stat_count",
        help="Task prompt/value pair to use. Examples: stat_count, selection_split.",
    )
    parser.add_argument(
        "--rag",
        default="True",
        choices=["True", "False"],
        help="Use retrieval-augmented generation. Set to False for a direct LLM call.",
    )
    parser.add_argument(
        "--weave",
        action="store_true",
        help="Trace the RAG retrieval/postprocessing step with Weave.",
    )
    parser.add_argument(
        "--weave-project",
        default=DEFAULT_WEAVE_PROJECT,
        help=f"Weave project name to use when --weave is set. Default: {DEFAULT_WEAVE_PROJECT}.",
    )
    return parser.parse_args()


def selected_embed_model(args: argparse.Namespace) -> str:
    embed_model_name = args.embed or LOCAL_EMBED_MODEL
    if embed_model_name not in EMBED_MODELS:
        raise ValueError(
            f"Unsupported embedding model: {embed_model_name}. "
            f"Choose from: {', '.join(EMBED_MODELS)}"
        )
    return embed_model_name


def selected_llm_model(args: argparse.Namespace) -> str:
    llm_model_name = args.llm or LOCAL_LLM_MODEL
    if llm_model_name not in LLM_MODELS:
        raise ValueError(
            f"Unsupported LLM model: {llm_model_name}. "
            f"Choose from: {', '.join(LLM_MODELS)}"
        )
    return llm_model_name


def configure_models(embed_model_name: str | None, llm_model_name: str) -> None:
    embed_provider = (
        EMBED_MODELS[embed_model_name]["provider"] if embed_model_name else None
    )
    llm_provider = LLM_MODELS[llm_model_name]

    if "openai" in {embed_provider, llm_provider}:
        os.environ["OPENAI_API_KEY"] = KEY_PATH.read_text(encoding="utf-8").strip()

    if embed_model_name:
        if embed_provider == "openai":
            from llama_index.embeddings.openai import OpenAIEmbedding

            Settings.embed_model = OpenAIEmbedding(model=embed_model_name)
        else:
            from llama_index.embeddings.ollama import OllamaEmbedding

            Settings.embed_model = OllamaEmbedding(model_name=embed_model_name)

    if llm_provider == "openai":
        from llama_index.llms.openai import OpenAI

        Settings.llm = OpenAI(model=llm_model_name, temperature=0.0)
    else:
        from llama_index.llms.ollama import Ollama

        Settings.llm = Ollama(model=llm_model_name, request_timeout=300.0)

    if embed_model_name:
        print(f"Using {embed_provider} embedding model: {embed_model_name}")
    print(f"Using {llm_provider} LLM model: {llm_model_name}")


def persist_dir_for_embedding(embed_model_name: str) -> Path:
    return EMBED_MODELS[embed_model_name]["persist_dir"]


def prompt_paths_for_task(task: str) -> tuple[Path, Path]:
    if task not in TASKS:
        raise ValueError(
            f"Unsupported task: {task}. Choose from: {list(TASKS.keys())}"
        )

    return TASKS[task]["prompt_path"], TASKS[task]["prompt_value_path"]


if __name__ == "__main__":
    args = parse_args()
    init_weave_tracing(args)

    llm_model_name = selected_llm_model(args)
    use_rag = args.rag == "True"
    needs_embedding = args.indexing or (args.querying and use_rag)
    embed_model_name = selected_embed_model(args) if needs_embedding else None
    configure_models(embed_model_name, llm_model_name)

    if needs_embedding:
        persist_dir = persist_dir_for_embedding(embed_model_name)
        print(f"Using persist directory: {persist_dir}")

    prompt_path, prompt_value_path = prompt_paths_for_task(args.task)
    prompt_values = load_prompt_values(prompt_value_path)
    print(f"Using task: {args.task}")

    if args.indexing:
        preprocessor = LlamaIndexRAGPreprocessor(rag_corpus_path=RAG_CORPUS_PATH)
        preprocessor.preprocess()

        print(
            f"Loaded {len(preprocessor.documents)} documents, "
            f"converted {len(preprocessor.document_sections)} document sections "
            f"into {len(preprocessor.nodes)} LlamaIndex nodes"
        )

        # preprocessor.print_document_nodes(document_name_or_path="rag_corpus/getting-started/tabular-data/essential-statistics.txt")

        if persist_dir.exists():
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            index = load_index_from_storage(storage_context)
            print(f"Loaded existing index from {persist_dir}")
        else:
            index = VectorStoreIndex(preprocessor.nodes, show_progress=True)
            index.storage_context.persist(persist_dir=persist_dir)
            print(f"Built and persisted index to {persist_dir}")

    if args.querying:
        prompt = load_prompt(prompt_path, prompt_value_path)
        print_prompt(prompt)

        if not use_rag:
            generation = trace_code_generation(
                embed_model=embed_model_name,
                llm_model=llm_model_name,
                rag_enabled=use_rag,
                task_name=prompt_values["query_type"],
                task_description=prompt_values["query_definition"],
                retrieved_nodes=[],
                unique_retrieved_paths=[],
                final_prompt=prompt,
            )
            print_generation_summary(generation)
        else:
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            index = load_index_from_storage(storage_context)
            print(f"Loaded existing index from {persist_dir} for querying")

            retrieval_query = load_retrieval_query(prompt_value_path)
            print_prompt(retrieval_query, title="Retrieval query")

            retriever = index.as_retriever(similarity_top_k=RETRIEVAL_TOP_K)
            retrieved_nodes = retriever.retrieve(retrieval_query)
            print_retrieved_nodes(retrieved_nodes)

            postprocessor = RAGQueryPostprocessor(
                prompt=prompt,
                candidate_nodes=retrieved_nodes,
                all_nodes=index.docstore.docs.values(),
                threshold=SIMILARITY_THRESHOLD,
            )
            final_prompt = postprocessor.postprocess()

            print(f"\nFinal context nodes: {len(postprocessor.final_nodes)}")
            print_prompt(final_prompt)

            generation = trace_code_generation(
                embed_model=embed_model_name,
                llm_model=llm_model_name,
                rag_enabled=use_rag,
                task_name=prompt_values["query_type"],
                task_description=prompt_values["query_definition"],
                retrieved_nodes=retrieved_node_trace_records(retrieved_nodes),
                unique_retrieved_paths=postprocessor.final_document_paths,
                final_prompt=final_prompt,
            )
            print_generation_summary(generation)
