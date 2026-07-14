import argparse

from helper import (
    RAG_CORPUS_PATH,
    LlamaIndexRAGPreprocessor,
    configure_models,
    persist_dir_for_embedding,
    selected_embed_model,
)
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core import load_index_from_storage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embed",
        default=None,
        help="Embedding model to use. Examples: nomic-embed-text, text-embedding-3-large.",
    )
    return parser.parse_args()


def run_indexing(args: argparse.Namespace) -> None:
    embed_model_name = selected_embed_model(args)
    configure_models(embed_model_name=embed_model_name)

    persist_dir = persist_dir_for_embedding(embed_model_name)
    print(f"Using persist directory: {persist_dir}")

    if persist_dir.exists():
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        load_index_from_storage(storage_context)
        print(f"Loaded existing index from {persist_dir}")
        return

    preprocessor = LlamaIndexRAGPreprocessor(rag_corpus_path=RAG_CORPUS_PATH)
    preprocessor.preprocess()

    print(
        f"Loaded {len(preprocessor.documents)} documents, "
        f"converted {len(preprocessor.document_sections)} document sections "
        f"into {len(preprocessor.nodes)} LlamaIndex nodes"
    )

    index = VectorStoreIndex(preprocessor.nodes, show_progress=True)
    index.storage_context.persist(persist_dir=persist_dir)
    print(f"Built and persisted index to {persist_dir}")


def main() -> None:
    run_indexing(parse_args())


if __name__ == "__main__":
    main()
