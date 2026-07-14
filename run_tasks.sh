#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python}"
RAG="${RAG:-False}"
EMBED_MODEL="${EMBED_MODEL:-}"
LLM_MODEL="${LLM_MODEL:-claude-haiku-4-5}"
FRAMEWORK="${FRAMEWORK:-OpenDP}"
RUN_COUNT="${RUN_COUNT:-1}"

tasks=(
  "stat_count"
  "stat_mean"
  "stat_histogram"
  "stat_composition"
  "selection_quantile"
  "selection_topk"
  "ml_regression"
  "ml_pca"
)

for task in "${tasks[@]}"; do
  echo "Running task: ${task}"
  command=(
    "${PYTHON_BIN}" llamaindex_querying.py
    --rag "${RAG}"
    --llm "${LLM_MODEL}"
    --task "${task}"
    --framework "${FRAMEWORK}"
    --n "${RUN_COUNT}"
    --weave
  )

  if [[ -n "${EMBED_MODEL}" ]]; then
    command+=(--embed "${EMBED_MODEL}")
  fi

  "${command[@]}"
done
