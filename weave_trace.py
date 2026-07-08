import argparse
import importlib.util
import re
import subprocess
import sys
import time
from pathlib import Path

import weave
from llama_index.core import Settings


DEFAULT_WEAVE_PROJECT = "llm-rag-opendp"
GENERATED_CODE_DIR = Path("generated_code")
TEST_DIR = Path("test")
TEST_TEMPLATE_DIR = Path("test_template")
RUN_ARGS = {
    "test_laplace.txt": {
        "csv_file": "adult_clean.csv",
        "preference": "utility-over-privacy",
    },
}


def init_weave_tracing(args: argparse.Namespace) -> None:
    if args.weave:
        weave.init(
            args.weave_project,
            settings={"implicitly_patch_integrations": False},
        )
        print(f"Weave tracing enabled for project: {args.weave_project}")


def retrieved_node_trace_records(retrieved_nodes) -> list[dict]:
    records = []

    for node_with_score in retrieved_nodes:
        metadata = node_with_score.node.metadata
        records.append(
            {
                "score": node_with_score.score,
                "filepath": metadata["document_filepath"],
                "section": metadata["document_section"],
                "has_code": metadata["has_code"],
            }
        )

    return records


def response_to_code(response) -> str:
    text = getattr(response, "text", None) or str(response)
    match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL)
    if match:
        text = match.group(1)
    return text.strip() + "\n"


def save_generated_code(
    *,
    generated_code: str,
    task_name: str,
    rag_enabled: bool,
    llm_model: str,
) -> Path:
    GENERATED_CODE_DIR.mkdir(parents=True, exist_ok=True)
    llm_name_match = re.match(r"[A-Za-z]+", llm_model)
    llm_name = llm_name_match.group(0) if llm_name_match else llm_model
    rag_suffix = "_rag" if rag_enabled else ""
    timestamp = int(time.time())
    save_code_file_path = (
        GENERATED_CODE_DIR / f"{task_name}_{llm_name}{rag_suffix}_{timestamp}.py"
    )
    save_code_file_path.write_text(generated_code, encoding="utf-8")
    return save_code_file_path


@weave.op()
def trace_code_generation(
    *,
    embed_model: str,
    llm_model: str,
    rag_enabled: bool,
    task_name: str,
    task_description: str,
    retrieved_nodes: list[dict],
    unique_retrieved_paths: list[str],
    final_prompt: str,
) -> dict:
    response = Settings.llm.complete(final_prompt)
    generated_code = response_to_code(response)
    save_code_file_path = save_generated_code(
        generated_code=generated_code,
        task_name=task_name,
        rag_enabled=rag_enabled,
        llm_model=llm_model,
    )

    return {
        "generated_code": generated_code,
        "save_code_file_path": str(save_code_file_path),
    }


@weave.op()
def trace_code_evaluation(
    *,
    generated_code: str,
    save_code_file_path: str,
    test: str,
) -> dict:
    code_file_path = Path(save_code_file_path)
    filename = code_file_path.stem
    test_template_path = TEST_TEMPLATE_DIR / test
    import_opendp = "import opendp.prelude as dp" in generated_code
    calls_opendp = re.findall(
        r"\bdp(?:\.[A-Za-z_][A-Za-z0-9_]*)+",
        generated_code,
    )
    use_opendp = any(call != "dp.enable_features" for call in calls_opendp)

    test_template = test_template_path.read_text(encoding="utf-8")
    test_code = test_template.format(filename=filename)

    TEST_DIR.mkdir(parents=True, exist_ok=True)
    test_code_file_path = TEST_DIR / f"{filename}_{test_template_path.stem}.py"
    test_code_file_path.write_text(test_code, encoding="utf-8")

    try:
        result = run_private_count(code_file_path, RUN_ARGS[test])
    except Exception as error:
        return {
            "test_code_file_path": str(test_code_file_path),
            "passed": False,
            "stdout": "",
            "stderr": repr(error),
            "import_opendp": import_opendp,
            "calls_opendp": calls_opendp,
            "use_opendp": use_opendp,
            "result": None,
        }

    test_result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_code_file_path), "-q"],
        text=True,
        capture_output=True,
    )
    passed = test_result.returncode == 0

    return {
        "test_code_file_path": str(test_code_file_path),
        "passed": passed,
        "stdout": test_result.stdout,
        "stderr": test_result.stderr,
        "import_opendp": import_opendp,
        "calls_opendp": calls_opendp,
        "use_opendp": use_opendp,
        "result": result,
    }


def run_private_count(code_file_path: Path, run_args: dict) -> dict:
    spec = importlib.util.spec_from_file_location(code_file_path.stem, code_file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.private_count(**run_args)
