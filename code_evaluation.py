import argparse
from pathlib import Path

from weave_trace import DEFAULT_WEAVE_PROJECT, trace_code_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--code-file",
        required=True,
        help="Name of the generated Python code file in the generated_code folder.",
    )
    parser.add_argument(
        "--test",
        required=True,
        help="Name of the test template file in the test_template folder.",
    )
    parser.add_argument(
        "--weave",
        action="store_true",
        help="Log the evaluation with Weave.",
    )
    parser.add_argument(
        "--weave-project",
        default=DEFAULT_WEAVE_PROJECT,
        help=f"Weave project name to use when --weave is set. Default: {DEFAULT_WEAVE_PROJECT}.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.weave:
        import weave

        weave.init(
            args.weave_project,
            settings={"implicitly_patch_integrations": False},
        )

    code_file = Path("generated_code") / args.code_file
    generated_code = code_file.read_text(encoding="utf-8")

    evaluation = trace_code_evaluation(
        generated_code=generated_code,
        save_code_file_path=str(code_file),
        test=args.test,
    )

    print("\nCode evaluation")
    print("=" * 128)
    print(f"Code file: {code_file}")
    print(f"Test template: {args.test}")
    print(f"Test code file: {evaluation['test_code_file_path']}")
    print(f"Passed: {evaluation['passed']}")
    print(f"Import OpenDP: {evaluation['import_opendp']}")
    print(f"OpenDP calls: {evaluation['calls_opendp']}")
    print(f"Use OpenDP: {evaluation['use_opendp']}")
    print(f"Result: {evaluation['result']}")
    print(f"Stdout: {evaluation['stdout']}")
    print(f"Stderr: {evaluation['stderr']}")
