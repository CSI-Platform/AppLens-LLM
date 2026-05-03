from __future__ import annotations

import argparse
from pathlib import Path

from applens_llm.bench import run_openai_chat_benchmark
from applens_llm.eval import evaluate_training_examples_file, write_eval_report
from applens_llm.schemas import SchemaValidationError, validate_document, validate_jsonl_file


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "validate":
            validate_document(args.schema, args.path)
            print(f"{args.path} valid")
            return 0
        if args.command == "validate-jsonl":
            rows = validate_jsonl_file(args.schema, args.path)
            print(f"{args.path} {len(rows)} rows valid")
            return 0
        if args.command == "bench":
            record = run_openai_chat_benchmark(
                endpoint=args.endpoint,
                model_name=args.model,
                prompt=args.prompt,
                output_path=args.output,
                max_tokens=args.max_tokens,
                timeout_seconds=args.timeout_seconds,
                backend=args.backend,
                model_path=args.model_path,
                quantization=args.quantization,
            )
            print(f"{args.output} valid benchmark record: {record['run_id']}")
            return 0
        if args.command == "eval":
            report = evaluate_training_examples_file(args.examples)
            write_eval_report(report, args.output)
            print(f"{report['scores']['passed']}/{report['total']} pass -> {args.output}")
            return 0
    except SchemaValidationError as exc:
        print(f"schema error: {exc}")
        return 2
    except OSError as exc:
        print(f"benchmark error: {exc}")
        return 3

    parser.print_help()
    return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="applens-llm")
    subparsers = parser.add_subparsers(dest="command")

    validate = subparsers.add_parser("validate")
    validate.add_argument("--schema", required=True)
    validate.add_argument("path", type=Path)

    validate_jsonl = subparsers.add_parser("validate-jsonl")
    validate_jsonl.add_argument("--schema", required=True)
    validate_jsonl.add_argument("path", type=Path)

    bench = subparsers.add_parser("bench")
    bench.add_argument("--endpoint", default="http://127.0.0.1:1337/v1")
    bench.add_argument("--model", required=True)
    bench.add_argument("--prompt", default="Return a compact JSON health check for AppLens-LLM.")
    bench.add_argument("--max-tokens", type=int, default=256)
    bench.add_argument("--timeout-seconds", type=int, default=120)
    bench.add_argument("--output", type=Path, default=Path("out/benchmark-record.json"))
    bench.add_argument("--backend", default="jan")
    bench.add_argument("--model-path", default="local")
    bench.add_argument("--quantization", default="unknown")

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--examples", type=Path, required=True)
    eval_parser.add_argument("--output", type=Path, default=Path("out/eval-report.json"))

    return parser


if __name__ == "__main__":
    raise SystemExit(main())
