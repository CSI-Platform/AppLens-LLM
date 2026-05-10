from __future__ import annotations

import argparse
import json
from pathlib import Path

from applens_llm.adrenalin_telemetry import write_hardware_summary
from applens_llm.bench import run_openai_chat_benchmark
from applens_llm.blackboard import append_event, start_experiment
from applens_llm.capture_ingest import write_capture_records_jsonl
from applens_llm.driver_evidence import collect_nvidia_driver_evidence
from applens_llm.eval import evaluate_training_examples_file, write_eval_report
from applens_llm.experiment_compare import write_experiment_comparison
from applens_llm.experiments import run_two_lane_experiment
from applens_llm.fit_report import write_fit_report
from applens_llm.lane_processes import build_server_command, start_lane, stop_lane
from applens_llm.llamacpp_probe import run_llamacpp_bench, write_llamacpp_devices
from applens_llm.orchestrator import run_lane_once
from applens_llm.runtime_lanes import get_lane, load_runtime_lanes
from applens_llm.schemas import SchemaValidationError, validate_document, validate_jsonl_file
from applens_llm.vgm_probe import compare_vgm_snapshots, write_vgm_snapshot


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
            driver_evidence = collect_nvidia_driver_evidence(driver_branch=args.nvidia_driver_branch)
            record = run_openai_chat_benchmark(
                endpoint=args.endpoint,
                model_name=args.model,
                prompt=args.prompt,
                output_path=args.output,
                max_tokens=args.max_tokens,
                timeout_seconds=args.timeout_seconds,
                engine=args.engine,
                backend=args.backend,
                model_path=args.model_path,
                quantization=args.quantization,
                driver_evidence=driver_evidence,
            )
            print(f"{args.output} valid benchmark record: {record['run_id']}")
            return 0
        if args.command == "eval":
            report = evaluate_training_examples_file(args.examples)
            write_eval_report(report, args.output)
            print(f"{report['scores']['passed']}/{report['total']} pass -> {args.output}")
            return 0
        if args.command == "ingest-captures":
            count = write_capture_records_jsonl(args.source, args.output)
            print(f"{count} capture records -> {args.output}")
            return 0
        if args.command == "vgm-snapshot":
            snapshot = write_vgm_snapshot(
                args.output,
                label=args.label,
                model_roots=args.model_root,
                llama_roots=args.llama_root,
                max_models=args.max_models,
            )
            vgm = snapshot["vgm_check"]
            readiness = snapshot["runtime_readiness"]
            print(
                f"vgm snapshot -> {args.output}; "
                f"AMD dedicated={vgm['amd_dedicated_memory_mb']} MB; "
                f"VGM 16GB active={vgm['vgm_16gb_active']}; "
                f"Vulkan llama.cpp ready={readiness['has_vulkan_llamacpp']}"
            )
            return 0
        if args.command == "vgm-compare":
            before = json.loads(args.before.read_text(encoding="utf-8"))
            after = json.loads(args.after.read_text(encoding="utf-8"))
            comparison = compare_vgm_snapshots(before, after)
            print(
                f"VGM activated: {comparison['vgm_activated']}; "
                f"AMD dedicated delta={comparison['amd_dedicated_memory_delta_mb']} MB; "
                f"next_step={comparison['next_step']}"
            )
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
                print(f"comparison -> {args.output}")
            return 0
        if args.command == "llamacpp-devices":
            devices = write_llamacpp_devices(
                binary=args.binary,
                output_path=args.output,
                from_output=args.from_output,
            )
            print(f"{len(devices)} llama.cpp devices -> {args.output}")
            return 0
        if args.command == "llamacpp-bench":
            record = run_llamacpp_bench(
                binary=args.binary,
                model=args.model,
                device=args.device,
                label=args.label,
                output_path=args.output,
                prompt_tokens=args.prompt_tokens,
                generation_tokens=args.generation_tokens,
                repetitions=args.repetitions,
                gpu_layers=args.gpu_layers,
                threads=args.threads,
                disable_vulkan_coopmat=args.disable_vulkan_coopmat,
            )
            summary = record["llamacpp"]["summary"]
            print(
                f"llama.cpp bench -> {args.output}; "
                f"device={args.device}; "
                f"prompt_tps={summary['prompt_tokens_per_second']}; "
                f"generation_tps={summary['generation_tokens_per_second']}; "
                f"returncode={record['process']['returncode']}"
            )
            return 0
        if args.command == "adrenalin-summary":
            summary = write_hardware_summary(args.input, args.output)
            metrics = summary["summary"]
            peak_memory = metrics.get("gpu_memory_utilization_mb", {}).get("max", 0)
            print(
                f"AMD Adrenalin telemetry -> {args.output}; "
                f"samples={metrics['sample_count']}; "
                f"peak_gpu_memory_mb={peak_memory}"
            )
            return 0
        if args.command == "lanes-check":
            config = load_runtime_lanes(args.config)
            print(f"{len(config['lanes'])} runtime lanes valid -> {args.config}")
            return 0
        if args.command == "blackboard-init":
            event = start_experiment(args.output, experiment_id=args.experiment_id, title=args.title)
            print(f"blackboard initialized -> {args.output}; event={event['event_id']}")
            return 0
        if args.command == "blackboard-task":
            event = append_event(
                args.output,
                experiment_id=args.experiment_id,
                event_type="task",
                payload={"task_id": args.task_id, "prompt": args.prompt, "metadata": {}},
                commit_safe=False,
            )
            print(f"task appended -> {args.output}; event={event['event_id']}")
            return 0
        if args.command == "orchestrate-once":
            config = load_runtime_lanes(args.config)
            lane = get_lane(config, args.lane)
            event = run_lane_once(
                args.blackboard,
                experiment_id=args.experiment_id,
                task_id=args.task_id,
                prompt=args.prompt,
                lane=lane,
                timeout_seconds=args.timeout_seconds,
                max_tokens=args.max_tokens,
            )
            print(
                f"orchestrated {args.lane} -> {args.blackboard}; "
                f"event_type={event['event_type']}; outcome={event['payload']['outcome']}"
            )
            return 0
        if args.command == "lane-start":
            config = load_runtime_lanes(args.config)
            lane = get_lane(config, args.lane)
            if args.dry_run:
                command = build_server_command(lane)
                print(f"lane start dry-run {args.lane}: {' '.join(command)}")
                return 0
            record = start_lane(
                lane,
                state_path=args.state,
                logs_dir=args.logs_dir,
            )
            print(
                f"lane started {args.lane}; pid={record['pid']}; "
                f"endpoint={record['endpoint']}; state={args.state}"
            )
            return 0
        if args.command == "lane-stop":
            record = stop_lane(args.lane, state_path=args.state, timeout_seconds=args.timeout_seconds)
            print(f"lane stop {args.lane}; status={record['status']}; state={args.state}")
            return 0
        if args.command == "experiment-run":
            config = load_runtime_lanes(args.config)
            driver_evidence = collect_nvidia_driver_evidence(driver_branch=args.nvidia_driver_branch)
            summary = run_two_lane_experiment(
                config=config,
                fast_lane_id=args.fast_lane,
                deep_lane_id=args.deep_lane,
                experiment_id=args.experiment_id,
                prompt=args.prompt,
                blackboard_path=args.blackboard,
                summary_path=args.summary,
                state_path=args.state,
                logs_dir=args.logs_dir,
                timeout_seconds=args.timeout_seconds,
                deep_timeout_seconds=args.deep_timeout_seconds,
                fast_max_tokens=args.fast_max_tokens,
                deep_max_tokens=args.deep_max_tokens,
                driver_evidence=driver_evidence,
                skip_start=args.skip_start,
                keep_running=args.keep_running,
            )
            print(
                f"experiment run {summary['experiment_id']} -> {args.summary}; "
                f"fast={summary['responses']['fast']['outcome']}; "
                f"deep={summary['responses']['deep']['outcome']}"
            )
            return 0
        if args.command == "experiment-compare":
            comparison = write_experiment_comparison(args.baseline, args.candidate, args.output)
            fast_delta = comparison["deltas"]["fast"]["latency_ms_delta"]
            deep_delta = comparison["deltas"]["deep"]["latency_ms_delta"]
            print(
                f"experiment compare -> {args.output}; "
                f"fast_delta_ms={fast_delta}; deep_delta_ms={deep_delta}; "
                f"verdict={comparison['verdict']}"
            )
            return 0
        if args.command == "fit-report":
            report = write_fit_report(
                machine_profile_path=args.machine_profile,
                machine_id=args.machine_id,
                output_path=args.output,
                benchmark_record_paths=args.benchmark_record,
                experiment_summary_paths=args.experiment_summary,
                experiment_comparison_paths=args.experiment_comparison,
                report_id=args.report_id,
            )
            print(
                f"fit report {report['report_id']} -> {args.output}; "
                f"class={report['fit']['class']}; "
                f"strategy={report['runtime_recommendation']['strategy']}"
            )
            return 0
    except SchemaValidationError as exc:
        print(f"schema error: {exc}")
        return 2
    except KeyError as exc:
        print(f"runtime lane error: {exc}")
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
    bench.add_argument("--engine", default="jan")
    bench.add_argument("--backend", default="unknown")
    bench.add_argument("--model-path", default="local")
    bench.add_argument("--quantization", default="unknown")
    bench.add_argument(
        "--nvidia-driver-branch",
        choices=["game_ready", "studio", "oem", "unknown"],
        default="unknown",
        help="NVIDIA driver branch reported by the NVIDIA App; version is collected with nvidia-smi.",
    )

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--examples", type=Path, required=True)
    eval_parser.add_argument("--output", type=Path, default=Path("out/eval-report.json"))

    ingest = subparsers.add_parser("ingest-captures")
    ingest.add_argument("--source", type=Path, required=True)
    ingest.add_argument("--output", type=Path, default=Path("data/raw/capture-records.jsonl"))

    vgm_snapshot = subparsers.add_parser("vgm-snapshot")
    vgm_snapshot.add_argument("--label", required=True)
    vgm_snapshot.add_argument("--output", type=Path, default=Path("out/vgm/snapshot.json"))
    vgm_snapshot.add_argument("--model-root", type=Path, action="append", default=[])
    vgm_snapshot.add_argument("--llama-root", type=Path, action="append", default=[])
    vgm_snapshot.add_argument("--max-models", type=int, default=30)

    vgm_compare = subparsers.add_parser("vgm-compare")
    vgm_compare.add_argument("--before", type=Path, required=True)
    vgm_compare.add_argument("--after", type=Path, required=True)
    vgm_compare.add_argument("--output", type=Path)

    llamacpp_devices = subparsers.add_parser("llamacpp-devices")
    llamacpp_devices.add_argument("--binary", type=Path, required=True)
    llamacpp_devices.add_argument("--output", type=Path, default=Path("out/vgm/llamacpp-devices.json"))
    llamacpp_devices.add_argument("--from-output")

    llamacpp_bench = subparsers.add_parser("llamacpp-bench")
    llamacpp_bench.add_argument("--binary", type=Path, required=True)
    llamacpp_bench.add_argument("--model", type=Path, required=True)
    llamacpp_bench.add_argument("--device", required=True)
    llamacpp_bench.add_argument("--label", required=True)
    llamacpp_bench.add_argument("--output", type=Path, required=True)
    llamacpp_bench.add_argument("--prompt-tokens", type=int, default=512)
    llamacpp_bench.add_argument("--generation-tokens", type=int, default=128)
    llamacpp_bench.add_argument("--repetitions", type=int, default=3)
    llamacpp_bench.add_argument("--gpu-layers", type=int, default=99)
    llamacpp_bench.add_argument("--threads", type=int, default=12)
    llamacpp_bench.add_argument("--disable-vulkan-coopmat", action="store_true")

    adrenalin_summary = subparsers.add_parser("adrenalin-summary")
    adrenalin_summary.add_argument("--input", type=Path, required=True)
    adrenalin_summary.add_argument("--output", type=Path, required=True)

    lanes_check = subparsers.add_parser("lanes-check")
    lanes_check.add_argument("--config", type=Path, required=True)

    blackboard_init = subparsers.add_parser("blackboard-init")
    blackboard_init.add_argument("--experiment-id", required=True)
    blackboard_init.add_argument("--title", required=True)
    blackboard_init.add_argument("--output", type=Path, required=True)

    blackboard_task = subparsers.add_parser("blackboard-task")
    blackboard_task.add_argument("--experiment-id", required=True)
    blackboard_task.add_argument("--task-id", required=True)
    blackboard_task.add_argument("--prompt", required=True)
    blackboard_task.add_argument("--output", type=Path, required=True)

    orchestrate_once = subparsers.add_parser("orchestrate-once")
    orchestrate_once.add_argument("--config", type=Path, required=True)
    orchestrate_once.add_argument("--lane", required=True)
    orchestrate_once.add_argument("--experiment-id", required=True)
    orchestrate_once.add_argument("--task-id", required=True)
    orchestrate_once.add_argument("--prompt", required=True)
    orchestrate_once.add_argument("--blackboard", type=Path, required=True)
    orchestrate_once.add_argument("--timeout-seconds", type=int, default=120)
    orchestrate_once.add_argument("--max-tokens", type=int, default=512)

    lane_start = subparsers.add_parser("lane-start")
    lane_start.add_argument("--config", type=Path, required=True)
    lane_start.add_argument("--lane", required=True)
    lane_start.add_argument("--state", type=Path, default=Path("out/runtime/lane-processes.json"))
    lane_start.add_argument("--logs-dir", type=Path, default=Path("out/logs"))
    lane_start.add_argument("--dry-run", action="store_true")

    lane_stop = subparsers.add_parser("lane-stop")
    lane_stop.add_argument("--lane", required=True)
    lane_stop.add_argument("--state", type=Path, default=Path("out/runtime/lane-processes.json"))
    lane_stop.add_argument("--timeout-seconds", type=int, default=15)

    experiment_run = subparsers.add_parser("experiment-run")
    experiment_run.add_argument("--config", type=Path, required=True)
    experiment_run.add_argument("--fast-lane", required=True)
    experiment_run.add_argument("--deep-lane", required=True)
    experiment_run.add_argument("--experiment-id", required=True)
    experiment_run.add_argument("--prompt", required=True)
    experiment_run.add_argument("--blackboard", type=Path, required=True)
    experiment_run.add_argument("--summary", type=Path, required=True)
    experiment_run.add_argument("--state", type=Path, default=Path("out/runtime/lane-processes.json"))
    experiment_run.add_argument("--logs-dir", type=Path, default=Path("out/logs"))
    experiment_run.add_argument("--timeout-seconds", type=int, default=120)
    experiment_run.add_argument("--deep-timeout-seconds", type=int, default=600)
    experiment_run.add_argument("--fast-max-tokens", type=int, default=256)
    experiment_run.add_argument("--deep-max-tokens", type=int, default=512)
    experiment_run.add_argument(
        "--nvidia-driver-branch",
        choices=["game_ready", "studio", "oem", "unknown"],
        default="unknown",
        help="NVIDIA driver branch reported by the NVIDIA App; version is collected with nvidia-smi.",
    )
    experiment_run.add_argument("--skip-start", action="store_true")
    experiment_run.add_argument("--keep-running", action="store_true")

    experiment_compare = subparsers.add_parser("experiment-compare")
    experiment_compare.add_argument("--baseline", type=Path, required=True)
    experiment_compare.add_argument("--candidate", type=Path, required=True)
    experiment_compare.add_argument("--output", type=Path, required=True)

    fit_report = subparsers.add_parser("fit-report")
    fit_report.add_argument("--machine-profile", type=Path, required=True)
    fit_report.add_argument("--machine-id")
    fit_report.add_argument("--benchmark-record", type=Path, action="append", default=[])
    fit_report.add_argument("--experiment-summary", type=Path, action="append", default=[])
    fit_report.add_argument("--experiment-comparison", type=Path, action="append", default=[])
    fit_report.add_argument("--report-id")
    fit_report.add_argument("--output", type=Path, required=True)

    return parser


if __name__ == "__main__":
    raise SystemExit(main())
