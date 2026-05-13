from __future__ import annotations

import argparse
import json
from pathlib import Path

from applens_llm.adrenalin_telemetry import write_hardware_summary
from applens_llm.autoresearch_eval import run_autoresearch_eval
from applens_llm.autoresearch_layout import init_workload_layout
from applens_llm.autoresearch_manifest import write_self_fit_result
from applens_llm.autoresearch_memory import promote_memory
from applens_llm.autoresearch_runner import run_autoresearch_once
from applens_llm.bench import run_openai_chat_benchmark
from applens_llm.benchmark_suite import write_benchmark_suite_run
from applens_llm.blackboard import append_event, start_experiment
from applens_llm.capture_ingest import write_capture_records_jsonl
from applens_llm.context_envelope import write_context_envelope
from applens_llm.context_quality_probe import run_llamacpp_context_quality_probe, write_context_quality_record
from applens_llm.deployment_plan import write_deployment_plan
from applens_llm.driver_evidence import collect_nvidia_driver_evidence
from applens_llm.eval import evaluate_training_examples_file, write_eval_report
from applens_llm.experiment_compare import write_experiment_comparison
from applens_llm.experiments import run_two_lane_experiment
from applens_llm.fit_report import write_fit_report
from applens_llm.lane_processes import build_server_command, start_lane, stop_lane
from applens_llm.llamacpp_probe import run_llamacpp_bench, write_llamacpp_devices
from applens_llm.local_capability_eval import run_local_capability_eval, write_local_capability_record
from applens_llm.model_fit_scorecard import write_model_fit_scorecard
from applens_llm.orchestrator import run_lane_once
from applens_llm.overnight_loop import load_loop_prompts, run_overnight_loop
from applens_llm.runtime_lanes import get_lane, load_runtime_lanes
from applens_llm.schemas import SchemaValidationError, validate_document, validate_jsonl_file
from applens_llm.scorecard_html import write_scorecard_html
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
        if args.command == "autoresearch" and args.autoresearch_command == "init":
            paths = init_workload_layout(args.workload_root, workload_id=args.workload_id, display_name=args.display_name)
            print(f"autoresearch layout -> {paths.applens_dir}")
            return 0
        if args.command == "autoresearch" and args.autoresearch_command == "self-fit":
            result = write_self_fit_result(
                args.workload_root,
                machine_fingerprint=args.machine_fingerprint,
                runtime_fingerprint=args.runtime_fingerprint,
                status=args.status,
            )
            print(f"autoresearch self-fit -> status={result['status']}")
            return 0
        if args.command == "autoresearch" and args.autoresearch_command == "run":
            summary = run_autoresearch_once(
                args.workload_root,
                manifest_path=args.manifest,
                skip_self_fit=args.skip_self_fit,
            )
            print(f"autoresearch run -> outcome={summary['outcome']}")
            return 0
        if args.command == "autoresearch" and args.autoresearch_command == "eval":
            summary = run_autoresearch_eval(args.workload_root, run_id=args.run_id)
            print(
                f"autoresearch eval -> probes={summary['probes']} cases={summary['cases']} "
                f"passed={summary['passed']} failed={summary['failed']}"
            )
            return 0
        if args.command == "autoresearch" and args.autoresearch_command == "promote-memory":
            target = promote_memory(args.workload_root, args.proposal)
            print(f"memory promoted -> {target}")
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
        if args.command == "benchmark-suite-plan":
            suite = write_benchmark_suite_run(
                output_path=args.output,
                suite_run_id=args.suite_run_id,
                suite_id=args.suite_id,
                scoring_mode=args.scoring_mode,
                run_intent="plan_only",
                repeats=args.repeats,
                timeout_seconds_per_task=args.timeout_seconds_per_task,
                allow_downloads=args.allow_downloads,
                allow_network=args.allow_network,
                expected_duration_minutes=args.expected_duration_minutes,
                artifact_root=args.artifact_root,
                model={
                    "model_id": args.model_id,
                    "display_name": args.display_name or args.model_id,
                    "family": args.family,
                    "parameter_size_b": args.parameter_size_b,
                    "quantization": args.quantization,
                    "model_format": args.model_format,
                    "path": args.model_path,
                    "sha256": args.model_sha256,
                    "chat_template": args.chat_template,
                    "thinking_mode": args.thinking_mode,
                    "reasoning_mode": args.reasoning_mode,
                },
                machine_condition={
                    "condition_id": args.condition_id,
                    "label": args.condition_label,
                    "os_family": args.os_family,
                    "ram_gb": args.ram_gb,
                    "vgm_state": {
                        "enabled": args.vgm_enabled,
                        "dedicated_mb": args.vgm_dedicated_mb,
                        "system_ram_available_gb": args.system_ram_available_gb,
                        "source": args.vgm_source,
                    },
                    "accelerator_ids": args.accelerator_id,
                    "required_preflight": args.required_preflight,
                    "evidence_paths": args.evidence_path,
                },
                runtime_lane={
                    "engine": args.engine,
                    "backend": args.backend,
                    "device_selector": args.device_selector,
                    "accelerator_ids": args.runtime_accelerator_id or args.accelerator_id,
                    "endpoint": args.endpoint,
                    "context_tokens": args.context_tokens,
                    "batch_size": args.batch_size,
                    "ubatch_size": args.ubatch_size,
                    "threads": args.threads,
                    "gpu_layers": args.gpu_layers,
                    "kv_cache_type": args.kv_cache_type,
                    "flash_attention": args.flash_attention,
                    "extra_flags": args.extra_flag,
                },
            )
            print(
                f"benchmark suite {suite['suite_run_id']} -> {args.output}; "
                f"suite={suite['suite']['suite_id']}; tasks={len(suite['benchmark_plan']['tasks'])}"
            )
            return 0
        if args.command == "eval":
            report = evaluate_training_examples_file(args.examples)
            write_eval_report(report, args.output)
            print(f"{report['scores']['passed']}/{report['total']} pass -> {args.output}")
            return 0
        if args.command == "local-capability-eval":
            if args.responses:
                record = write_local_capability_record(
                    responses_path=args.responses,
                    output_path=args.output,
                    thinking_mode=args.thinking_mode,
                    execute_code_checks=args.execute_code_checks,
                )
            else:
                record = run_local_capability_eval(
                    endpoint=args.endpoint,
                    model={
                        "model_id": args.model,
                        "display_name": args.display_name or args.model,
                        "family": args.family,
                        "parameter_size_b": args.parameter_size_b,
                        "quantization": args.quantization,
                    },
                    runtime={
                        "engine": args.engine,
                        "backend": args.backend,
                        "devices_used": args.device or ["unknown-accelerator-0"],
                    },
                    output_path=args.output,
                    thinking_mode=args.thinking_mode,
                    max_tokens=args.max_tokens,
                    timeout_seconds=args.timeout_seconds,
                    execute_code_checks=args.execute_code_checks,
                    thinking_control=args.thinking_control,
                )
            print(
                f"local capability eval {record['run_id']} -> {args.output}; "
                f"score={record['scores']['score_pct']}; band={record['outcome']['band']}"
            )
            return 0
        if args.command == "ingest-captures":
            count = write_capture_records_jsonl(args.source, args.output)
            print(f"{count} capture records -> {args.output}")
            return 0
        if args.command == "context-envelope":
            envelope = write_context_envelope(
                machine_profile_path=args.machine_profile,
                machine_id=args.machine_id,
                model_candidates_path=args.model_candidates,
                context_observation_paths=args.context_observation,
                envelope_id=args.envelope_id,
                output_path=args.output,
            )
            evidence_models = sum(
                1 for model in envelope["models"] if model.get("context_evidence_status") != "advertised_unproven"
            )
            useful_models = sum(1 for model in envelope["models"] if model["max_recommended_context_tokens"] > 0)
            print(
                f"context envelope {envelope['envelope_id']} -> {args.output}; "
                f"context_evidence_models={evidence_models}; useful_context_models={useful_models}"
            )
            return 0
        if args.command == "context-quality-probe":
            model = {
                "model_id": args.model_id,
                "display_name": args.display_name or args.model_id,
                "family": args.family,
                "parameter_size_b": args.parameter_size_b,
                "quantization": args.quantization,
            }
            runtime = {
                "engine": "llama.cpp",
                "backend": args.backend,
                "device_selector": args.device_selector,
                "devices_used": args.accelerator_id,
            }
            if args.response_file:
                record = write_context_quality_record(
                    response_path=args.response_file,
                    output_path=args.output,
                    observation_output_path=args.context_observation_output,
                    model=model,
                    runtime=runtime,
                    context_tokens=args.context_tokens,
                    prompt_token_budget=args.prompt_token_budget,
                    expected_needle=args.expected_needle,
                    max_tokens=args.max_tokens,
                    elapsed_seconds=args.elapsed_seconds,
                    process_returncode=args.process_returncode,
                    prompt_tokens_per_second=args.prompt_tokens_per_second,
                    generation_tokens_per_second=args.generation_tokens_per_second,
                    execute_code_checks=args.execute_code_checks,
                )
            else:
                if not args.binary or not args.gguf_model:
                    raise ValueError("--binary and --gguf-model are required unless --response-file is provided")
                record = run_llamacpp_context_quality_probe(
                    binary=args.binary,
                    gguf_model=args.gguf_model,
                    model=model,
                    runtime=runtime,
                    context_tokens=args.context_tokens,
                    prompt_token_budget=args.prompt_token_budget,
                    output_path=args.output,
                    observation_output_path=args.context_observation_output,
                    response_output_path=args.response_output,
                    max_tokens=args.max_tokens,
                    gpu_layers=args.gpu_layers,
                    threads=args.threads,
                    disable_vulkan_coopmat=args.disable_vulkan_coopmat,
                    execute_code_checks=args.execute_code_checks,
                )
            print(
                f"context quality {record['run_id']} -> {args.output}; "
                f"score={record['scores']['quality_score_pct']}; "
                f"status={record['outcome']['status']}; "
                f"observation_status={record['observation']['status']}"
            )
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
                machine_profile=_load_json(args.machine_profile) if args.machine_profile else None,
                devices_inventory=_load_json(args.llamacpp_devices) if args.llamacpp_devices else None,
                benchmark_record_path=args.benchmark_record,
                model_name=args.model_name,
                quantization=args.quantization,
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
        if args.command == "overnight-loop":
            config = load_runtime_lanes(args.config)
            prompts = load_loop_prompts(prompt_args=args.prompt, prompt_file=args.prompt_file)
            driver_evidence = collect_nvidia_driver_evidence(driver_branch=args.nvidia_driver_branch)
            summary = run_overnight_loop(
                config=config,
                fast_lane_id=args.fast_lane,
                deep_lane_id=args.deep_lane,
                experiment_id=args.experiment_id,
                prompts=prompts,
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
                continue_on_failure=args.continue_on_failure,
                max_iterations=args.max_iterations,
                max_runtime_minutes=args.max_runtime_minutes,
                sleep_seconds=args.sleep_seconds,
            )
            print(
                f"overnight loop {summary['experiment_id']} -> {args.summary}; "
                f"completed={summary['completed_iterations']}; "
                f"attempted={summary['attempted_iterations']}; "
                f"stop_reason={summary['stop_reason']}"
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
        if args.command == "model-fit-scorecard":
            scorecard = write_model_fit_scorecard(
                machine_profile_path=args.machine_profile,
                machine_id=args.machine_id,
                model_candidates_path=args.model_candidates,
                benchmark_record_paths=args.benchmark_record,
                experiment_summary_paths=args.experiment_summary,
                capability_record_paths=args.capability_record,
                context_envelope_paths=args.context_envelope,
                workload_profile_path=args.workload_profile,
                scorecard_id=args.scorecard_id,
                output_path=args.output,
            )
            top = scorecard["rankings"][0]
            print(
                f"model fit scorecard {scorecard['scorecard_id']} -> {args.output}; "
                f"top={top['model_id']}; score={top['fit_score']}"
            )
            return 0
        if args.command == "deployment-plan":
            plan = write_deployment_plan(
                scorecard_path=args.scorecard,
                output_path=args.output,
                plan_id=args.plan_id,
                workload_name=args.workload_name,
                workload_intent=args.workload_intent,
            )
            print(
                f"deployment plan {plan['plan_id']} -> {args.output}; "
                f"primary={plan['recommended_runtime']['model']}"
            )
            return 0
        if args.command == "model-fit-html":
            write_scorecard_html(
                scorecard_path=args.scorecard,
                output_path=args.output,
                experiment_comparison_paths=args.experiment_comparison,
                title=args.title,
            )
            print(f"model fit html -> {args.output}")
            return 0
    except SchemaValidationError as exc:
        print(f"schema error: {exc}")
        return 2
    except KeyError as exc:
        print(f"runtime lane error: {exc}")
        return 2
    except ValueError as exc:
        print(f"configuration error: {exc}")
        return 2
    except OSError as exc:
        print(f"benchmark error: {exc}")
        return 3

    parser.print_help()
    return 1


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="applens-llm")
    subparsers = parser.add_subparsers(dest="command")

    validate = subparsers.add_parser("validate")
    validate.add_argument("--schema", required=True)
    validate.add_argument("path", type=Path)

    validate_jsonl = subparsers.add_parser("validate-jsonl")
    validate_jsonl.add_argument("--schema", required=True)
    validate_jsonl.add_argument("path", type=Path)

    autoresearch = subparsers.add_parser("autoresearch")
    autoresearch_sub = autoresearch.add_subparsers(dest="autoresearch_command")

    autoresearch_init = autoresearch_sub.add_parser("init")
    autoresearch_init.add_argument("--workload-root", type=Path, required=True)
    autoresearch_init.add_argument("--workload-id", required=True)
    autoresearch_init.add_argument("--display-name", required=True)

    autoresearch_self_fit = autoresearch_sub.add_parser("self-fit")
    autoresearch_self_fit.add_argument("--workload-root", type=Path, required=True)
    autoresearch_self_fit.add_argument("--machine-fingerprint", required=True)
    autoresearch_self_fit.add_argument("--runtime-fingerprint", required=True)
    autoresearch_self_fit.add_argument("--status", choices=["passed", "failed"], default="passed")

    autoresearch_run = autoresearch_sub.add_parser("run")
    autoresearch_run.add_argument("--workload-root", type=Path, required=True)
    autoresearch_run.add_argument("--manifest", type=Path, required=True)
    autoresearch_run.add_argument("--skip-self-fit", action="store_true")

    autoresearch_eval = autoresearch_sub.add_parser("eval")
    autoresearch_eval.add_argument("--workload-root", type=Path, required=True)
    autoresearch_eval.add_argument("--run-id", default="autoresearch-eval")

    autoresearch_promote = autoresearch_sub.add_parser("promote-memory")
    autoresearch_promote.add_argument("--workload-root", type=Path, required=True)
    autoresearch_promote.add_argument("--proposal", type=Path, required=True)

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

    benchmark_suite = subparsers.add_parser("benchmark-suite-plan")
    benchmark_suite.add_argument("--suite-run-id", required=True)
    benchmark_suite.add_argument("--suite-id", choices=["tiny-v1", "small-v1"])
    benchmark_suite.add_argument("--scoring-mode", choices=["local_screening", "certification"], default="local_screening")
    benchmark_suite.add_argument("--model-id", required=True)
    benchmark_suite.add_argument("--display-name")
    benchmark_suite.add_argument("--family", default="unknown")
    benchmark_suite.add_argument("--parameter-size-b", type=float, required=True)
    benchmark_suite.add_argument("--quantization", default="unknown")
    benchmark_suite.add_argument("--model-format", choices=["gguf", "safetensors", "ollama", "jan", "unknown"], default="unknown")
    benchmark_suite.add_argument("--model-path", default="local")
    benchmark_suite.add_argument("--model-sha256", default="unknown")
    benchmark_suite.add_argument("--chat-template", default="unknown")
    benchmark_suite.add_argument(
        "--thinking-mode",
        choices=["on", "off", "auto", "unsupported", "unknown"],
        default="unknown",
    )
    benchmark_suite.add_argument(
        "--reasoning-mode",
        choices=["on", "off", "auto", "unsupported", "unknown"],
        default="unknown",
    )
    benchmark_suite.add_argument("--condition-id", required=True)
    benchmark_suite.add_argument("--condition-label", required=True)
    benchmark_suite.add_argument("--os-family", default="unknown")
    benchmark_suite.add_argument("--ram-gb", type=int, required=True)
    benchmark_suite.add_argument("--vgm-enabled", action="store_true")
    benchmark_suite.add_argument("--vgm-dedicated-mb", type=int, default=0)
    benchmark_suite.add_argument("--system-ram-available-gb", type=int, required=True)
    benchmark_suite.add_argument("--vgm-source", default="unknown")
    benchmark_suite.add_argument("--accelerator-id", action="append", required=True)
    benchmark_suite.add_argument("--required-preflight", action="append", default=["close_competing_llm_apps", "record_power_mode"])
    benchmark_suite.add_argument("--evidence-path", action="append", default=[])
    benchmark_suite.add_argument("--engine", default="llama.cpp")
    benchmark_suite.add_argument(
        "--backend",
        choices=["cuda", "vulkan", "rocm", "hip", "directml", "metal", "openvino", "cpu", "npu", "mixed", "unknown"],
        default="unknown",
    )
    benchmark_suite.add_argument("--device-selector", required=True)
    benchmark_suite.add_argument("--runtime-accelerator-id", action="append", default=[])
    benchmark_suite.add_argument("--endpoint", default="http://127.0.0.1:18080/v1")
    benchmark_suite.add_argument("--context-tokens", type=int, required=True)
    benchmark_suite.add_argument("--batch-size", type=int, default=2048)
    benchmark_suite.add_argument("--ubatch-size", type=int, default=512)
    benchmark_suite.add_argument("--threads", type=int, default=12)
    benchmark_suite.add_argument("--gpu-layers", type=int, default=99)
    benchmark_suite.add_argument("--kv-cache-type", choices=["f16", "q8_0", "q4_0", "auto", "unknown"], default="auto")
    benchmark_suite.add_argument("--flash-attention", choices=["on", "off", "auto", "unsupported", "unknown"], default="auto")
    benchmark_suite.add_argument("--extra-flag", action="append", default=[])
    benchmark_suite.add_argument("--repeats", type=int, default=1)
    benchmark_suite.add_argument("--timeout-seconds-per-task", type=int, default=900)
    benchmark_suite.add_argument("--allow-downloads", action="store_true")
    benchmark_suite.add_argument("--allow-network", action="store_true")
    benchmark_suite.add_argument("--expected-duration-minutes", type=int)
    benchmark_suite.add_argument("--artifact-root")
    benchmark_suite.add_argument("--output", type=Path, required=True)

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--examples", type=Path, required=True)
    eval_parser.add_argument("--output", type=Path, default=Path("out/eval-report.json"))

    local_capability = subparsers.add_parser("local-capability-eval")
    local_capability.add_argument("--responses", type=Path, help="Offline response JSON with model, runtime, and responses.")
    local_capability.add_argument("--endpoint", default="http://127.0.0.1:18080/v1")
    local_capability.add_argument("--model", default="local-model")
    local_capability.add_argument("--display-name")
    local_capability.add_argument("--family", default="unknown")
    local_capability.add_argument("--parameter-size-b", type=float, default=0)
    local_capability.add_argument("--quantization", default="unknown")
    local_capability.add_argument(
        "--thinking-mode",
        choices=["on", "off", "auto", "unsupported", "unknown"],
        default="unknown",
    )
    local_capability.add_argument(
        "--thinking-control",
        choices=["metadata_only", "chat_template_kwargs"],
        default="metadata_only",
        help="Record thinking mode by default; only send runtime-specific controls when explicitly requested.",
    )
    local_capability.add_argument("--engine", default="llama.cpp")
    local_capability.add_argument("--backend", default="unknown")
    local_capability.add_argument("--device", action="append", default=[])
    local_capability.add_argument("--max-tokens", type=int, default=512)
    local_capability.add_argument("--timeout-seconds", type=int, default=180)
    local_capability.add_argument("--execute-code-checks", action="store_true")
    local_capability.add_argument("--output", type=Path, default=Path("out/local-capability/applens-local-v1.json"))

    ingest = subparsers.add_parser("ingest-captures")
    ingest.add_argument("--source", type=Path, required=True)
    ingest.add_argument("--output", type=Path, default=Path("data/raw/capture-records.jsonl"))

    context_envelope = subparsers.add_parser("context-envelope")
    context_envelope.add_argument("--machine-profile", type=Path, required=True)
    context_envelope.add_argument("--machine-id")
    context_envelope.add_argument("--model-candidates", type=Path, required=True)
    context_envelope.add_argument("--context-observation", type=Path, action="append", default=[])
    context_envelope.add_argument("--envelope-id")
    context_envelope.add_argument("--output", type=Path, required=True)

    context_quality = subparsers.add_parser("context-quality-probe")
    context_quality.add_argument("--response-file", type=Path)
    context_quality.add_argument("--binary", type=Path)
    context_quality.add_argument("--gguf-model", type=Path)
    context_quality.add_argument("--model-id", required=True)
    context_quality.add_argument("--display-name")
    context_quality.add_argument("--family", default="unknown")
    context_quality.add_argument("--parameter-size-b", type=float, default=0)
    context_quality.add_argument("--quantization", default="unknown")
    context_quality.add_argument("--backend", default="vulkan")
    context_quality.add_argument("--device-selector", required=True)
    context_quality.add_argument("--accelerator-id", action="append", required=True)
    context_quality.add_argument("--context-tokens", type=int, required=True)
    context_quality.add_argument("--prompt-token-budget", type=int, required=True)
    context_quality.add_argument("--expected-needle", default="APPLENS-CTX-MANUAL")
    context_quality.add_argument("--max-tokens", type=int, default=256)
    context_quality.add_argument("--elapsed-seconds", type=float, default=0)
    context_quality.add_argument("--process-returncode", type=int, default=0)
    context_quality.add_argument("--prompt-tokens-per-second", type=float, default=0)
    context_quality.add_argument("--generation-tokens-per-second", type=float)
    context_quality.add_argument("--gpu-layers", type=int, default=99)
    context_quality.add_argument("--threads", type=int, default=12)
    context_quality.add_argument("--disable-vulkan-coopmat", action="store_true")
    context_quality.add_argument("--execute-code-checks", action="store_true")
    context_quality.add_argument("--response-output", type=Path)
    context_quality.add_argument("--context-observation-output", type=Path)
    context_quality.add_argument("--output", type=Path, required=True)

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
    llamacpp_bench.add_argument("--machine-profile", type=Path)
    llamacpp_bench.add_argument("--llamacpp-devices", type=Path)
    llamacpp_bench.add_argument("--benchmark-record", type=Path)
    llamacpp_bench.add_argument("--model-name")
    llamacpp_bench.add_argument("--quantization", default="unknown")

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

    overnight_loop = subparsers.add_parser("overnight-loop")
    overnight_loop.add_argument("--config", type=Path, required=True)
    overnight_loop.add_argument("--fast-lane", required=True)
    overnight_loop.add_argument("--deep-lane", required=True)
    overnight_loop.add_argument("--experiment-id", required=True)
    overnight_loop.add_argument("--prompt", action="append", default=[])
    overnight_loop.add_argument("--prompt-file", type=Path)
    overnight_loop.add_argument("--blackboard", type=Path, required=True)
    overnight_loop.add_argument("--summary", type=Path, required=True)
    overnight_loop.add_argument("--state", type=Path, default=Path("out/runtime/lane-processes.json"))
    overnight_loop.add_argument("--logs-dir", type=Path, default=Path("out/logs"))
    overnight_loop.add_argument("--timeout-seconds", type=int, default=120)
    overnight_loop.add_argument("--deep-timeout-seconds", type=int, default=600)
    overnight_loop.add_argument("--fast-max-tokens", type=int, default=256)
    overnight_loop.add_argument("--deep-max-tokens", type=int, default=512)
    overnight_loop.add_argument("--max-iterations", type=int, default=8)
    overnight_loop.add_argument("--max-runtime-minutes", type=float, default=480)
    overnight_loop.add_argument("--sleep-seconds", type=float, default=30)
    overnight_loop.add_argument("--skip-start", action="store_true")
    overnight_loop.add_argument("--keep-running", action="store_true")
    overnight_loop.add_argument("--continue-on-failure", action="store_true")
    overnight_loop.add_argument(
        "--nvidia-driver-branch",
        choices=["game_ready", "studio", "oem", "unknown"],
        default="unknown",
        help="NVIDIA driver branch reported by the NVIDIA App; version is collected with nvidia-smi.",
    )

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

    model_fit_scorecard = subparsers.add_parser("model-fit-scorecard")
    model_fit_scorecard.add_argument("--machine-profile", type=Path, required=True)
    model_fit_scorecard.add_argument("--machine-id")
    model_fit_scorecard.add_argument("--model-candidates", type=Path)
    model_fit_scorecard.add_argument("--benchmark-record", type=Path, action="append", default=[])
    model_fit_scorecard.add_argument("--experiment-summary", type=Path, action="append", default=[])
    model_fit_scorecard.add_argument("--capability-record", type=Path, action="append", default=[])
    model_fit_scorecard.add_argument("--context-envelope", type=Path, action="append", default=[])
    model_fit_scorecard.add_argument("--workload-profile", type=Path)
    model_fit_scorecard.add_argument("--scorecard-id")
    model_fit_scorecard.add_argument("--output", type=Path, required=True)

    deployment_plan = subparsers.add_parser("deployment-plan")
    deployment_plan.add_argument("--scorecard", type=Path, required=True)
    deployment_plan.add_argument("--plan-id")
    deployment_plan.add_argument("--workload-name", default="Local LLM outfitting")
    deployment_plan.add_argument(
        "--workload-intent",
        choices=["inference", "benchmark", "dataset_prep", "tiny_training_smoke", "training", "agent_runtime"],
        default="agent_runtime",
    )
    deployment_plan.add_argument("--output", type=Path, required=True)

    model_fit_html = subparsers.add_parser("model-fit-html")
    model_fit_html.add_argument("--scorecard", type=Path, required=True)
    model_fit_html.add_argument("--experiment-comparison", type=Path, action="append", default=[])
    model_fit_html.add_argument("--title")
    model_fit_html.add_argument("--output", type=Path, required=True)

    return parser


if __name__ == "__main__":
    raise SystemExit(main())
