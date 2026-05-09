# VGM Test Guide

Use this guide to test whether AMD Variable Graphics Memory is active before testing any larger inference claims.

Raw outputs belong under ignored `out/vgm/`. They may include local model paths and should not be committed.

## Before Changing VGM

Capture the current state:

```powershell
$env:APPLENS_LOCAL_AI_ROOT = "C:\path\to\local-ai"

uv run applens-llm vgm-snapshot `
  --label before-vgm `
  --output out/vgm/before-vgm.json `
  --model-root "$env:APPLENS_LOCAL_AI_ROOT" `
  --llama-root "$env:APPLENS_LOCAL_AI_ROOT"
```

Expected current ASUS PX13 baseline:

- Radeon 890M dedicated memory near `512 MB`
- RTX 4050 memory near `6141 MiB`
- Vulkan sees both Radeon 890M and RTX 4050
- CUDA llama.cpp may be present, but Vulkan llama.cpp is required for AMD iGPU benchmark proof

## Enable VGM

In AMD Software: Adrenalin Edition, set Variable Graphics Memory to the desired 16 GB option, then restart the PC.

## After Restart

Capture the post-restart state:

```powershell
$env:APPLENS_LOCAL_AI_ROOT = "C:\path\to\local-ai"

uv run applens-llm vgm-snapshot `
  --label after-vgm `
  --output out/vgm/after-vgm.json `
  --model-root "$env:APPLENS_LOCAL_AI_ROOT" `
  --llama-root "$env:APPLENS_LOCAL_AI_ROOT"
```

Compare before and after:

```powershell
uv run applens-llm vgm-compare `
  --before out/vgm/before-vgm.json `
  --after out/vgm/after-vgm.json `
  --output out/vgm/vgm-comparison.json
```

## Result Interpretation

- `vgm_activated=true`: Radeon dedicated memory increased to roughly 16 GB or higher.
- `next_step=install_or_build_vulkan_llamacpp`: VGM is active, but the local llama.cpp build cannot benchmark AMD/Vulkan yet.
- `next_step=run_vulkan_benchmark`: VGM is active and a Vulkan-capable llama.cpp build was found.
- `next_step=fix_vgm_setting`: Radeon dedicated memory did not increase enough; revisit AMD settings before benchmarking.

The first pass only proves VGM activation. It does not prove an RTX plus Radeon pooled inference device. That requires a separate runtime benchmark that records backend, devices used, memory use, CPU spill, fallback, OOM/crash status, tokens/sec, and thermal notes.

## llama.cpp Vulkan Test

Use one llama.cpp Vulkan build for the equal-backend comparison. The b8892 Windows Vulkan release sees this ASUS PX13 as:

```text
Vulkan0: AMD Radeon(TM) 890M Graphics (24380 MiB, 23161 MiB free)
Vulkan1: NVIDIA GeForce RTX 4050 Laptop GPU (5920 MiB, 5152 MiB free)
```

On this machine, AMD Vulkan model loading required disabling cooperative matrix use:

```powershell
$env:GGML_VK_DISABLE_COOPMAT = "1"
```

The AppLens-LLM harness command:

```powershell
$env:APPLENS_LOCAL_AI_ROOT = "C:\path\to\local-ai"

uv run applens-llm llamacpp-devices `
  --binary "$env:APPLENS_LOCAL_AI_ROOT\llamacpp\backends\b8892\win-vulkan-x64\llama-bench.exe" `
  --output out/vgm/llamacpp-vulkan-devices.json
```

Small equal-backend comparison:

```powershell
uv run applens-llm llamacpp-bench `
  --binary "$env:APPLENS_LOCAL_AI_ROOT\llamacpp\backends\b8892\win-vulkan-x64\llama-bench.exe" `
  --model "$env:APPLENS_LOCAL_AI_ROOT\llamacpp\models\Jan-v3.5-4B-Q4_K_XL\model.gguf" `
  --device Vulkan0 `
  --label amd-890m-vgm-vulkan-small `
  --output out/vgm/bench-amd-890m-vulkan-small.json `
  --prompt-tokens 512 `
  --generation-tokens 128 `
  --repetitions 3 `
  --disable-vulkan-coopmat

uv run applens-llm llamacpp-bench `
  --binary "$env:APPLENS_LOCAL_AI_ROOT\llamacpp\backends\b8892\win-vulkan-x64\llama-bench.exe" `
  --model "$env:APPLENS_LOCAL_AI_ROOT\llamacpp\models\Jan-v3.5-4B-Q4_K_XL\model.gguf" `
  --device Vulkan1 `
  --label nvidia-rtx4050-vulkan-small `
  --output out/vgm/bench-nvidia-rtx4050-vulkan-small.json `
  --prompt-tokens 512 `
  --generation-tokens 128 `
  --repetitions 3 `
  --disable-vulkan-coopmat
```

First pass results:

| Device | Model | Prompt tok/s | Generation tok/s | Outcome |
| --- | --- | ---: | ---: | --- |
| AMD Radeon 890M VGM, Vulkan0 | qwen3 4B Q4_K_M | 426.09 | 26.66 | pass |
| NVIDIA RTX 4050, Vulkan1 | qwen3 4B Q4_K_M | 1916.37 | 53.00 | pass |
| AMD Radeon 890M VGM, Vulkan0 | qwen35 27B IQ3_S, 11.7 GB file | 54.18 | 5.71 | pass |
| NVIDIA RTX 4050, Vulkan1 | qwen35 27B IQ3_S, 11.7 GB file | 0 | 0 | failed: `ErrorOutOfDeviceMemory` |

Interpretation: the RTX 4050 is faster for models that fit in 6 GB. The Radeon 890M with 16 GB VGM is a usable capacity path for larger local models that the RTX 4050 cannot load alone.

## Two-Lane Orchestration

After separate llama.cpp servers are running, AppLens-LLM can record a blackboard experiment through runtime lanes instead of treating this ASUS-specific test as a special case. Start from the sanitized lane example and keep the machine-specific copy under ignored `out/`:

```powershell
uv run applens-llm lanes-check --config examples/runtime-lanes.example.json
uv run applens-llm lane-start --config out/runtime/local-lanes.json --lane fast-nvidia
uv run applens-llm lane-start --config out/runtime/local-lanes.json --lane deep-amd-vgm
uv run applens-llm blackboard-init --experiment-id exp-vgm-two-lane --title "VGM two-lane smoke" --output out/blackboard/exp-vgm-two-lane.jsonl
uv run applens-llm orchestrate-once --config out/runtime/local-lanes.json --lane fast-nvidia --experiment-id exp-vgm-two-lane --task-id task-1 --prompt "Summarize the local runtime state." --blackboard out/blackboard/exp-vgm-two-lane.jsonl --timeout-seconds 120
uv run applens-llm orchestrate-once --config out/runtime/local-lanes.json --lane deep-amd-vgm --experiment-id exp-vgm-two-lane --task-id task-1 --prompt "Review the fast lane answer and add capacity-focused concerns." --blackboard out/blackboard/exp-vgm-two-lane.jsonl --timeout-seconds 600 --max-tokens 320
uv run applens-llm lane-stop --lane fast-nvidia
uv run applens-llm lane-stop --lane deep-amd-vgm
```

This proves routed model behavior, not pooled VRAM. Each lane still records its backend, device selector, accelerator IDs, latency, response, and failure mode independently.

The equivalent one-command experiment is:

```powershell
uv run applens-llm experiment-run --config out/runtime/local-lanes.json --fast-lane fast-nvidia --deep-lane deep-amd-vgm --experiment-id exp-vgm-two-lane --prompt "Explain why AppLens-LLM treats the RTX 4050 and Radeon 890M VGM as separate runtime lanes instead of one proven pooled VRAM device." --blackboard out/blackboard/exp-vgm-two-lane.jsonl --summary out/blackboard/exp-vgm-two-lane-summary.json --deep-timeout-seconds 600 --deep-max-tokens 320 --nvidia-driver-branch game_ready
```

When comparing NVIDIA Game Ready and Studio drivers, run this once on the current branch, switch drivers manually in NVIDIA App, restart if prompted, then rerun the same command with `--nvidia-driver-branch studio`. AppLens records this as driver evidence in the ignored summary JSON and treats driver changes as benchmark invalidators.

Then compare the summaries:

```powershell
uv run applens-llm experiment-compare `
  --baseline out/blackboard/exp-game-ready-summary.json `
  --candidate out/blackboard/exp-studio-summary.json `
  --output out/blackboard/driver-comparison.json
```

The first Game Ready versus Studio run on this PX13 kept NVIDIA driver version `596.36` in both cases. The NVIDIA/CUDA fast lane differed by only `36 ms`, so driver branch did not appear to be a major local-LLM factor in that single run. The AMD/VGM lane varied more, but output token counts also differed, so repeat runs are required before treating that as a driver effect.

The first Studio repeatability pass with the same prompt and token caps produced:

| Lane | Average latency | Min | Max | Spread | Average tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| NVIDIA/CUDA fast lane | 2173 ms | 2037 ms | 2284 ms | 247 ms | 133 |
| AMD/VGM Vulkan deep lane | 20966.75 ms | 18999 ms | 22377 ms | 3378 ms | 269.75 |

Interpretation: the observed Game Ready versus Studio difference is smaller than normal run-to-run variation for this workflow. AppLens should record driver branch/version, but it should prioritize backend, device, model size, quantization, and repeat benchmark evidence.

## AMD Software Telemetry

For AMD/VGM benchmarks, use AMD Software: Adrenalin Edition as the preferred AMD-side telemetry source. AppLens-LLM should ask for this log instead of trying to reproduce every metric through Windows counters.

Recommended logging setup:

- Source: AMD Software > Performance > Metrics > Tracking.
- Default folder observed on this ASUS PX13: `%LOCALAPPDATA%\AMD\CN`.
- Sampling interval: `2 seconds` for long runs, `1 second` for short smoke tests.
- Enable: GPU utilization, GPU clock speed, GPU power consumption, GPU temperature, GPU memory utilization, GPU memory clock speed, CPU utilization, and system memory utilization.
- Optional: GPU junction temperature and fan speed if shown on this laptop.
- Skip for local LLM testing: frame rate, frame time, frame generation lag, and FPS-oriented latency unless the workload is a graphical application.

Benchmark records can reference this evidence through `telemetry_sources` with `source=amd_adrenalin`. Keep raw CSV logs under ignored `out/vgm/` or another local-only folder.

AMD documents that Adrenalin can log performance stats to a file, defaults logs under `%LOCALAPPDATA%\AMD\CN`, supports 0.25 to 5 second sampling intervals, and includes GPU memory utilization plus CPU and system memory metrics.

Summarize a hardware CSV:

```powershell
uv run applens-llm adrenalin-summary `
  --input "$env:LOCALAPPDATA\AMD\CN\Hardware.20260508-172749.CSV" `
  --output out/vgm/adrenalin-hardware-20260508-172749-summary.json
```

The first captured 11.7 GB AMD/VGM capacity run produced 67 samples from `17:25:34` to `17:27:48` and showed:

| Metric | Peak |
| --- | ---: |
| GPU memory utilization | 13281 MB |
| GPU utilization | 99% |
| GPU clock | 2896 MHz |
| GPU power | 54 W |
| GPU temperature | 78 C |
| CPU utilization | 43.78% |
| System memory utilization | 15.12 GB |

This is useful evidence because the GPU memory peak aligns with the large model loading on the AMD/VGM path.
