# Capture Guide

Use this guide to turn each available machine into a clean AppLens-LLM training and eval source.

## Priority Order

1. ASUS ProArt PX13: raw AppLens/AppLens-Tune reports captured. Next step is AppLens-Bench.
2. HP ZBook: AppLens report captured, AppLens-Tune still needs a run log or fixed runner retry.
3. Dell XPS 13 9350: raw Linux AppLens/AppLens-Tune reports captured. Exact SKU and benchmark still pending.
4. HP EliteBook 845 G8: raw Linux AppLens/AppLens-Tune reports captured. Exact SKU and benchmark still pending.
5. HP laptop: consumer laptop baseline.
6. Lenovo laptop: vendor diversity.
7. Old MacBook: macOS baseline, best effort.
8. Gaming PC: raw AppLens/AppLens-Tune reports captured over SSH using `/dev/shm` because disk space is full. Sanitize before promoting into training/eval data.

## Per-Machine Checklist

For every machine, collect:

- sanitized AppLens report
- sanitized AppLens-Tune report
- local AI readiness profile
- vendor, model, and exact SKU/product number
- GPU/VRAM/RAM/CPU/storage summary
- hardware topology with separate accelerator rows for dGPU, iGPU, NPU, and CPU fallback when useful
- memory evidence per accelerator: physical dedicated VRAM, VGM or reserved memory, shared graphics memory, reported total graphics memory, estimated usable inference memory, confidence, and source
- one small local inference benchmark when practical
- benchmark proof fields: backend, devices used, mixed-device offload, tokens/sec, OOM/fallback/crash, CPU spill, and thermal notes
- local model candidate inventory: sanitized model label, family, parameter size, quantization, file size, local/candidate/missing status, and intended roles
- telemetry source notes, especially AMD Software: Adrenalin Edition logs for Radeon/VGM benchmark runs

Do not collect browser history, tokens, SSH keys, user files, raw application data, client documents, or private paths unless they are manually redacted.

Do not promote serial numbers, UUIDs, raw adapter device IDs, or user-specific local paths into committed examples. Use stable sanitized identifiers such as `nvidia-dgpu-0`, `amd-igpu-0`, and `cpu-0`.

## AppLens-LLM Bench Targets

For GPU-capable Windows/Linux machines:

- Jan endpoint on `http://127.0.0.1:1337/v1`
- llama.cpp endpoint on localhost if already installed
- small model smoke test first
- context sweep only after the smoke test passes
- run separate backend checks when hardware is hybrid, such as CUDA for NVIDIA, Vulkan or DirectML for Windows iGPU paths, and CPU fallback

For CPU-only or weak machines:

- record CPU-only baseline
- prefer tiny inference smoke tests
- mark local training unsupported
- include remote/cloud recommendation examples

## Promotion Rules

Raw local captures stay out of git.

Create an ignored review manifest after captures are dropped into `../AppLens/raw`:

```powershell
uv run applens-llm ingest-captures --source ../AppLens/raw --output data/raw/capture-records.jsonl
```

Promote only:

- sanitized `machine-profile` rows
- sanitized model candidate lists for scorecards
- sanitized `training-example` rows
- benchmark records with no private paths
- brief notes that explain failures or policy gates

For advertised or reserved graphics memory claims, promote both the claim and the verification status. A claim like "RTX VRAM plus AMD VGM creates a 22 GB pool" stays `user_claimed` and unproven or partially verified until a benchmark proves the runtime actually used the relevant devices together.

When AMD Software logs are used, promote only summarized metrics or sanitized `telemetry_sources` references. Raw CSV logs remain local-only unless manually scrubbed.

The eval set should prefer real machines. Synthetic examples should fill missing edge cases after real coverage exists.

## Capture Backlog

- AppLens capture reports are standardized as `.md` files. Diagnostic logs stay `.txt`; Markdown is the curated AppLens-LLM training intake contract because section headings, tables, and bullets are easier to review and parse.
- Capture-folder ingestion now produces ignored `capture-record` manifests for review before promotion.
- Add JSON export after Markdown. JSON should become the best machine-readable format for schema-backed training and eval generation.
- First target: promote 5 real AppLens reports into sanitized machine profiles and training/eval candidates.

## SKU Capture

Capture SKU separately from model. `model` can remain a broad family like `XPS` or `EliteBook`; `sku` should identify the specific verified hardware variant.

Windows examples:

```powershell
Get-CimInstance Win32_ComputerSystem | Select-Object Manufacturer,Model,SystemSKUNumber
Get-CimInstance Win32_ComputerSystemProduct | Select-Object Vendor,Name,Version
```

Do not promote `IdentifyingNumber`, UUID, serial number, PNP device ID, or adapter LUID fields.

Linux examples:

```bash
sudo dmidecode -s system-sku-number
sudo dmidecode -s system-product-name
```

macOS examples:

```bash
system_profiler SPHardwareDataType
```

Use the Mac model identifier as the SKU equivalent when Apple does not expose a normal OEM SKU.
