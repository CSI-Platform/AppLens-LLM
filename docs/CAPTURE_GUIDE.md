# Capture Guide

Use this guide to turn each available machine into a clean AppLens-LLM training and eval source.

## Priority Order

1. ASUS laptop: likely best training candidate if it is the RTX 4050 machine.
2. HP ZBook: workstation-class sample and likely strong local AI candidate.
3. Dell XPS: thin laptop and CPU/iGPU baseline.
4. HP EliteBook: business laptop and conservative policy baseline.
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
- one small local inference benchmark when practical
- failure notes for OOM, missing CUDA, thermal throttling, or unsupported runtime

Do not collect browser history, tokens, SSH keys, user files, raw application data, client documents, or private paths unless they are manually redacted.

## AppLens-LLM Bench Targets

For GPU-capable Windows/Linux machines:

- Jan endpoint on `http://127.0.0.1:1337/v1`
- llama.cpp endpoint on localhost if already installed
- small model smoke test first
- context sweep only after the smoke test passes

For CPU-only or weak machines:

- record CPU-only baseline
- prefer tiny inference smoke tests
- mark local training unsupported
- include remote/cloud recommendation examples

## Promotion Rules

Raw local captures stay out of git.

Promote only:

- sanitized `machine-profile` rows
- sanitized `training-example` rows
- benchmark records with no private paths
- brief notes that explain failures or policy gates

The eval set should prefer real machines. Synthetic examples should fill missing edge cases after real coverage exists.

## Capture Backlog

- Add Markdown report export to the AppLens script bundle. Keep `.txt` as the raw fallback, but prefer `.md` for curated AppLens-LLM training intake because section headings, tables, and bullets are easier to review and parse.
- Add JSON export after Markdown. JSON should become the best machine-readable format for schema-backed training and eval generation.
- First target: promote 5 real AppLens reports into sanitized machine profiles and training/eval candidates.

## SKU Capture

Capture SKU separately from model. `model` can remain a broad family like `XPS` or `EliteBook`; `sku` should identify the specific verified hardware variant.

Windows examples:

```powershell
Get-CimInstance Win32_ComputerSystem | Select-Object Manufacturer,Model,SystemSKUNumber
Get-CimInstance Win32_ComputerSystemProduct | Select-Object Vendor,Name,Version,IdentifyingNumber,UUID
```

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
