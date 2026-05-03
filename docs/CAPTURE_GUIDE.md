# Capture Guide

Use this guide to turn each available machine into a clean AppLens-LLM training and eval source.

## Priority Order

1. Gaming PC: already benchmarked with llama.cpp; capture fresh AppLens/AppLens-Tune reports next.
2. ASUS laptop: likely best training candidate if it is the RTX 4050 machine.
3. HP ZBook: workstation-class sample and likely strong local AI candidate.
4. Dell XPS: thin laptop and CPU/iGPU baseline.
5. HP EliteBook: business laptop and conservative policy baseline.
6. HP laptop: consumer laptop baseline.
7. Lenovo laptop: vendor diversity.
8. Old MacBook: macOS baseline, best effort.

## Per-Machine Checklist

For every machine, collect:

- sanitized AppLens report
- sanitized AppLens-Tune report
- local AI readiness profile
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
