# Blackboard Handoff Contract

Date: 2026-05-10

## Decision

Fast-to-deep local model handoffs must carry an explicit blackboard contract.

The AppLens blackboard is an append-only JSONL evidence ledger written by `blackboard.py`. It records tasks, model responses, failures, handoffs, benchmark references, and verdicts. It is not CUDA/Vulkan/ROCm shared memory, not inter-GPU IPC, not pooled VRAM, and not a claim that NVIDIA and AMD memory are one device.

## Required Distinction

When a task asks whether lanes or models communicate through the blackboard, the answer is:

- Yes at the AppLens controller/file-ledger layer.
- No at the GPU memory/native API layer.

A plain "no" answer is wrong because it erases the AppLens handoff mechanism. A "shared GPU memory" answer is wrong because it turns a coordination ledger into a hardware memory claim.

The full contract should be injected into prompts only when the task or response concerns blackboards, handoffs, lane communication, pooled memory, shared memory, VRAM, or backend memory claims. Unrelated scorecard or tuning prompts should get a shorter guardrail so the models do not turn every review into blackboard semantics.

## Runtime Result

The May 10, 2026 bounded local smoke validated this contract on the two configured lanes:

- fast lane: NVIDIA/CUDA, Jan 4B Q4
- deep lane: AMD/VGM Vulkan, Qwen 27B IQ3
- result: 3 attempted, 3 completed, schema-valid blackboard JSONL

This proves the controller-level handoff loop works conceptually. It does not prove pooled VRAM, mixed-device offload, or zero-copy inter-GPU communication.
