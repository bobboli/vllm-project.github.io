---
layout: post
title: "Production Serving MiniMax H3 with vLLM-Omni"
author: "vLLM-Omni Team"
summary: "How vLLM-Omni combines dense lossless kernels, four-step adapters, quantization, distributed offload, and encoder disaggregation to serve MiniMax H3 in production."
description: "An evidence-driven guide to the architecture and optimization stack for production MiniMax H3 serving with vLLM-Omni."
image: /assets/logos/vllm-logo-text-light.png
tags:
  - performance
  - large-scale-serving
  - multimodal
  - vllm-omni
published: false
---

> [!NOTE]
> This is an unpublished draft. The architecture and feature descriptions are
> linked to their implementation sources, while the B300 benchmark and quality
> fields intentionally remain `TBD` until collaborators provide final,
> reproducible validation data.

MiniMax H3 is a joint video-and-audio diffusion model: one request can combine
text, images, videos, and audio references, then return an MP4 containing H.264
video and synchronized stereo audio. That capability also makes production
serving unusually demanding. A deployment must coordinate a large Qwen3-VL
encoder, a long-sequence audio-video DiT, separate video and audio VAEs, and a
CPU media path without letting any one stage dominate latency or memory.

<p align="center">
  <img src="/assets/figures/2026-08-29-minimax-h3-production-serving/h3-model-pipeline.svg" alt="MiniMax H3 model pipeline from multimodal inputs through shared encoders, a packed sequence, task-specific joint audio-video diffusion, separate VAE decoders, and MP4 muxing" width="100%">
</p>

*Figure 1: Text is encoded by the H3/Qwen3-VL encoder; visual conditions use
both that encoder and the Visual VAE; audio references use the Audio VAE. Their
representations join noisy target video/audio latents in one packed sequence.
The selected FL2VA or Ref2VA DiT jointly predicts both output latents before
separate VAE decode and CPU mux. Sources: the official
[MiniMax H3 model card](https://huggingface.co/MiniMaxAI/MiniMax-H3), the
[vLLM-Omni recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3.md),
and the [Diffusers pipeline description](https://huggingface.co/docs/diffusers/main/en/api/pipelines/minimax_h3).*

This post explains how [vLLM-Omni](https://github.com/vllm-project/vllm-omni)
turns that pipeline into a production serving system. We separate the stack
into three decision lanes: dense reference-quality runtime optimization,
quality- or precision-changing acceleration, and production deployment
features. To keep the evidence compact and comparable, the article benchmarks
one eight-GPU NVIDIA B300 node: the 10-second request remains the canonical A/B,
and one selected four-step profile adds a 5/10/15-second generation-speed
reference. H200, RTX PRO 5000, consumer GPUs, ROCm, and NPU deployments remain
covered by maintained recipes rather than additional result matrices.

## TL;DR

- **One serving contract, three H3 tasks.** vLLM-Omni serves text-to-video-and-audio
  (T2VA), first/last-frame-to-video-and-audio (FL2VA), and mixed-reference
  video-and-audio generation (Ref2VA) through `/v1/videos`.
- **Optimize the lossless lane first.** Dense attention, packed-sequence and
  Ulysses boundaries, fused DiT operators, VAE parallelism/kernels, GPU output
  packing/transport, and CPU MP4 construction are compared end to end against
  Diffusers on B300.
- **Treat acceleration knobs as separate quality decisions.** Turbo and
  FastH3 reduce the denoiser to four forwards; online FP8, SVDQuant, SAGE,
  Skip-Softmax, Sol-Attn, and Cache-DiT change precision, coverage, weights, or
  executed work and therefore require their own quality evidence.
- **System architecture determines the production frontier.** Distributed
  layerwise offload changes the latency/throughput/memory trade-off;
  disaggregated encoding makes the Qwen3-VL stage independently schedulable
  and cacheable. Step execution implements admission and abort boundaries, but
  current H3 measurements show no latency or throughput benefit; its useful
  production case remains to be demonstrated.
- **One canonical benchmark keeps the comparison tractable.** Every feature A/B
  uses the official 10-second, 1344×768 T2VA case. The selected four-step B300
  profile also runs at 5 and 15 seconds to report complete-MP4 real-time factor;
  FL2VA and Ref2VA remain capability and recipe coverage.

## 1. Commercial workloads and serving goals

MiniMax H3 has two checkpoint partitions. `FL2VA` serves both text-only and
first/last-frame-conditioned generation; `Ref2VA` handles mixed image, video,
and audio references. Shared components include the tokenizer and processor,
the Qwen3-VL encoder, and the video and audio VAEs. The full task and input
contract is documented in the
[MiniMax H3 recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3.md).

| Task | Commercial examples | Primary serving pressure |
|---|---|---|
| T2VA | Creative generation, advertising drafts, synthetic media pipelines | DiT latency and output throughput |
| FL2VA | Branded content, controlled transitions, image animation | Image encoding plus denoising latency |
| Ref2VA | Character consistency, video editing, audio-conditioned generation | Long multimodal encoder and packed-attention sequences |

This post uses **T2VA on 8× B300 as its only canonical benchmark**. It covers
the base, Turbo, FastH3, quantization, and kernel paths without introducing
reference-media preprocessing or another hardware matrix. FL2VA, Ref2VA, and
other accelerators remain important capabilities documented by their recipes
and implementation sources.

For production users, the relevant question is not simply whether one request
completes. A useful deployment has to balance five objectives:

1. client-visible end-to-end latency;
2. sustained node throughput and tail latency under an explicit arrival model;
3. device HBM, host RAM, and checkpoint-storage requirements;
4. video, audio, and reference-conditioning quality; and
5. operational behavior, including startup, warmup, failure recovery, and
   output transport.

## 2. Benchmark contract and evidence rules

### 2.1 Frozen canonical workload

Every comparable row uses one official T2VA workload before platform-specific
tuning:

| Control | Canonical value |
|---|---|
| Task | T2VA through the FL2VA partition; no reference media |
| Output | 10.0 seconds requested; 1344×768, 243 aligned frames at 24 FPS, 10.125 seconds encoded |
| Base schedule | 50 requested sigma points and 49 expected DiT forwards; record both |
| Prompt | The official MiniMax H3 model-card `case-T2VA` H3-Context-IR output, frozen at model revision `42ed227e`; SHA-256 `98f36b879692095e099ae824c18d9e93e7006a490e082fd474a5f531769dcf06` |
| Seed | `0`, matching the official H3-Base script |
| vLLM-Omni | vLLM-Omni with vLLM `v0.28.0`; dense BF16 and Fast Ulysses |
| Diffusers lane | Diffusers `v0.40.0`, PyTorch `2.13.0+cu130`, and Transformers `5.14.1`; eight replicated-weight ranks with native context parallelism, Ulysses8, and Ring1 |
| Model | [MiniMax H3 `42ed227e`](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/42ed227ee7df40d41602854ae760620d6eb651fe) |
| Repetitions | One full-shape feasibility request, also recorded as the excluded compile/kernel warmup, then at least two measured repetitions per claimed A/B; the B300 Diffusers run uses five |
| Output checks | HTTP/process success; full H.264/AAC decode; 1344×768, exactly 243 frames at 24 FPS; 32 kHz stereo audio; nonzero frame variance and audio RMS; prompt-adherence review |
| Raw evidence | Retained by each benchmark contributor outside the blog repository |

The canonical prompt is the 380-word structured output shown in the official
[MiniMax H3 `case-T2VA`](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/42ed227ee7df40d41602854ae760620d6eb651fe/README.md#case-t2va):
a two-shot space-opera sequence with a fleet jump, a captain's physical
reaction, synchronized bridge impacts and room tone, and an orchestral
rise-and-cut. Contributors retrieve that expanded prompt once, verify its
SHA-256, and pass it directly to H3-Base. The hosted H3-Context-IR call is not
part of readiness or request latency and must not be rerun independently per
platform.

The article keeps three comparison lanes separate:

1. **Lossless runtime A/B:** 50 sigma points, BF16 checkpoint weights, dense
   full attention, and no adapter, cache, sparsity, or quantization. Declare
   any capacity-required component staging; keep vLLM-Omni DLO in Section 5.
2. **Accelerated-path A/B:** change one adapter, numerical format, cache, or
   sparse-attention policy and attach same-seed quality evidence.
3. **Production-topology study:** change placement or parallelism to measure a
   latency/throughput/memory frontier rather than a kernel speedup.

The first feasibility request is the stop condition. OOM, accelerator error,
invalid media, missing audio, failed quality gates, or an unexpected backend
fallback stops the profile before repeated measurement. Tail latency requires
a separately declared arrival process and enough samples; it is never inferred
from the two single-request repetitions.

### 2.2 Measurement methodology

The cross-runtime headline is client-visible latency: synchronous request
submission through receipt of the complete MP4. Model download, checkpoint
conversion, compilation, startup, and warmup remain outside that interval.
Diffusers reports this E2E value only.

For diagnosis, vLLM-Omni exposes a hierarchy of native timing boundaries:

| Source label used below | Corresponding object | Scope |
|---|---|---|
| `client` | `curl` | The external HTTP request, through receipt of the complete MP4 |
| `request` | `RequestE2EStats` | One generation handled by the orchestrator; it may traverse one or more stages |
| `stage` | `StageRequestStats` | One independently scheduled execution unit with its own engine, workers, and device group |
| `engine` | `StageRequestStats.diffusion_metrics` | Request admission, execution, asynchronous output readiness, and output formatting inside a diffusion stage |
| `profiler` | `DiffusionPipelineProfiler` | Selected `MiniMaxH3Pipeline` method boundaries: prompt encode, denoise, and VAE decode |
| `server` | HTTP server timer | MP4 encode and mux after the final stage |

The standard MiniMax H3 profile has one diffusion stage. Prompt encoding, DiT
denoising, and both VAE decoders are model operations inside that stage.

```text
Timing boundary                                      Measurement
Client request: complete-response E2E                client: time_total
├─ Orchestrator request                              request: e2e_total_ms
│  ├─ Queue before stage dispatch                    stage: pipeline_timings.queue_wait_ms
│  └─ Diffusion stage: MiniMax H3 generation         stage: stage_gen_time_ms
│     ├─ Diffusion engine: execution                 engine: diffusion_engine_exec_time_s
│     │  ├─ Scheduler queue                          engine: scheduler_queue_wait_s
│     │  ├─ Model operation: prompt encode           profiler: encode_prompt
│     │  ├─ Model operation: DiT denoise loop        profiler: diffuse
│     │  ├─ Model operation: Video VAE decode        profiler: video_vae.decode_latent
│     │  └─ Model operation: Audio VAE decode        profiler: audio_vae.decode_latent
│     ├─ Diffusion engine: output-ready wait         engine: output_ready_wait_time_s
│     └─ Diffusion engine: output formatting         engine: postprocess_time_s
└─ HTTP response
   ├─ MP4 encode and mux                             server: Video response encoding (MP4 bytes)
   └─ Response delivery                              client: included only in time_total
```

These measurements are nested. `stage_gen_time_ms` contains the diffusion
engine work, and `diffusion_engine_exec_time_s` contains the model operations,
so parent and child values are never added together. Divide the `diffuse` time
by the observed DiT-forward count for the per-forward value.

Each result also carries this compact placement manifest:

| Profile | Encoder | DiT | Video/audio VAE | Output |
|---|---|---|---|---|
| TBD | Devices + TP/replicas/cache | Devices + TP/USP/Ring/DP/CFG/PP + offload/backend | Devices + VAE PP/mode/tiling + audio placement | Transport + CPU affinity/threads + mux path |

#### Reproducing the vLLM-Omni measurements

The server remains resident for one excluded full-shape warmup and the measured
requests. The external client boundary is:

```bash
curl -sS -o output.mp4 \
  -w 'client_e2e_s=%{time_total}\n' \
  -X POST "http://127.0.0.1:${PORT}/v1/videos/sync" \
  -F "prompt=<${PROMPT_FILE}" \
  -F 'width=1344' -F 'height=768' -F 'fps=24' \
  -F 'num_inference_steps=50' -F 'seed=0' \
  -F 'extra_params={"task":"t2va","duration":10.0,"aspect_ratio":"16:9","flow_shift":12.0,"audio_flow_shift":3.0}'
```

Enable the native pipeline profiler on the same eight-GPU configuration for
the diagnostic run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
vllm serve "${MODEL}" \
  --omni --host 0.0.0.0 --port "${PORT}" --trust-remote-code \
  --task-type fl2va \
  --num-gpus 8 --usp 8 --ring 1 --ulysses-a2a-permute \
  --vae-patch-parallel-size 8 --vae-parallel-mode tile --vae-use-tiling \
  --diffusion-attention-backend TRTLLM_ATTN \
  --enable-diffusion-pipeline-profiler --log-stats \
  2>&1 | tee vllm-omni-breakdown.log
```

The log contains the synchronized `DiffusionPipelineProfiler` timers,
`RequestE2EStats`, `StageRequestStats`, and the MP4 encode-and-mux timer. The
profiled request is diagnostic; the unprofiled client requests provide the
headline latency. Nsight Systems verifies GPU work and transfers but does not
replace these request-level timers.

### 2.3 Benchmark scope and deployment recipes

The benchmark covers **8× B300 only**: one lossless Diffusers/vLLM-Omni A/B,
isolated lossless kernel/transport checks, and selected four-step or
precision-changing profiles. Only the selected four-step profile receives the
5/10/15-second duration sweep. Other platforms are deployment guidance, not
cross-hardware comparisons:

| Deployment | Maintained guidance |
|---|---|
| H200 and other high-memory CUDA systems | [Full MiniMax H3 recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3.md) |
| RTX PRO 5000 Blackwell | [Dedicated RTX PRO 5000 recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3-RTX-PRO-5000.md) |
| RTX 4090 / 5090 | [RTX 4090](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3-4090.md) / [RTX 5090](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3-5090.md) |
| DGX Spark GB10 | [GB10 recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3-Spark-GB10.md) |
| AMD ROCm | [ROCm section of the full recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3.md#amd-rocm-gfx942--gfx950) |
| Ascend NPU | Vendor-specific recipes and validation; no benchmark claim in this post |

## 3. Lossless runtime optimization

Here, **lossless** means that the comparison keeps the released BF16 model,
the 50-point schedule, and dense full-attention coverage. It does not promise
bitwise identity: changing reduction order, attention kernels, or FP32
accumulation can perturb a diffusion trajectory. Every row still needs the
same media and quality gates.

### 3.1 Dense attention, packed sequences, and Ulysses boundaries

H3 denoises one long packed text/audio/video sequence. vLLM-Omni preserves
document boundaries with variable-length attention metadata and can select a
platform-qualified dense backend without changing attention coverage. The
[TRTLLM refinement](https://github.com/vllm-project/vllm-omni/pull/5779)
trims structural padding before the kernel; the
[strict-Ulysses optimization](https://github.com/vllm-project/vllm-omni/pull/6173)
constructs rank-local input rows and gathers compact projected outputs instead
of full hidden states.

On B300, qualify dense `TRTLLM_ATTN`, cuDNN, and FA4 under one frozen topology
and record the resolved kernel. Keep that backend choice explicit in the
complete Diffusers/vLLM-Omni comparison so an engine result does not silently
mix in a kernel change.

[PR #6340](https://github.com/vllm-project/vllm-omni/pull/6340) adds an
orthogonal, opt-in Fast Ulysses transport through
`--ulysses-a2a-permute`. For the strict scatter-heads/gather-sequence layout,
it replaces regular Ulysses all-to-all plus explicit relayout with functional
CUDA custom ops backed by NCCL SymmetricMemory. The attention math, packed
documents, and selected dense backend remain unchanged.

The merged implementation JIT-builds the extension during worker/model
initialization, retains one grow-only workspace per device/process group,
requires that workspace to stay on one CUDA stream, and releases it before the
distributed environment is destroyed. A CUDA-graph deployment must warm its
maximum request shape before capture because the workspace cannot grow during
capture. Unsupported/non-strict layouts retain regular Ulysses.

The H3 gain is workload-sensitive: the contributor's long-video A/B improved
steady E2E by 1.36% with an 18.738-second additional warmup, while an
[independent four-step A/B](https://github.com/vllm-project/vllm-omni/pull/6340#issuecomment-5466589923)
reduced the diffusion stage by 8.8% with PSNR 35.3–39.4 dB and SSIM
0.977–0.988. Attribute only the isolated diffusion delta, keep JIT/readiness
separate, and do not add this gain to #6173 or attention-backend results from a
different base.

### 3.2 Fused DiT operators

The dense DiT path removes repeated launches, intermediate tensors, and layout
materialization without changing the model or step count:

- [Fused Q/K RMSNorm and RoPE](https://github.com/vllm-project/vllm-omni/pull/5990)
  fuses RMSNorm and RoPE into one kernel for Q and one for K.
- [FP32 fused modulation](https://github.com/vllm-project/vllm-omni/pull/6281)
  combines gather, modulation, normalization, and residual work while retaining
  FP32 accumulation.
- [Fused SwiGLU](https://github.com/vllm-project/vllm-omni/pull/6283) replaces
  separate SiLU and multiply launches.
- [Strict Ulysses boundaries](https://github.com/vllm-project/vllm-omni/pull/6173)
  keep the pre/post-transformer boundaries rank-local.
- [Fast Ulysses](https://github.com/vllm-project/vllm-omni/pull/6340)
  exchanges shards directly into the attention layout, avoiding separate
  permute and contiguous copies around the all-to-all.

These changes are not summed from their individual PR benchmarks. The article
measures the merged stack once, then uses isolated A/Bs only when the hardware,
topology, workload, and base revision are identical.

### 3.3 VAE parallelism and exact eager kernels

At short denoising schedules, decode becomes a much larger fraction of request
latency. H3 distributes native tiled video decode through VAE patch
parallelism. The merged [exact eager operator path](https://github.com/vllm-project/vllm-omni/pull/6607)
also accelerates Q/K normalization plus RoPE, SwiGLU, and scaled residuals on
SM90, SM100, and SM103 while retaining guarded fallbacks. Report the video VAE
critical path and audio VAE separately; an isolated decoder speedup is not an
equal-sized end-to-end claim.

### 3.4 Video output transport and CPU MP4 construction

The merged lossless output path now reduces data before it optimizes CPU muxing:

1. [PR #6824](https://github.com/vllm-project/vllm-omni/pull/6824)
   clamps, scales, and rounds decoded FP32 BCTHW frames on the GPU, combines
   dtype plus BCTHW→BTHWC conversion into one contiguous uint8 allocation, and
   avoids a redundant `torch.cat` for the common single-output case.
2. The existing pinned D2H and worker-to-engine path transports the four-times
   smaller uint8 payload. A subprocess boundary may still materialize a
   C-interleaved array whose individual RGB planes are strided.
3. [PR #6776](https://github.com/vllm-project/vllm-omni/pull/6776)
   lets the server-owned parallel converter accept those strided RGB planes,
   keeping transported output on the direct-planar route.
4. The [direct planar encoder](https://github.com/vllm-project/vllm-omni/pull/6288)
   plus [persistent eight-worker pool](https://github.com/vllm-project/vllm-omni/pull/6499)
   converts frames in order and writes H.264/AAC without a second full
   interleaved RGB buffer.

The resulting chain is:

`FP32 BCTHW on GPU → uint8 BTHWC → pinned D2H/IPC → parallel direct-planar frames → H.264/AAC MP4`

Source evidence isolates both gains:

- On 8× B300, [#6824](https://github.com/vllm-project/vllm-omni/pull/6824)
  reduced the worker payload by 75% and steady inference from 22.578 to
  21.683 seconds (−3.96%) with unchanged peak HBM and a byte-identical MP4.
- In a 243-frame paired CPU benchmark,
  [#6776](https://github.com/vllm-project/vllm-omni/pull/6776) reduced encoding
  wall time from 2.430 to 1.422 seconds (−40.94%) while process CPU increased
  9.02%; every output was byte-identical.

#6776 merged first and #6824 merged directly on top of it, so current main
contains both halves of the intended path. [PR #6764](https://github.com/vllm-project/vllm-omni/pull/6764)
closed unmerged; default single-stage serving remains subprocess-based and no
longer needs inline placement to reach direct-planar encoding. A fresh combined
A/B on the frozen ten-second workload is still required before the blog fills
its canonical vLLM-Omni result row.

The online H.264/AAC response remains byte-identical in the submitted tests.
The raw offline contract does change: callers that consume
`OmniRequestOutput.images[0]` directly now receive contiguous uint8 `[0,255]`
frames rather than float32 `[0,1]` frames and must branch on dtype.

For Diffusers, the equivalent end boundary includes its caller-side
`encode_video()`/mux step. For vLLM-Omni, it includes the complete non-streaming
HTTP response. Report output-preparation time, payload bytes, transport wall
time, MP4 wall/process CPU, peak RSS, chosen route, and CPU/NUMA placement
rather than attributing transport or CPU gains to the DiT.

### 3.5 Diffusers versus vLLM-Omni A/B

Both runtimes use eight B300 GPUs, the same prompt and seed, 50 sigma points,
and the same complete-MP4 timing boundary. Diffusers keeps one resident
`ModularPipeline`, replicates the weights across eight ranks, and uses native
context parallelism with Ulysses8, Ring1, and dense BF16 attention. vLLM-Omni
uses encoder TP1, DiT USP8 with Ring1 and Fast Ulysses, VAE PP8 tile decode, and
dense `TRTLLM_ATTN`.

vLLM-Omni completes the request in 58.371 seconds versus 82.239 seconds for
Diffusers: 29.0% lower latency, or a 1.409× complete-response speedup.

Before using pixelwise or waveform metrics across runtimes, verify that both
implementations consume the same generator state, draw order, latent shapes,
and scheduler grid. If that contract differs, report matched-prompt perceptual
and semantic quality instead of presenting SSIM/PSNR as numerical parity.

| B300 runtime | Prompt encode | DiT denoise | Video / audio VAE | MP4 encode + mux | Client E2E | Peak reserved HBM | Output |
|---|---:|---:|---:|---:|---:|---:|---|
| Diffusers | — | — | — | — | **82.239 s** | 151.699 GiB, peak rank | Coherent 243-frame H.264/AAC |
| vLLM-Omni | 0.057 s | 51.800 s total; 49 forwards; 1.057 s/forward | 0.952 s / 0.055 s | 1.528 s | **58.371 s** | 128.232 GiB, peak rank | Coherent 243-frame H.264/AAC |

Diffusers phase timings were not isolated. The vLLM-Omni phase values come from
a separate diagnostic request and are nested measurements rather than terms to
sum into E2E. Startup readiness is omitted because the two runs did not use a
comparable startup boundary. Generator draw-order parity is not established,
so the comparison makes no pixelwise claim. The raw commands, manifests,
samples, logs, and generated media are retained outside the blog repository.

## 4. Acceleration paths with explicit quality or precision trade-offs

### 4.1 Four-step adapters: Turbo and FastH3

The largest denoising win comes from reducing the number of expensive DiT
forwards. vLLM-Omni currently has two distinct integration models; they should
not be treated as interchangeable LoRAs.

| Path | Integration model | Current scope | Deployment status |
|---|---|---|---|
| [Turbo LoRA](https://github.com/vllm-project/vllm-omni/pull/6476) | Dynamically activated request adapter | FL2VA/T2VA, published four-forward schedule | Merged; [DLO support](https://github.com/vllm-project/vllm-omni/pull/6550) merged |
| [FastH3](https://github.com/vllm-project/vllm-omni/pull/6714) | Adapter fused into the checkpoint stream at load time | Dense/Data-Free T2VA | Merged; current integration rejects offload and VSA variants |

Turbo is the flexible serving option: a server can keep the base model and
activate the supported adapter per request. FastH3 is a specialized server
profile. Its artifact contains low-rank factors plus full-rank deltas that an
ordinary request-switchable LoRA layer cannot express, so the loader fuses it
before sharding. The sparse FastH3 VSA variants additionally depend on a
backend not yet implemented by this vLLM-Omni integration.

<p align="center">
  <img src="/assets/figures/2026-08-29-minimax-h3-production-serving/h3-few-step-adapters.svg" alt="Comparison of request-switchable Turbo LoRA and load-time-fused FastH3 weights" width="100%">
</p>

*Figure 2: Turbo keeps the base weights unchanged and adds request-selected A/B
sidecars, while FastH3 fuses low-rank and full-rank deltas into a
specialized student before sharding. Source contracts: vLLM-Omni
[#6476](https://github.com/vllm-project/vllm-omni/pull/6476),
[#6550](https://github.com/vllm-project/vllm-omni/pull/6550), and
[#6714](https://github.com/vllm-project/vllm-omni/pull/6714).*

The source PRs establish the mechanisms, but their workloads are not mixed
into the canonical comparison. For example, the merged FastH3 PR reports a
3.30× framework E2E and 8.37× diffusion-stage improvement on a different
B300 workload. Section 4.4 remains `TBD` until both paths are measured with the
frozen 10-second request and current revisions.

Any performance table must identify the adapter, number of requested sigma
points, actual DiT forward count, task, attention backend, and quality
comparison. A few-step result is not directly comparable to a 50-step baseline
unless those differences are explicit.

The canonical T2VA matrix compares base H3, Turbo, and FastH3 under one output
contract. Section 6 turns the merged Turbo path into concrete production
profiles; FastH3 remains a specialized, T2VA-only server profile whose
canonical quality and performance row is still pending.

### 4.2 Weight and activation quantization

- **Online FP8.** The merged
  [global FP8 path](https://github.com/vllm-project/vllm-omni/pull/5910)
  starts from the released BF16 checkpoint and quantizes eligible DiT and
  Qwen3-VL text-decoder linears at load time. Embeddings, norms, RoPE, the
  vision tower, both VAEs, and the model's FP32 projections retain their
  declared precision. Resident serving and supported DLO paths are distinct
  profiles and must be reported separately.
- **SVDQuant NVFP4 W4A4.** The merged
  [offline loader](https://github.com/vllm-project/vllm-omni/pull/6162)
  combines an NVFP4 W4A4 base GEMM with a BF16 low-rank correction. The current
  SM103 FL2VA evidence establishes checkpoint and correctness compatibility;
  native fused residual-GEMM performance remains follow-up work.

<p align="center">
  <img src="/assets/figures/2026-08-29-minimax-h3-production-serving/h3-quantization-paths.svg" alt="Comparison of online FP8 and offline SVDQuant W4A4 execution paths for MiniMax H3" width="100%">
</p>

*Figure 3: Online FP8 keeps runtime-created FP8 weights plus frozen weight
scales and dynamically quantizes activations. Offline SVDQuant adds a BF16
low-rank correction to an NVFP4 W4A4 base branch. Adapted from the vLLM-Omni
cookbook [online FP8](https://github.com/hsliuustc0106/vllm-omni-cookbook/blob/main/blog/_posts/2026-08-18-online-quantization-fp8.md)
and [SVDQuant](https://github.com/hsliuustc0106/vllm-omni-cookbook/blob/main/blog/_posts/2026-08-16-understanding-pr-6162-svdquant-w4a4-blackwell.md)
explainers.*

Quantization claims include peak HBM, host RAM during startup, checkpoint
storage, latency, and same-seed video/audio metrics. A capacity win is not
automatically a speed win, and a loader correctness result is not a fused
kernel result.

### 4.3 Sparse, skipped, and cached computation

These paths reduce or approximate attention/denoising work and therefore stay
outside Section 3's lossless comparison:

| Path | Current status | Boundary to report |
|---|---|---|
| [TRTLLM Skip-Softmax](https://github.com/vllm-project/vllm-omni/pull/5283) | Merged backend capability | Datacenter Blackwell only; threshold/sparsity, timestep gate, role overrides, dense baseline, and quality curve |
| [TRTLLM SAGE](https://github.com/vllm-project/vllm-omni/pull/5509) | Merged backend capability | Q/K type and block sizes, V precision, dense token-refiner override, hardware/kernel version, and audio/video quality |
| [Sol-Attn for H3](https://github.com/vllm-project/vllm-omni/pull/5851) | Preview | Report dense guards, `tau`, sink tokens, KV splits, and quality gates; use the hardware recipe for platform-specific setup |
| [Dynamic Cache-DiT quality](https://github.com/vllm-project/vllm-omni/pull/5853) | Merged | `quality=lossless` removes the cache policy; `quality=high` installs the H3 profile and needs deployment-specific hit-rate/quality evidence |
| FastH3 VSA variants | Rejected by the current FastH3 path | Sparse student artifacts require a VSA backend not implemented by that integration |

SAGE and Skip-Softmax may be composed because they alter different parts of the
same attention kernel, but their end-to-end gains must be measured together,
not multiplied. Sol-Attn remains a separately labeled preview. Cache-DiT is a
request policy, not a dense-kernel backend, and is incompatible with step
execution.

### 4.4 Acceleration A/B summary

Each row starts from the Section 3 lossless vLLM-Omni result on the same
platform and changes one declared acceleration policy:

| Platform / path | Weights / precision | Sigma points / actual forwards | Attention or cache policy | E2E / speedup | Peak HBM | Video/audio quality | Maturity / artifacts |
|---|---|---|---|---:|---:|---|---|
| B300 / Turbo | BF16 + dynamic LoRA | 5 / 4 | Dense TBD | TBD | TBD | TBD | Merged / TBD |
| B300 / FastH3 | Fused artifact | 5 / 4 | Dense only | TBD | TBD | TBD | Merged / TBD |
| B300 / online FP8 | Runtime FP8 | 50 / 49 | Dense TBD | TBD | TBD | TBD | Merged / TBD |
| B300 / SVDQuant | Offline W4A4 + BF16 correction | 50 / 49 | Dense TBD | TBD | TBD | TBD | Correctness baseline / TBD |
| B300 / SAGE or Skip-Softmax | BF16 weights | 50 / 49 | Exact policy TBD | TBD | TBD | TBD | Backend merged / TBD |

## 5. Production deployment features

### 5.1 Distributed Layerwise Offload

[Distributed Layerwise Offload (DLO)](https://vllm.ai/blog/2026-08-17-distributed-layerwise-offload)
keeps a bounded window of DiT layers in device memory and streams the remaining
weights from the host. It supports two materially different execution modes:

- **AllGather:** ranks retain weight shards and reconstruct each active layer
  collectively. This reduces aggregate host residency and can pair data
  parallel replicas with sequence parallelism.
- **Rank-local:** each rank streams the tensors produced by its ordinary model
  loader without reconstructing complete weights. It avoids AllGather but has
  a different host-memory and topology trade-off.

<p align="center">
  <img src="/assets/figures/2026-07-30-distributed-layerwise-offload/dlo_pipeline_last_frame.png" alt="DLO double-buffer pipeline overlapping compute, host-to-device copies, and AllGather" width="100%">
</p>

*Figure 4: DLO prepares layer N+1 with H2D and optional AllGather while layer N
computes, alternating between two bounded device slots. Reused from the
[official DLO post](https://vllm.ai/blog/2026-08-17-distributed-layerwise-offload).*

DLO is not a universal speed flag. Its value depends on interconnect, DP/SP
topology, resident-layer count, request concurrency, host memory, and host
bandwidth. This post explains that contract without adding another hardware
matrix; use the [full H3 recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3.md)
or the [RTX PRO 5000 recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3-RTX-PRO-5000.md)
to select and qualify AllGather or rank-local offload for a target system.

### 5.2 Disaggregated encoding

MiniMax H3 retains approximately 51.5 GB of Qwen3-VL encoder weights in BF16.
The [disaggregated encoder path](https://github.com/vllm-project/vllm-omni/pull/5885)
moves that encoder into an independent vLLM stage:

- Stage 0 runs the Qwen3-VL encoder with its own tensor-parallel topology,
  scheduler, kernels, and prefix cache.
- The orchestrator receives Stage 0 output, and `text_encoder2diffusion` merges
  layer-50 hidden states and token-role tags with the original prompt/media.
- Stage 1 runs H3 diffusion inline without loading a second local encoder.

<p align="center">
  <img src="/assets/figures/2026-08-29-minimax-h3-production-serving/h3-encoder-disaggregation.svg" alt="MiniMax H3 request flow through an independently scaled vLLM-native encoder and diffusion stage" width="100%">
</p>

*Figure 5: Every data-flow arrow runs left to right. The encoder owns its
processor, TP/replicas, and prefix cache; the orchestrator combines its typed
conditioning with the original request before dispatching the separately
parallelized DiT/VAE stage. Adapted from vLLM-Omni
[#5885](https://github.com/vllm-project/vllm-omni/pull/5885) and
[RFC #5707](https://github.com/vllm-project/vllm-omni/issues/5707).*

The merged single-node deployment does **not** configure OmniConnector: Stage 0
output returns to the orchestrator, the adapter builds the enriched diffusion
prompt there, and `InlineStageDiffusionClient` owns the Stage 1 DiffusionEngine
in the orchestrator process. RFC #5707 sketches OmniConnector with SHM/RDMA as
a future or cross-node transport option, but that path is outside the current
recipe.

This boundary is primarily a production-architecture feature. It lets encoder
and diffusion capacity scale independently, enables prefix reuse for repeated
presentations, and avoids serializing a decoded raw-video payload across a
process boundary when the diffusion stage is kept inline.

### 5.3 Step execution: functional, not yet beneficial

The merged [MiniMax H3 step-execution path](https://github.com/vllm-project/vllm-omni/pull/5810)
lets the scheduler admit, retire, and abort requests between denoise steps.
For H3, this is currently a functional scheduling capability—not a production
optimization or recommendation.

Existing simultaneous and staggered-arrival tests reduced throughput and
increased latency: one H3 request already presents a long, compute-bound packed
sequence, so co-batching makes each forward more expensive. Keep request mode
as the production default. The remaining useful hypotheses—fast cancellation
and HBM reclamation, low-rate admission, and co-batching only for small
under-utilized requests—are tracked in
[roadmap issue #5700](https://github.com/vllm-project/vllm-omni/issues/5700).
Step mode currently requires `FLASH_ATTN`, does not support cache backends, and
is incompatible with H3 DLO.

### 5.4 Compatibility stays living

Base H3 serves T2VA, FL2VA, and Ref2VA; Turbo serves T2VA/FL2VA; the current
FastH3 path is T2VA-only. Cross-feature composition changes too quickly for
a release-oriented blog snapshot, so the living
[MiniMax H3 feature×feature matrix in issue #5700](https://github.com/vllm-project/vllm-omni/issues/5700)
is the authoritative source for supported, partial, incompatible, and
unverified combinations. The task and hardware commands remain in the
[maintained H3 recipes](https://github.com/vllm-project/vllm-omni/tree/main/recipes/MiniMaxAI),
while Section 3.4 records the merged output-layout and MP4 boundaries.

## 6. Four-step LoRA deployment recommendations

This section recommends the merged, request-switchable Turbo path for
production planning. FastH3 also executes four transformer forwards, but its
merged vLLM-Omni integration is a T2VA-only load-time-fusion path rather than a
request-switchable adapter, so it is not used for the general production
profile recommendation.

Preload and allowlist the Turbo artifact, keep one adapter active per request,
and use its published five sigma points, four DiT forwards, video flow shift 6,
and audio flow shift 3. Start with dense attention; add quantization, caching,
or sparse attention only as a separately qualified composition.

### 6.1 Candidate eight-GPU profiles

The profiles below are benchmark candidates, not results. DP and USP describe
the complete eight-GPU factorization; encoder TP and VAE PP are per replica
unless the row uses model TP across the full node.

| B300 objective | Encoder | DiT parallelism | Video VAE | Weights / attention | Status |
|---|---|---|---|---|---|
| Lowest latency | TP8 | DP1 × TP1 × USP8, Ring1 | PP8 tile | Resident BF16 + Turbo; selected dense backend and Ulysses transport from Section 3 | Candidate |
| Node throughput | TP2 per replica | DP4 × TP1 × USP2, Ring1 | PP2 per replica | Four resident Turbo replicas; identical backend and transport across replicas | Candidate; compare with latency row |

Wider USP usually favors one-request latency; more DP replicas favor node
throughput. USP adds activation collectives, while DP duplicates request-local
state and requires enough concurrent arrivals.

### 6.2 Four-step stage decomposition

For Turbo, the client-visible pipeline is:

`T_client = T_queue + T_encoder + 4 × T_DiT_forward + T_video_VAE + T_audio_VAE + T_transport + T_MP4 + T_residual`

Reducing 49 denoiser evaluations to four changes the bottleneck. Encoder,
VAE, transport, and CPU muxing no longer disappear in the noise, so a four-step
headline without stage decomposition is incomplete.

| Platform / profile | Encoder | DiT total / 4 / per-forward | Video VAE | Audio VAE | Transport | CPU MP4 wall / process CPU | Client E2E / residual | Outputs/hour |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B300 / latency | TBD | TBD / 4 / TBD | TBD | TBD | TBD | TBD / TBD | TBD / TBD | TBD |
| B300 / throughput wave | TBD | TBD / 4 / TBD | TBD | TBD | TBD | TBD / TBD | TBD / TBD | TBD |

<!-- FIGURE TODO: Render one stacked critical-path bar per selected four-step
     profile using encoder, four DiT forwards, video/audio VAE, transport, CPU
     MP4, and residual. Keep queue time separate from the single-request bar. -->

For a DP wave, report both critical-path request latency and per-node
throughput; do not divide aggregate wall time by DP and call it latency. For
disaggregated serving, split encoder compute from handoff wait and state whether
the prompt hit the prefix cache. For every row, retain the adapter SHA, scale,
resident LoRA HBM, sigma points, actual forwards, and same-seed quality result.

### 6.3 Five-, ten-, and fifteen-second generation-speed reference

After Section 4.4 selects a publishable four-step path, run exactly one duration
sweep with its lowest-latency B300 topology. Turbo is the default production
candidate; if FastH3 is selected instead, name the fused artifact and use that
same path for all three rows. Do not sweep every acceleration combination.
The hypothesis is that fixed encoder and serving overheads are amortized as the
clip grows, so the selected profile's real-time factor stays flat or improves.

At 24 FPS, the frozen H3 implementation rounds requested seconds to frames and
then aligns upward to the model's `17n+5` boundary:

| Requested | Requested frames | Aligned frames | Nominal video duration |
|---:|---:|---:|---:|
| 5.0 s | 120 | 124 | 5.167 s |
| 10.0 s | 240 | 243 | 10.125 s |
| 15.0 s | 360 | 362 | 15.083 s |

Hold the prompt, seed, resolution, FPS, adapter and schedule, attention backend,
TP/USP/DP/Ring groups, VAE PP, compile policy, output path, and CPU affinity
fixed; duration is the only independent variable. Record one excluded
feasibility/compile request per shape, prewarm the 15-second maximum before any
CUDA-graph capture, then collect two runs in the fixed interleaved order
5→10→15→5→10→15 seconds. The 10-second row may reuse the canonical
measurement only when every control and timing boundary matches.
Stop a duration before repeated measurement on the Section 2 feasibility
conditions and retain the failed row. A real-time claim requires valid media
and `RTF_client ≤ 1.0` in both measured repetitions.

| Requested / aligned / nominal video | Encoder | DiT total / 4 / per-forward | Video/audio VAE | Transport + CPU MP4 | Client E2E run 1 / run 2 | Validated MP4 duration | Client RTF / × real time | Peak HBM | Evidence |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 5.0 s / 124 / 5.167 s | TBD | TBD / 4 / TBD | TBD / TBD | TBD | TBD / TBD | TBD | TBD / TBD | TBD | TBD |
| 10.0 s / 243 / 10.125 s | TBD | TBD / 4 / TBD | TBD / TBD | TBD | TBD / TBD | TBD | TBD / TBD | TBD | TBD |
| 15.0 s / 362 / 15.083 s | TBD | TBD / 4 / TBD | TBD / TBD | TBD | TBD / TBD | TBD | TBD / TBD | TBD | TBD |

Let `T_media` be the validated MP4 playback duration from `ffprobe`, retaining
the video and audio stream durations beside it. Report
`RTF_client = T_client / T_media` and `×_real_time = T_media / T_client`.
`RTF_client ≤ 1.0` means the complete MP4 was produced faster than playback;
it is **not** a live-streaming claim or a time-to-first-frame measurement.

### 6.4 Deployment decision rule

- Choose the one-replica wide-USP profile for the lowest validated request
  latency.
- Choose a DP profile only when the arrival process can keep its replicas busy;
  report P95 from the declared multi-request run.
- Use encoder disaggregation when independent scaling or prompt-prefix reuse
  offsets its orchestration cost.
- Re-profile VAE and CPU MP4 after every denoise acceleration; at four forwards
  they can determine the complete response latency.
- Treat Fast Ulysses as an opt-in transport A/B, not a universal default:
  include JIT/readiness time separately and warm the maximum serving shape
  before any compile or CUDA-graph capture.

## 7. Results and deployment recommendations

The final B300 recommendation will draw from the lossless Diffusers/vLLM-Omni
A/B in Section 3.5, isolated acceleration results in Section 4.4, the four-step
critical path in Section 6.2, and the duration-scaling reference in Section 6.3.
It will name the lowest complete-MP4 latency profile and the highest measured
node-throughput profile, with peak HBM/host RAM and quality gates beside each
claim.

Do not infer a recommendation from nominal FLOPS, multiply gains from unrelated
microbenchmarks, or treat a cold request as steady state. H200, RTX PRO 5000,
consumer GPU, ROCm, and NPU deployments are intentionally not ranked here; use
the recipes in Section 2.3 and validate against the local service objective.

## 8. RL integration with VeRL-Omni

vLLM-Omni also serves as the rollout engine for MiniMax H3 post-training in
[VeRL-Omni](https://github.com/verl-project/verl-omni). Current integrations
cover H3 DiffusionNFT and FlowGRPO paths, preserve joint video/audio rollouts,
and feed CLAP and ImageBind rewards before synchronizing full-weight or LoRA
updates back to the optimized rollout model. The resulting policy can return
to the same production serving stack described above.

This post treats RL as an ecosystem integration, not a serving benchmark. See
the [MiniMax H3 VeRL-Omni recipe](https://github.com/verl-project/verl-omni/blob/main/examples/diffusionnft_trainer/minimax_h3/README.md)
for the training architecture, data preparation, rewards, and launch commands.

## 9. Production readiness

### Promotion gate

The living
[feature×feature matrix in issue #5700](https://github.com/vllm-project/vllm-omni/issues/5700)
describes software compatibility; it does not replace platform qualification.
Promote a profile only after its frozen workload passes the stage-level timing,
memory, quality, media-validation, and operational checks in Sections 2–7.
Ascend NPU remains intentionally unclassified until vendor-aligned evidence is
available.

### Operational and security considerations

- Report model download, weight preparation, compilation, and readiness
  separately from warmed request latency.
- Treat HBM, host RAM, checkpoint storage, and output-transport memory as
  independent capacity budgets.
- Preload or allowlist production LoRA adapters. Do not expose arbitrary
  request-supplied adapter paths to an untrusted endpoint.
- Treat the canonical results as T2VA evidence only. FL2VA and Ref2VA require
  separate validation before their recipes are promoted to production SLOs.
- Keep generated-media safety controls and abuse-reporting mechanisms outside
  the model server's performance-critical path, but inside the production
  service boundary.

### Licensing

MiniMax H3 is distributed under the
[MiniMax H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE),
not an unconditional permissive model license. Commercial and hosted-service
operators should review the current territorial, attribution, revenue,
acceptable-use, and safeguard requirements with their own counsel before
deployment.

## Roadmap

- complete the lossless Diffusers-versus-vLLM-Omni A/B, four-step profiles, and
  5/10/15-second generation-speed reference on 8× B300 with reconciled timing
  boundaries;
- qualify FastH3, SAGE/Skip-Softmax, and Sol-Attn against released dense
  baselines and multi-seed audio/video gates;
- complete native SVDQuant performance kernels and validation;
- identify and validate a useful H3 step-execution case—cancellation,
  staggered-arrival admission, or small-workload co-batching—or retain the
  explicit no-production-benefit conclusion;
- keep H200, RTX PRO 5000, consumer GPU, ROCm, and NPU guidance in maintained
  recipes rather than extending this benchmark matrix; and
- continue hardening disaggregated serving, output transport, step-level
  control, and RL rollout integration.

## Acknowledgments

<!-- AUTHOR TODO: Add named benchmark collaborators, hardware vendors,
     MiniMax/FastH3 contributors, PR authors, and reviewers after
     the final evidence and author list are agreed. -->

This work builds on contributions across vLLM, vLLM-Omni, VeRL-Omni, MiniMax
H3, FastH3, Diffusers, and NVIDIA. We especially thank the
[FastH3 team](https://haoailab.com/blogs/fasth3-preview/) for open-sourcing its
four-step adapter and collaborating with the vLLM-Omni community on the merged
serving integration. We also thank the contributors who implemented and
validated the model, serving, quantization, offload, kernel, VAE, media,
hardware, and training paths referenced throughout this post.

## References

- [vLLM-Omni repository](https://github.com/vllm-project/vllm-omni)
- [MiniMax H3 model](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- [Diffusers MiniMax H3 pipeline](https://huggingface.co/docs/diffusers/v0.40.0/api/pipelines/minimax_h3)
- [MiniMax H3 serving recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3.md)
- [Distributed Layerwise Offload](https://vllm.ai/blog/2026-08-17-distributed-layerwise-offload)
- [Diffusion execution modes](https://github.com/vllm-project/vllm-omni/blob/main/docs/user_guide/diffusion/execution_modes.md)
- [Online FP8 explainer and editable figure sources](https://github.com/hsliuustc0106/vllm-omni-cookbook/blob/main/blog/_posts/2026-08-18-online-quantization-fp8.md)
- [MiniMax H3 SVDQuant explainer](https://github.com/hsliuustc0106/vllm-omni-cookbook/blob/main/blog/_posts/2026-08-16-understanding-pr-6162-svdquant-w4a4-blackwell.md)
- [VeRL-Omni repository](https://github.com/verl-project/verl-omni)
