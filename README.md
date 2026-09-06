# Scooter Parking Detection

A repository to fine-tune and benchmark vision models on the Kaggle [Improper Scooter Parking Detection](https://www.kaggle.com/datasets/prodigyanalysis/improper-scooter-parking-detection) dataset.

## Quick Start

### 1. Prepare Dataset

Downloads the dataset via `kagglehub` and organizes images into `data/train` and `data/val` folders:

```bash
uv run data.py
```

### 2. Fine-Tune Vision Classifiers

```bash
# Train all models defined in models.py
uv run train.py --all --epochs 10

# Train a single model (e.g. resnet18, vit, dinov2, mobilenetv3, efficientnetb0, convnextv2)
uv run train.py --model resnet18 --epochs 10
```

### 3. Benchmark Vision Classifier Quality & Speed

Evaluates accuracy, F1-score, and GPU latency/throughput:

```bash
uv run benchmark.py
```

### 4. Vision-Language Models (VLMs) & Actionable Feedback

Evaluate multimodal VLMs (zero-shot) for proper/improper classification and actionable natural language feedback:

> Gated models (e.g. Gemma 4) require a Hugging Face token—copy `.env.example` to `.env` and set `HF_TOKEN`.

Supported models:

- `lfm2.5-vl-450m` (Liquid AI LFM2.5-VL 450M, 2026, FP16)
- `lfm2.5-vl-1.6b` (Liquid AI LFM2.5-VL 1.6B, 2026, FP16)
- `lfm2.5-vl-3b` (Liquid AI LFM2.5-VL 3B, 2026, 4-bit)
- `ministral-3-3b-instruct-2512` (Mistral 3 3B Instruct 2512, 2026, 4-bit)
- `qwen3.5-0.8b` (Qwen3.5 0.8B, 2026, FP16)
- `qwen3.5-2b` (Qwen3.5 2B, 2026, 4-bit)
- `qwen3.5-4b` (Qwen3.5 4B, 2026, 4-bit)
- `qwen3-vl-2b-instruct` (Qwen3-VL 2B Instruct, 2026, 4-bit)
- `gemma-4-e2b-it` (Google Gemma 4 E2B-it, April 2026, 4-bit)
- `minicpm-v-4_6` (MiniCPM-V 4.6, 2026, FP16)
- `smolvlm2-2.2b-instruct` (SmolVLM2 2.2B Instruct, 4-bit)
- `smolvlm2-500m-instruct` (SmolVLM2 500M Instruct, FP16)
- `qwen2.5-vl-3b-instruct` (Qwen2.5-VL 3B Instruct, 4-bit)
- `internvl3_5-1b` (InternVL 3.5 1B, 2026, FP16)

```bash
# Benchmark any supported model
uv run vlm_benchmark.py --model <model-name> --split [train|val]

# Force re-evaluation even if results already exist in output JSON
uv run vlm_benchmark.py --model <model-name> --split [train|val] --force

# Or evaluate all models in sequence (skips already evaluated models by default)
uv run vlm_benchmark.py --model all --split [train|val]
```

#### Outputs

- `vlm_output/vlm_predictions_{split}.json`

where `{split}` is the dataset split (`train` or `val`) that was evaluated.

#### View Results in Browser

Serve from the repository root:

```bash
uv run python -m http.server 8000
```

Then open [http://localhost:8000/gallery/](http://localhost:8000/gallery/).

### 5. Attempted Models & Technical Findings

During exploration of modern compact VLMs (2025–2026 releases) under local hardware constraints (NVIDIA GTX 1660 SUPER 6 GB VRAM), several candidate models were evaluated but could not be integrated due to hardware limits, library incompatibilities, or architectural degeneracy:

| Model                                                          | Release     | Status                         | Reason                                                                                                        |
| -------------------------------------------------------------- | ----------- | ------------------------------ | ------------------------------------------------------------------------------------------------------------- |
| **VisionPsy-Nano-460M** (`qvac/VisionPsy-Nano-460M`)           | 2025 / 2026 | ❌ Custom Code & Triton Issues | Hardcodes `torch.compile` (crashes on Windows without Triton) and has FP16 rotary embedding mismatches.       |
| **Qwen 3.6 (`Qwen/qwen36`)**                                   | 2026        | ❌ Hardware Limit (OOM)        | Only 27B+ dense and 35B MoE variants available; no compact sub-10B models exist.                              |
| **Qwen 3.8 (`Qwen/qwen38`)**                                   | 2026        | ❌ Hardware Limit (OOM)        | Massive datacenter models (27B and 2.4T MoE) far exceeding 6 GB VRAM capacity.                                |
| **YannQi/R-4B** (`YannQi/R-4B`)                                | 2025 / 2026 | ❌ Bottlenecked & Incompatible | Incompatible with `transformers 5.x` tied weights; 4-bit spikes to 5.8 GB VRAM at ~87s/image.                 |
| **Apple FastVLM** (`apple/FastVLM-0.5B`, `apple/FastVLM-1.5B`) | 2025        | ❌ Generation Degeneracy Bug   | Multimodal projection fails during autoregressive decode in `transformers 5.x`, outputting exclamation marks. |
| **Moonshot Kimi-VL-A3B** (`moonshotai/Kimi-VL-A3B-Instruct`)   | 2025        | ❌ Hardware Limit (OOM)        | 64-expert MoE architecture (~16B+ total parameters) requiring >10 GB VRAM even in 4-bit.                      |
| **AllenAI MolmoE (1B active)** (`allenai/MolmoE-1B-0924`)      | Late 2024   | ❌ Hardware Limit (OOM)        | MoE architecture totaling 26.9 GB of underlying weights, exceeding 6 GB VRAM.                                 |
| **DeepSeek Janus-Pro-1B** (`deepseek-community/Janus-Pro-1B`)  | Jan 2025    | ❌ Generation Mismatch         | Fails standard `.generate()` in `transformers 5.x` due to custom `<image_placeholder>` expansion.             |
| **Moondream2** (`vikhyatk/moondream2`)                         | 2024 / 2025 | ❌ Library Incompatible        | Incompatible with `transformers 5.x` (`all_tied_weights_keys` attribute error).                               |
| **Microsoft Florence-2-Large** (`microsoft/Florence-2-large`)  | 2024        | ⚠️ Task-Token Mismatch         | Specialized token encoder-decoder (`<OD>`, `<CAPTION>`) lacking conversational JSON instruction following.    |
