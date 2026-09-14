# Vision-Language Models (VLMs) & Actionable Feedback

Evaluate modern multimodal Vision-Language Models (zero-shot) for proper/improper scooter parking classification and actionable natural language feedback generation.

## Supported Models

Supported models are defined in [`config.py`](config.py):

> Gated models (e.g. Gemma 4) require a Hugging Face token - copy `.env.example` to `.env` and set `HF_TOKEN`.

## Benchmarking VLMs

```bash
# Benchmark a single model on the validation split
uv run benchmark.py --model qwen3.5-2b --split val

# Benchmark all models sequentially
uv run benchmark.py --model all --split val

# Force re-evaluation even if results already exist in output JSON
uv run benchmark.py --model qwen3.5-2b --split val --force
```

### Outputs

Predictions and evaluation metrics are saved to:

- `output/vlm_predictions_{split}.json` (e.g. `val` or `train`)

## Interactive Web Gallery

The interactive web gallery allows visual exploration and side-by-side comparison of VLM predictions and generated feedback against ground truth.

To run the gallery locally, start a local HTTP server from the `vlms/` directory:

```bash
uv run python -m http.server 8000
```

Then open [http://localhost:8000/gallery/](http://localhost:8000/gallery/).

See [`gallery/README.md`](gallery/README.md) for instructions on regenerating thumbnails.

## Attempted Models & Technical Findings

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
