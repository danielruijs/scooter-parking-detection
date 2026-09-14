# VLM Benchmark Gallery

Static web gallery for visual exploration and comparison of zero-shot VLM predictions.

## Running Locally

Serve the gallery:

```bash
uv run python -m http.server 8000
```

Then open [http://localhost:8000/](http://localhost:8000/).

## Regenerating Thumbnails

To regenerate web thumbnails from `data/`:

```bash
uv run generate_thumbnails.py
```
