# VLM Benchmark Gallery

Static web gallery for visual exploration and comparison of zero-shot VLM predictions.

## Running Locally

Run the server from `vlms/` (so both `gallery/` and `output/` predictions can be accessed):

```bash
cd ..
uv run python -m http.server 8000
```

Then open [http://localhost:8000/gallery/](http://localhost:8000/gallery/).

## Regenerating Thumbnails

To regenerate web thumbnails from `data/`:

```bash
uv run generate_thumbnails.py
```
