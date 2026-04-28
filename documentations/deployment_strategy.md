# Deployment Strategy

The deployed dashboard should be lightweight and CPU-only. The expensive
ACE-Step VAE extraction belongs in Colab; deployment only loads finished
artifacts.

## Recommended Host

Use Hugging Face Spaces with the Streamlit SDK.

Recommended app command:

```bash
streamlit run app.py
```

Why this is the default:

- Streamlit is already the dashboard framework.
- Hugging Face is a natural home for model-adjacent artifacts and datasets.
- A separate Hugging Face Dataset repo can host larger previews or artifact
  bundles if GitHub/Git LFS becomes inconvenient.

## Artifact Options

Option A: Git LFS in this repository.

- Best for moderate artifact bundles.
- `.gitattributes` already marks FAISS, SQLite, NumPy, parquet, and MP3 outputs
  for LFS.
- Keep generated files under `artifacts/`.

Option B: Hugging Face Dataset repository.

- Best if MP3 previews make the repo too large.
- Upload the generated `artifacts/merged` bundle to a dataset repo.
- Add a small pre-deploy step that downloads the bundle into `artifacts/merged`
  before the app starts, or commit a reduced demo bundle directly to the Space.

Option C: Reduced demo Space.

- Best if the final 1,734-track artifact bundle exceeds free-tier limits.
- Deploy a representative subset, for example 50-100 tracks per pole.
- Keep the full artifact metrics in the poster/video even if the public Space is
  a smaller demo.

## Pre-Deployment Checklist

- Run `.venv/bin/python scripts/rebuild_metadata_model.py --artifacts artifacts/merged`.
- Run `.venv/bin/python scripts/validate_artifacts.py --artifacts artifacts/merged`.
- Run `.venv/bin/python scripts/evaluate_artifacts.py --artifacts artifacts/merged --k 10`.
- Run `.venv/bin/streamlit run app.py` locally.
- Confirm `artifacts/merged/metrics.json` contains Precision@K, playlist
  completion, MRR, triplet success, centroid separation, and ablation results.
- Confirm dashboard boot time and memory are acceptable with the final artifact
  bundle.
- Confirm missing preview files show captions instead of crashing.

## Free-Tier Constraints

- Do not install `torch`, `torchaudio`, `diffusers`, or `transformers` in the
  deployed app unless the Space is specifically intended for extraction.
- Keep `requirements.txt` limited to dashboard/evaluation dependencies.
- Prefer `faiss-cpu`, SQLite, NumPy, and Plotly for low-overhead runtime.
- Store only compressed preview snippets, not full uncompressed audio.
