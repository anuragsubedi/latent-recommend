"""ACE-Step VAE-only latent extraction helpers.

These functions intentionally avoid ACE-Step's full inference handlers. Loading
only `AutoencoderOobleck` keeps the extraction path usable on Colab GPUs.
"""

from __future__ import annotations

import gc
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch


def cleanup_device() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_standalone_vae(device: str = "cuda"):
    from diffusers.models import AutoencoderOobleck

    vae = AutoencoderOobleck.from_pretrained(
        "ACE-Step/Ace-Step1.5",
        subfolder="vae",
        torch_dtype=torch.float32,
    )
    vae.to(device)
    vae.eval()
    return vae


def prepare_waveform_for_oobleck(
    waveform: torch.Tensor,
    sample_rate: int,
    target_sr: int = 48000,
    device: str = "cuda",
) -> torch.Tensor:
    import torchaudio

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=target_sr,
        )
        waveform = resampler(waveform)

    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2, :]

    return waveform.unsqueeze(0).to(device)


def process_audio_for_oobleck(
    file_path: str | Path,
    target_sr: int = 48000,
    device: str = "cuda",
) -> torch.Tensor:
    import torchaudio

    waveform, sample_rate = torchaudio.load(str(file_path))
    return prepare_waveform_for_oobleck(
        waveform,
        sample_rate=sample_rate,
        target_sr=target_sr,
        device=device,
    )


def process_hf_audio_for_oobleck(
    audio: dict,
    target_sr: int = 48000,
    device: str = "cuda",
) -> torch.Tensor:
    array = np.asarray(audio["array"], dtype="float32")
    waveform = torch.from_numpy(array)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.T
    return prepare_waveform_for_oobleck(
        waveform,
        sample_rate=int(audio["sampling_rate"]),
        target_sr=target_sr,
        device=device,
    )


def extract_track_embedding(
    vae,
    waveform: torch.Tensor,
    chunk_duration_sec: int = 10,
    sr: int = 48000,
) -> np.ndarray:
    chunk_size = int(chunk_duration_sec * sr)
    latents = []

    with torch.no_grad():
        for start in range(0, waveform.shape[-1], chunk_size):
            chunk = waveform[:, :, start : start + chunk_size]
            if chunk.shape[-1] < sr:
                continue
            latent = vae.encode(chunk).latent_dist.mode()
            latents.append(latent.detach())
            del chunk, latent
            cleanup_device()

    if not latents:
        raise ValueError("Audio is too short to produce a latent embedding.")

    full_latent = torch.cat(latents, dim=2)
    pooled = full_latent.mean(dim=2).squeeze(0)
    embedding = pooled.detach().cpu().numpy().astype("float32")
    del full_latent, pooled, latents
    cleanup_device()

    if embedding.shape != (64,):
        raise ValueError(f"Expected a 64-D embedding, received shape {embedding.shape}.")
    if not np.isfinite(embedding).all():
        raise ValueError("Embedding contains NaN or infinite values.")
    return embedding


def save_reconstruction(vae, waveform: torch.Tensor, output_path: str | Path, sr: int = 48000) -> None:
    import torchaudio

    with torch.no_grad():
        latents = vae.encode(waveform).latent_dist.mode()
        reconstructed = vae.decode(latents).sample
        torchaudio.save(str(output_path), reconstructed.squeeze(0).cpu(), sample_rate=sr)
    cleanup_device()


def export_preview_mp3(
    waveform: torch.Tensor,
    output_path: str | Path,
    sr: int = 48000,
    start_sec: int = 0,
    duration_sec: int = 20,
    bitrate: str = "64k",
) -> None:
    """Write a small MP3 preview using ffmpeg when available."""

    import torchaudio

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    start = int(start_sec * sr)
    stop = start + int(duration_sec * sr)
    snippet = waveform.squeeze(0).detach().cpu()[:, start:stop]

    with tempfile.NamedTemporaryFile(suffix=".wav") as temp_wav:
        torchaudio.save(temp_wav.name, snippet, sample_rate=sr)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                temp_wav.name,
                "-b:a",
                bitrate,
                str(output_path),
            ],
            check=True,
        )
