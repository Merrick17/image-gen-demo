#!/usr/bin/env python3
"""
Download SDXL LoRA .safetensors files into models/ before the server starts.
All repos below are confirmed SDXL-compatible (base: stabilityai/stable-diffusion-xl-base-1.0).
"""
import os
import sys

from huggingface_hub import hf_hub_download, login
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

MODELS_DIR = os.getenv("MODELS_DIR", "/app/models")
HF_TOKEN = os.getenv("HF_TOKEN", "")
os.makedirs(MODELS_DIR, exist_ok=True)

if HF_TOKEN:
    login(token=HF_TOKEN, add_to_git_credential=False)
    print(f"  Authenticated with HuggingFace Hub")

LORAS = [
    # ── Speed (always loaded — enables 4-step CPU inference via LCM) ──────────
    {
        "name": "lcm_lora.safetensors",
        "repo_id": "latent-consistency/lcm-lora-sdxl",
        "filename": "pytorch_lora_weights.safetensors",
        "category": "speed/LCM",
    },

    # ── Photography / Realistic ────────────────────────────────────────────────
    {
        "name": "photography.safetensors",
        "repo_id": "ostris/photorealistic-slider-sdxl-lora",
        "filename": "sdxl_photorealistic_slider_v1-0.safetensors",
        "category": "photography",
    },

    # ── Fashion / Editorial ────────────────────────────────────────────────────
    {
        "name": "fashion_avantgarde.safetensors",
        "repo_id": "KappaNeuro/avant-garde-fashion",
        "filename": "Avant-garde Fashion.safetensors",
        "category": "fashion (avant-garde)",
    },
    {
        "name": "fashion_selfie.safetensors",
        "repo_id": "artificialguybr/selfiephotographyredmond-selfie-photography-lora-for-sdxl",
        "filename": "SelfiePhotographyRedmond.safetensors",
        "category": "fashion (selfie/photography)",
    },
    {
        "name": "fashion_weird.safetensors",
        "repo_id": "Norod78/weird-fashion-show-outfits-sdxl-lora",
        "filename": "sdxl-WeirdOutfit-Dreambooh.safetensors",
        "category": "fashion (editorial/weird)",
    },

    # ── Fast Generation Alternative ────────────────────────────────────────────
    {
        "name": "dmd2_fast.safetensors",
        "repo_id": "tianweiy/DMD2",
        "filename": "dmd2_sdxl_4step_lora.safetensors",
        "category": "speed/DMD2",
    },
]


def download_lora(entry: dict) -> bool:
    dest = os.path.join(MODELS_DIR, entry["name"])
    if os.path.exists(dest):
        size_mb = os.path.getsize(dest) / 1_048_576
        print(f"  ✓ {entry['name']} already present ({size_mb:.1f} MB) — skipping")
        return True

    print(f"  ↓ [{entry['category']}] {entry['repo_id']} → {entry['name']} ...")
    try:
        path = hf_hub_download(
            repo_id=entry["repo_id"],
            filename=entry["filename"],
            local_dir=MODELS_DIR,
            token=HF_TOKEN or None,
        )
        flat_dest = os.path.join(MODELS_DIR, entry["name"])
        if os.path.abspath(path) != os.path.abspath(flat_dest):
            os.rename(path, flat_dest)
        size_mb = os.path.getsize(flat_dest) / 1_048_576
        print(f"  ✓ {entry['name']} saved ({size_mb:.1f} MB)")
        return True
    except (EntryNotFoundError, RepositoryNotFoundError) as exc:
        print(f"  ✗ SKIPPED {entry['name']}: {exc}", file=sys.stderr)
        return False
    except Exception as exc:
        print(f"  ✗ ERROR {entry['name']}: {exc}", file=sys.stderr)
        return False


def main() -> None:
    print(f"\nSDXL LoRA downloader — saving to {MODELS_DIR}\n")
    results = [download_lora(entry) for entry in LORAS]
    ok = sum(results)
    print(f"\nDone: {ok}/{len(LORAS)} LoRAs downloaded.\n")

    files = sorted(f for f in os.listdir(MODELS_DIR) if f.endswith(".safetensors"))
    if files:
        print("Available in models/:")
        for f in files:
            size_mb = os.path.getsize(os.path.join(MODELS_DIR, f)) / 1_048_576
            print(f"  {f}  ({size_mb:.1f} MB)")
    print()


if __name__ == "__main__":
    main()
