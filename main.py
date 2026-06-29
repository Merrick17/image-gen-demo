import base64
import io
import os

import torch
from diffusers import LCMScheduler, StableDiffusionXLPipeline
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import login
from PIL import Image
from pydantic import BaseModel, Field

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN", "")
MODEL_ID = os.getenv("MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0")
NUM_STEPS = int(os.getenv("NUM_STEPS", "4"))
GUIDANCE_SCALE = float(os.getenv("GUIDANCE_SCALE", "1.5"))
NUM_THREADS = os.cpu_count() or 4

USE_LCM = os.getenv("USE_LCM", "true").lower() == "true"
LCM_LORA_PATH = os.getenv("LCM_LORA_PATH", "models/lcm_lora.safetensors")

STYLE_LORA_PATH = os.getenv("LORA_PATH", "")
LORA_SCALE = float(os.getenv("LORA_SCALE", "0.8"))

torch.set_num_threads(NUM_THREADS)

if HF_TOKEN:
    login(token=HF_TOKEN, add_to_git_credential=False)

app = FastAPI(title="SDXL Image Generation API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

pipeline: StableDiffusionXLPipeline | None = None
active_loras: list[str] = []


@app.on_event("startup")
async def load_model() -> None:
    global pipeline, active_loras

    pipeline = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        token=HF_TOKEN or None,
    )
    pipeline.enable_attention_slicing()

    adapters: list[str] = []
    weights: list[float] = []

    if USE_LCM and os.path.exists(LCM_LORA_PATH):
        pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
        pipeline.load_lora_weights(LCM_LORA_PATH, adapter_name="lcm")
        adapters.append("lcm")
        weights.append(1.0)
        active_loras.append("lcm")

    if STYLE_LORA_PATH and os.path.exists(STYLE_LORA_PATH):
        pipeline.load_lora_weights(STYLE_LORA_PATH, adapter_name="style")
        adapters.append("style")
        weights.append(LORA_SCALE)
        active_loras.append(os.path.basename(STYLE_LORA_PATH))

    if adapters:
        pipeline.set_adapters(adapters, adapter_weights=weights)
        pipeline.fuse_lora()
        pipeline.unload_lora_weights()


class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = "blurry, low quality, distorted, watermark, deformed"
    n: int = Field(default=1, ge=1, le=4)
    width: int = Field(default=1024, ge=64, le=2048, multiple_of=8)
    height: int = Field(default=1024, ge=64, le=2048, multiple_of=8)
    num_inference_steps: int = Field(default=NUM_STEPS, ge=1, le=100)
    guidance_scale: float = Field(default=GUIDANCE_SCALE, ge=0.0, le=20.0)
    seed: int | None = None


class ImageResult(BaseModel):
    b64_json: str


class GenerateResponse(BaseModel):
    images: list[ImageResult]


def _image_to_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> GenerateResponse:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    generator = (
        torch.Generator(device="cpu").manual_seed(request.seed)
        if request.seed is not None
        else None
    )

    results: list[ImageResult] = []
    for _ in range(request.n):
        output = pipeline(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            height=request.height,
            width=request.width,
            guidance_scale=request.guidance_scale,
            num_inference_steps=request.num_inference_steps,
            generator=generator,
        )
        results.append(ImageResult(b64_json=_image_to_b64(output.images[0])))

    return GenerateResponse(images=results)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok" if pipeline is not None else "loading",
        "model": MODEL_ID,
        "device": "cpu",
        "num_threads": NUM_THREADS,
        "num_steps": NUM_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "lcm_enabled": USE_LCM,
        "loras": active_loras,
    }


@app.get("/loras")
async def list_loras() -> dict:
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    if not os.path.isdir(models_dir):
        return {"loras": []}
    return {"loras": sorted(f for f in os.listdir(models_dir) if f.endswith(".safetensors"))}
