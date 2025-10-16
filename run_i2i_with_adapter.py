# run_i2i_with_adapter.py
import os
import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
from torchvision.transforms import ToTensor, ToPILImage
import numpy as np

# optional: safetensors loader
try:
    from safetensors.torch import load_file as safeload
except Exception:
    safeload = None

# ---------------------------
# helper: 유연한 adapter 로더
# ---------------------------
def load_adapter_weights_to_pipe(adapter_path: str, pipe: StableDiffusionImg2ImgPipeline, device="cuda"):
    """
    1) 먼저 IP-Adapter 저장소의 helper 함수가 있다면 그것을 시도합니다 (import 실패해도 무시).
    2) 없다면 safetensors / torch.load로 weight들을 읽고 파이프라인의 여러 모듈(UNet, image_encoder, text_encoder, VAE 등)
       중 키가 일치하는 곳에 매칭해서 로드합니다. (strict=False 방식으로)
    """
    # 1) 시도: ip-adapter repo의 helper가 설치되어 있으면 사용
    try:
        # many community repos expose helper like ip_adapter.load_adapter or pipe.load_adapter
        import ip_adapter
        try:
            # try the common helper
            if hasattr(pipe, "load_adapter"):
                pipe.load_adapter(adapter_path)
                print("Loaded adapter using pipe.load_adapter()")
                return
            elif hasattr(ip_adapter, "load_adapter_to_pipe"):
                ip_adapter.load_adapter_to_pipe(adapter_path, pipe)
                print("Loaded adapter using ip_adapter.load_adapter_to_pipe()")
                return
        except Exception:
            # fallthrough to generic loader
            pass
    except Exception:
        pass

    # 2) Generic loader
    print("Using generic adapter loader...")

    # load state dict (safetensors preferred)
    if adapter_path.endswith(".safetensors") and safeload is not None:
        adapter_state = safeload(adapter_path)
    else:
        # torch.load fallback (may be .pt or .pth)
        adapter_state = torch.load(adapter_path, map_location="cpu")

    # adapter_state might be a mapping of names->tensor or {'state_dict': {...}}
    if isinstance(adapter_state, dict) and "state_dict" in adapter_state and isinstance(adapter_state["state_dict"], dict):
        adapter_state = adapter_state["state_dict"]

    # ensure tensors are torch tensors
    adapter_state = {k: (v.cpu() if isinstance(v, torch.Tensor) else torch.tensor(v)) for k, v in adapter_state.items()}

    # candidate modules to try loading into (order matters)
    candidates = [
        ("unet", pipe.unet),
        ("image_encoder", getattr(pipe, "image_encoder", None)),  # if pipeline contains
        ("text_encoder", getattr(pipe, "text_encoder", None)),
        ("vae", getattr(pipe, "vae", None)),
    ]

    # For each candidate, try to build intersecting state_dict and load with strict=False
    for name, module in candidates:
        if module is None:
            continue
        module_sd = module.state_dict()
        # two matching strategies: exact keys, or keys with prefix (e.g., "unet.") removed
        matched = {}
        for k, v in adapter_state.items():
            # try exact match
            if k in module_sd and module_sd[k].shape == v.shape:
                matched[k] = v
                continue
            # try removing common prefix like "unet.", "text_encoder.", etc.
            if "." in k:
                _, rest = k.split(".", 1)
                if rest in module_sd and module_sd[rest].shape == v.shape:
                    matched[rest] = v
                    continue
            # try matching with name prefix
            pref = f"{name}.{k}"
            if pref in module_sd and module_sd[pref].shape == v.shape:
                matched[pref] = v

        if len(matched) == 0:
            print(f"[loader] No weights matched for module '{name}' (skipping).")
            continue

        # convert tensors to appropriate dtype and device
        processed = {}
        for k, v in matched.items():
            # module expects params on cpu for load_state_dict; we'll load then move module to device later
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v)
            processed[k] = v

        # load into module
        try:
            # prepare a state dict with only the module's keys -> load with strict=False
            module.load_state_dict(processed, strict=False)
            print(f"[loader] Loaded {len(processed)} weights into {name}")
        except Exception as e:
            # fallback: try to update module.state_dict() then load
            try:
                sd = module.state_dict()
                sd.update(processed)
                module.load_state_dict(sd, strict=False)
                print(f"[loader] (fallback) Loaded {len(processed)} weights into {name}")
            except Exception as e2:
                print(f"[loader] Failed to load into {name}: {e} / {e2}")

    # Done
    print("Generic adapter load finished. Note: if behavior is wrong, prefer repo-provided loader.")


# ---------------------------
# main: pipeline 생성 및 i2i 실행
# ---------------------------
def run_image_to_image(
    pretrained_model="runwayml/stable-diffusion-v1-5",
    adapter_path="D:/backup/finetuned_i2i/checkpoint-10000/model.safetensors",
    init_image_path="D:/backup/test/hello.jpg",
    prompt="anime style portrait",
    output_path="D:/backup/test.png",
    device="cuda",
    strength=0.7,
    guidance_scale=7.5,
    num_inference_steps=50,
    seed=42,
    use_fp16=True
):
    # device & dtype
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if (use_fp16 and device.type == "cuda") else torch.float32

    # load pipeline
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        pretrained_model,
        torch_dtype=dtype,
        safety_checker=None,          # safety checker off (optional)
        revision=None,
    ).to(device)

    # load adapter weights
    if adapter_path:
        load_adapter_weights_to_pipe(adapter_path, pipe, device=device)

    # prepare input image
    init_image = Image.open(init_image_path).convert("RGB")
    # resize to model resolution if you want:
    # init_image = init_image.resize((512,512), resample=Image.LANCZOS)

    generator = torch.Generator(device).manual_seed(seed) if device.type == "cuda" else torch.Generator().manual_seed(seed)

    # run
    with torch.autocast(device.type) if device.type == "cuda" and use_fp16 else torch.no_grad():
        result = pipe(
            prompt=prompt,
            image=init_image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )

    # save
    result.images[0].save(output_path)
    print(f"Saved: {output_path}")


# ---------------------------
# 실행 예시
# ---------------------------
if __name__ == "__main__":
    # 경로들 필요에 맞게 바꿔서 사용하세요
    run_image_to_image(
        pretrained_model="runwayml/stable-diffusion-v1-5",
        adapter_path="D:/backup/finetuned_i2i/checkpoint-20000/model.safetensors",  # 학습으로 생성된 .safetensors 경로
        init_image_path="D:/backup/test/hello.jpg",
        prompt="A bespectacled boy smiling with a wine glass on the ocean view veranda drawn by KSH Drawing Style",
        output_path="D:/backup/test.png",
        device="cuda",
        strength=0.7,
        guidance_scale=7.5,
        num_inference_steps=30,
        seed=1234,
        use_fp16=True
    )
