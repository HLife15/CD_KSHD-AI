"""
run_img2img_with_lora.py

사용법 예시:
python run_img2img_with_lora.py

환경: GPU 권장 (CUDA), torch + diffusers, safetensors 설치
"""

import os
import torch
from safetensors.torch import load_file as load_safetensors
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
import warnings

# ---------------------------
# 설정 (사용자 필요에 따라 수정)
# ---------------------------
BASE_MODEL = "runwayml/stable-diffusion-v1-5"
LORA_PATH = "D:/backup/finetuned/pytorch_lora_weights.safetensors"   # 또는 .pt/.bin
INIT_IMAGE = "D:/backup/test/1.png"
OUT_IMAGE = "D:/backup/out_img.png"
PROMPT = "drawn by KSH Drawing Style"
DEVICE = "cuda"  # or "cpu"
STRENGTH = 0.7
GUIDANCE_SCALE = 7.5
NUM_INFERENCE_STEPS = 30
MERGE_INPLACE = True   # True면 모델 가중치에 LoRA를 직접 더함 (권장), False면 복사본에 적용
BACKUP_FILE = "model_backup.pth"  # MERGE_INPLACE True일 때 원본 백업 파일
# ---------------------------


def load_lora_state(lora_path):
    """safetensors 우선, 아니면 torch.load 사용."""
    lora_path = str(lora_path)
    if lora_path.endswith(".safetensors"):
        sd = load_safetensors(lora_path)
        # safetensors returns numpy arrays; convert to tensors
        sd = {k: torch.as_tensor(v) for k, v in sd.items()}
    else:
        sd_obj = torch.load(lora_path, map_location="cpu")
        # handle dict with 'state_dict'
        if isinstance(sd_obj, dict) and "state_dict" in sd_obj:
            sd_obj = sd_obj["state_dict"]
        sd = {k: v.cpu() if isinstance(v, torch.Tensor) else torch.as_tensor(v) for k, v in sd_obj.items()}
    return sd


def find_lora_pairs(lora_sd):
    """
    LoRA 키 규칙에서 up/down pair와 alpha를 찾아 그룹화.
    반환: dict base_name -> { 'up': tensor, 'down': tensor, 'alpha': float or None }
    base_name: 원래 파라미터 이름을 유추하기 위한 key-ish (내부 매칭용)
    """
    pairs = {}
    for k, v in lora_sd.items():
        # common conventions:
        #  - <module>.<...>.lora_up.weight
        #  - <module>.<...>.lora_down.weight
        #  - <module>.<...>.alpha  (optional)
        if "lora_up" in k and k.endswith("weight"):
            base = k.replace(".lora_up.weight", "")
            pairs.setdefault(base, {})["up"] = v
        elif "lora_down" in k and k.endswith("weight"):
            base = k.replace(".lora_down.weight", "")
            pairs.setdefault(base, {})["down"] = v
        elif k.endswith(".alpha"):
            base = k.replace(".alpha", "")
            pairs.setdefault(base, {})["alpha"] = float(v.item() if isinstance(v, torch.Tensor) else v)
        # some variants use "lora_A" / "lora_B"
        elif k.endswith(".lora_A") or k.endswith(".lora_A.weight"):
            base = k.rsplit(".", 1)[0]
            pairs.setdefault(base, {})["down"] = v
        elif k.endswith(".lora_B") or k.endswith(".lora_B.weight"):
            base = k.rsplit(".", 1)[0]
            pairs.setdefault(base, {})["up"] = v
    return pairs


def apply_lora_to_state_dict(model_sd, lora_pairs, alpha_default=1.0, verbose=False):
    """
    model_sd: 모델의 state_dict (mutable dict)
    lora_pairs: find_lora_pairs()가 반환한 dict
    이 함수는 model_sd를 직접 수정(병합)하거나, 변경된 키/값의 dict를 반환.
    """
    merged = {}
    for base, comp in lora_pairs.items():
        if "up" not in comp or "down" not in comp:
            if verbose:
                print(f"[WARN] incomplete lora pair for {base}, skipping")
            continue

        up = comp["up"].to(torch.float32)
        down = comp["down"].to(torch.float32)
        alpha = comp.get("alpha", None)
        # rank 추정
        rank = up.shape[0] if up.ndim == 2 else up.shape[1] if up.ndim == 1 else None

        # compute delta = (up @ down) * (alpha / rank)
        # handle shapes:
        try:
            if up.ndim == 2 and down.ndim == 2:
                # typical linear LoRA: up: (r, out), down: (in, r)  OR vice versa
                # many implementations: down: (in, r), up: (r, out)
                # ensure multiplication shapes: (out, r) @ (r, in) -> (out, in) OR (r,out) @ (in,r) invalid
                # We'll assume down: (in, r), up: (r, out) -> up @ down -> (r,out)@(in,r) invalid.
                # Common correct operation: up (out, r), down (r, in) => up @ down -> (out, in)
                # Try both orders to find matching shape with target param
                candidate = None
                try:
                    delta = up @ down  # (out, in) if shapes align
                    candidate = delta
                except Exception:
                    try:
                        delta = (up.T @ down.T).T
                        candidate = delta
                    except Exception:
                        # fallback: try up.matmul(down)
                        delta = up.matmul(down)
                        candidate = delta

                delta = candidate
            else:
                # fallback: attempt matmul after flattening last two dims
                delta = up.reshape(up.shape[0], -1) @ down.reshape(-1, down.shape[-1])
        except Exception as e:
            if verbose:
                print(f"[ERROR] failed to matmul for {base}: {e}")
            continue

        if alpha is None:
            alpha = alpha_default
        # normalize by rank if rank known
        r = up.shape[0] if up.ndim >= 2 else None
        if r is None or r == 0:
            scale = alpha
        else:
            scale = alpha / r

        delta = delta * scale  # float32

        # guess target param key candidates in model_sd
        # common mapping: base + ".weight" or base + ".bias" (but bias usually not)
        candidates = [base + ".weight", base + ".bias", base]
        matched = False
        for cand in candidates:
            if cand in model_sd:
                # ensure shapes compatible (or try transpose)
                tgt = model_sd[cand].to(torch.float32)
                if tgt.shape == delta.shape:
                    merged[cand] = tgt + delta.to(tgt.device)
                    matched = True
                    if verbose:
                        print(f"[MERGE] merged LoRA -> {cand} (shape {tgt.shape})")
                    break
                # try transpose match
                if delta.T.shape == tgt.shape:
                    merged[cand] = tgt + delta.T.to(tgt.device)
                    matched = True
                    if verbose:
                        print(f"[MERGE] merged LoRA (transposed) -> {cand} (shape {tgt.shape})")
                    break
                # try reshape (conv kernels): delta (out,in) -> conv weight (out,in,k,k)
                if tgt.ndim == 4 and delta.ndim == 2:
                    # try expand center
                    k_h = tgt.shape[2]
                    k_w = tgt.shape[3]
                    try:
                        delta4 = delta.view(tgt.shape[0], tgt.shape[1], 1, 1).expand_as(tgt)
                        merged[cand] = tgt + delta4.to(tgt.device)
                        matched = True
                        if verbose:
                            print(f"[MERGE] merged LoRA-> {cand} by expanding to 4D (conv) (shape {tgt.shape})")
                        break
                    except Exception:
                        pass
        if not matched and verbose:
            print(f"[WARN] could not find exact target for {base}. Tried {candidates}")

    # apply merged updates to model_sd
    for k, v in merged.items():
        model_sd[k] = v.to(model_sd[k].dtype)  # keep original dtype

    return model_sd, merged.keys()


def merge_lora_into_pipeline(pipe, lora_path, backup_path=None, inplace=True, verbose=False):
    """
    pipe: StableDiffusion...Pipeline (이미 로드된 상태)
    lora_path: safetensors or torch file containing LoRA weights
    backup_path: (optional) 원본 state_dict 백업 경로 (inplace=True일 때 권장)
    inplace: True면 pipe의 state_dict를 직접 수정해서 적용
    """
    lora_sd = load_lora_state(lora_path)
    lora_pairs = find_lora_pairs(lora_sd)
    if verbose:
        print(f"[INFO] Found {len(lora_pairs)} lora pairs")

    # gather model state_dict (UNet + text_encoder are common targets)
    # We'll merge into both unet and text_encoder state dicts when keys match
    unet = pipe.unet
    text_enc = getattr(pipe, "text_encoder", None)

    # construct combined state dict mapping key->tensor with module name prefixes:
    base_sd = {}
    # unet keys
    for k, v in unet.state_dict().items():
        base_sd["unet." + k] = v.clone().cpu()
    # text encoder keys
    if text_enc is not None:
        for k, v in text_enc.state_dict().items():
            base_sd["text_encoder." + k] = v.clone().cpu()

    # attempt merges: our find_lora_pairs may have keys that already include module prefixes
    # Try to merge into base_sd keys directly, or try removing a top-level 'lora_unet' prefix
    # For simplicity, we'll 1) try matching base (as-is), 2) try "unet." + base, 3) try "text_encoder." + base
    # Prepare a mapping from model keys to actual underlying module/state keys
    model_sd_for_merge = {k: v for k, v in base_sd.items()}

    # apply lora merges
    merged_sd, merged_keys = apply_lora_to_state_dict(model_sd_for_merge, lora_pairs, verbose=verbose)

    # If nothing merged, warn
    if len(list(merged_keys)) == 0:
        warnings.warn("No LoRA weights were merged. Check naming convention in your LoRA file.")
        return

    # Optionally backup original weights
    if inplace and backup_path:
        torch.save(base_sd, backup_path)
        if verbose:
            print(f"[INFO] Saved backup of original weights to {backup_path}")

    # Now write merged values back to pipe modules
    # For keys like "unet.some.key", strip the "unet." prefix and load to module
    # Load unet
    unet_sd = unet.state_dict()
    text_sd = text_enc.state_dict() if text_enc is not None else {}

    for full_k, val in model_sd_for_merge.items():
        if not full_k.startswith("unet.") and not full_k.startswith("text_encoder."):
            continue
        if full_k in merged_sd:
            # this is merged value
            new_val = merged_sd[full_k]
        else:
            new_val = model_sd_for_merge[full_k]

        # write back to module state dict
        if full_k.startswith("unet."):
            k = full_k[len("unet."):]
            if k in unet_sd:
                unet_sd[k] = new_val.to(unet_sd[k].dtype).to(unet.device)
        elif full_k.startswith("text_encoder."):
            k = full_k[len("text_encoder."):]
            if k in text_sd:
                text_sd[k] = new_val.to(text_sd[k].dtype).to(text_enc.device)

    # load back into modules
    unet.load_state_dict(unet_sd, strict=False)
    if text_enc is not None:
        text_enc.load_state_dict(text_sd, strict=False)

    if verbose:
        print(f"[INFO] LoRA merged into pipeline (applied keys: {len(list(merged_keys))})")


def run_img2img_with_lora():
    # load base pipeline
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if DEVICE.startswith("cuda") else torch.float32,
    ).to(DEVICE)

    # merge lora
    merge_lora_into_pipeline(pipe, LORA_PATH, backup_path=BACKUP_FILE if MERGE_INPLACE else None, inplace=MERGE_INPLACE, verbose=True)

    # load input image
    init_image = Image.open(INIT_IMAGE).convert("RGB")
    # optional resize to model res
    init_image = init_image.resize((512, 512))

    generator = torch.Generator(device=DEVICE).manual_seed(42) if DEVICE.startswith("cuda") else None

    # run img2img
    with torch.autocast(DEVICE) if DEVICE.startswith("cuda") else torch.no_grad():
        result = pipe(
            prompt=PROMPT,
            image=init_image,
            strength=STRENGTH,
            guidance_scale=GUIDANCE_SCALE,
            num_inference_steps=NUM_INFERENCE_STEPS,
            generator=generator,
        )

    out = result.images[0]
    out.save(OUT_IMAGE)
    print(f"Saved result -> {OUT_IMAGE}")


if __name__ == "__main__":
    run_img2img_with_lora()
