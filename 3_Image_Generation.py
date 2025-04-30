from diffusers import StableDiffusionPipeline
import torch
import os

# 일부 CPU 환경에서 발생하는 오류 방지를 위한 환경 변수 설정
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Hugging Face에 업로드된 LoRA 파인튜닝 모델의 경로 (UNet attention weight만 저장됨)
model_path='HLife15/kshdrawing'  # 사용자가 훈련한 LoRA 가중치

# Stable Diffusion 기반 모델을 불러옴 (float16으로 메모리 절약)
pipe = StableDiffusionPipeline.from_pretrained('stablediffusionapi/anything-v5', 
                                                torch_dtype=torch.float16, 
                                                use_auth_token=True)

# UNet의 attention 프로세서(attn_procs)에 LoRA 가중치를 적용
pipe.unet.load_attn_procs(model_path)

# 모델을 CUDA(GPU)로 이동
pipe.to("cuda")

# 긍정 프롬프트 (이미지 생성에 반영될 요소)
prompt = '''(drawn by KSH drawing style : 1.5), a anime girl, blonde hair, green eyes, school uniform, smile, sunshine, bright background'''

# 부정 프롬프트 (생성되지 않기를 원하는 요소들)
neg_prompt = '''FastNegativeV2,(bad-artist:1.0),
(worst quality, low quality:1.4), (bad_prompt_version2:0.8),
bad-hands-5,lowres, bad anatomy, bad hands, ((text)), (watermark),
error, missing fingers, extra digit, fewer digits, cropped,
worst quality, low quality, normal quality, ((username)), blurry,
(extra limbs), bad-artist-anime, badhandv4, EasyNegative,
ng_deepnegative_v1_75t, verybadimagenegative_v1.3, BadDream,
(three hands:1.1),(three legs:1.1),(more than two hands:1.4),
(more than two legs,:1.2),badhandv4,EasyNegative,ng_deepnegative_v1_75t,
verybadimagenegative_v1.3,(worst quality, low quality:1.4),text,words,logo,watermark,
'''

# 이미지 생성: LoRA가 적용된 모델로 prompt를 기반으로 1장의 이미지를 생성
image = pipe(prompt, 
             negative_prompt=neg_prompt, 
             num_inference_steps=30, 
             guidance_scale=7.5).images[0]

# 생성된 이미지를 로컬에 저장
image.save("C:/Users/USER/Desktop/character.png")
