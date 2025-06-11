from diffusers import StableDiffusionPipeline
import torch
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

model_path='D:/backup/finetuned' #hugging face

pipe = StableDiffusionPipeline.from_pretrained('stablediffusionapi/anything-v5', torch_dtype=torch.float16,safety_checker = None,use_auth_token=True)
pipe.unet.load_attn_procs(model_path)
pipe.to("cuda")

#positive prompt
prompt = '''(drawn by KSH drawing style : 1.5), a anime girl, blonde hair, green eyes, school uniform, smile, sunshine, bright background
'''

#negative prompt
neg_prompt='''cement texture background, FastNegativeV2,(bad-artist:1.0),
(worst quality, low quality:1.4), (bad_prompt_version2:0.8),
bad-hands-5,lowres, bad anatomy, bad hands, ((text)), (watermark),
error, missing fingers, extra digit, fewer digits, cropped,
worst quality, low quality, normal quality, ((username)), blurry,
 (extra limbs), bad-artist-anime, badhandv4, EasyNegative,
 ng_deepnegative_v1_75t, verybadimagenegative_v1.3, BadDream,
(three hands:1.1),(three legs:1.1),(more than two hands:1.4),
(more than two legs,:1.2),badhandv4,EasyNegative,ng_deepnegative_v1_75t,verybadimagenegative_v1.3,(worst quality, low quality:1.4),text,words,logo,watermark,
'''

image = pipe(prompt, negative_prompt=neg_prompt,num_inference_steps=30, guidance_scale=7.5).images[0]
image.save("C:/Users/USER/Desktop/character.png")
