import os
import pandas as pd
import gc
import torch
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration

prompt = "A painting of"
caption = []
img_name = []

file_list = os.listdir('/content/drawing')

processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

for param in model.parameters():
    # Check if parameter dtype is  Float (float32)
    if param.dtype == torch.float32:
        param.data = param.data.to(torch.float16)

for i in range(len(file_list)):
  gc.collect()
  torch.cuda.empty_cache()

  img_name.append(file_list[i])

  img_path = os.path.join('/content/drawing/', file_list[i])
  img=Image.open(img_path).convert('RGB')
  img=img.resize((512, 512))

  gc.collect()
  torch.cuda.empty_cache()

  inputs = processor(img, text=prompt, return_tensors="pt").to(device, torch.float16)
  generated_ids = model.generate(**inputs, max_new_tokens=2000, do_sample=True, temperature=0.8, top_p=0.9, repetition_penalty=1.2)
  generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
  caption.append(generated_text)
  print(str(i) + " "  + img_name[i] + ", " + caption[i])

  gc.collect()
  torch.cuda.empty_cache()

with open('info.csv', 'w') as f:
  f.write('image,text\n')
  for i in range(len(img_name)):
    f.write(img_name[i] + ',' + caption[i] + ' drawn by KSH drawing style' + '\n')
