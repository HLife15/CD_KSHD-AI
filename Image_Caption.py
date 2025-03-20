!pip install transformers
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

for i in range(len(file_list)):
  gc.collect()
  torch.cuda.empty_cache()

  img_name.append(file_list[i])

  img_path = os.path.join('/content/drawing', file_list[i])
  img=Image.open(img_path).convert('RGB')
  img=img.resize((256, 256))

  gc.collect()
  torch.cuda.empty_cache()

  inputs = processor(img, text=prompt, return_tensors="pt").to(device, torch.float16)
  generated_ids = model.generate(**inputs, max_new_tokens=600, do_sample=True, temperature=0.8, top_p=0.9, repetition_penalty=1.2)
  generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
  caption.append(generated_text)
  print(generated_text)

  gc.collect()
  torch.cuda.empty_cache()
