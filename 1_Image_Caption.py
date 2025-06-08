import os
import gc
import torch
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration

prompt = "Describe this image's appearance in a detailed and natural-sounding sentence."
caption = []
img_name = []

file_list = sorted(os.listdir('D:/backup/drawing'))

processor = AutoProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-flan-t5-xl", torch_dtype=torch.float16
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

  img_path = os.path.join('D:/backup/drawing/', file_list[i])
  img=Image.open(img_path).convert('RGB')

  img=img.resize((512, 512))

  gc.collect()
  torch.cuda.empty_cache()

  inputs = processor(img, text=prompt, return_tensors="pt").to(device, torch.float16)
  generated_ids = model.generate(**inputs, max_new_tokens = 80, min_length = 10, do_sample=True, temperature = 0.5, top_p = 0.9, repetition_penalty=2.0, eos_token_id=processor.tokenizer.eos_token_id)
  generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
  caption.append(generated_text)

  #while True:
    #generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    #caption.append(generated_text)

    #if caption[i].find("-") == -1 and caption[i].find(".") == -1:
       #break
    
    #del caption[-1]

  caption[i] = caption[i].replace(',', '')
  print(img_name[i] + ", " + caption[i])
  

  gc.collect()
  torch.cuda.empty_cache()

with open('info.csv', 'w') as f:
  f.write('image,text\n')
  for i in range(len(img_name)):
    f.write(img_name[i] + ',' + caption[i] + ' drawn by KSH drawing style' + '\n')
