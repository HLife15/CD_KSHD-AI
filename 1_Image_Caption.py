import os  # 파일 및 디렉터리 경로 관련 작업을 위한 모듈
import pandas as pd  # 데이터프레임 작업을 위한 pandas 모듈 (여기서는 사용되지 않음)
import gc  # 가비지 컬렉션 모듈 (메모리 정리용)
import torch  # 파이토치 모듈 (딥러닝 연산용)
from PIL import Image  # 이미지 열기 및 처리용 라이브러리
from transformers import AutoProcessor, Blip2ForConditionalGeneration  # BLIP2 모델 로딩과 전처리용 클래스

prompt = "A painting of"  # 이미지 캡션 생성을 위한 기본 프롬프트 문장
caption = []  # 생성된 캡션을 저장할 리스트
img_name = []  # 이미지 파일 이름을 저장할 리스트

file_list = sorted(os.listdir('D:/backup/drawing'))  # 파일 목록을 정렬하여 가져오기

# BLIP2 모델과 프로세서를 불러오기
processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")  # 사전학습된 BLIP2 프로세서 불러오기
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16  # 사전학습된 BLIP2 모델을 float16 타입으로 불러오기
)

device = "cuda" if torch.cuda.is_available() else "cpu"  # GPU(cuda)가 사용 가능하면 GPU 사용, 아니면 CPU 사용
model.to(device)  # 모델을 지정한 디바이스로 이동

# 모델 파라미터 중 float32 타입을 모두 float16으로 변환하여 메모리 절약
for param in model.parameters():
    if param.dtype == torch.float32:  # 데이터 타입이 float32인 경우
        param.data = param.data.to(torch.float16)  # float16으로 변환

# 이미지 리스트를 하나씩 순회하며 캡션 생성
for i in range(len(file_list)):
  gc.collect()  # 불필요한 메모리 수거
  torch.cuda.empty_cache()  # GPU 캐시 비우기

  img_name.append(file_list[i])  # 현재 이미지 파일 이름 저장

  img_path = os.path.join('D:/backup/drawing/', file_list[i])  # 이미지 파일 경로 생성
  img = Image.open(img_path).convert('RGB')  # 이미지 파일 열고 RGB로 변환
  img = img.resize((512, 512))  # 이미지를 512x512 크기로 리사이즈

  gc.collect()  # 메모리 정리
  torch.cuda.empty_cache()  # GPU 캐시 비우기

  inputs = processor(img, text=prompt, return_tensors="pt").to(device, torch.float16)  # 이미지와 텍스트를 모델 입력 형식으로 변환하고 디바이스에 올림
  generated_ids = model.generate(
      **inputs, max_new_tokens=2000, do_sample=True, temperature=0.8, top_p=0.9, repetition_penalty=1.2
  )  # 캡션 생성 (샘플링 방식, 다양한 문장을 생성하기 위해 하이퍼파라미터 설정)

  generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()  # 생성된 문장을 디코딩하고 양끝 공백 제거
  caption.append(generated_text)  # 캡션 리스트에 저장
  caption[i] = caption[i].replace(',', '')  # 캡션 내 쉼표(,) 제거

  print(img_name[i] + ", " + caption[i])  # 생성된 결과를 출력

  gc.collect()  # 메모리 정리
  torch.cuda.empty_cache()  # GPU 캐시 비우기

# 결과를 info.csv 파일로 저장
with open('info.csv', 'w') as f:
  f.write('image,text\n')  # CSV 헤더 작성
  for i in range(len(img_name)):
    f.write(img_name[i] + ',' + caption[i] + ' drawn by KSH drawing style' + '\n')  # 각 이미지 이름과 캡션을 CSV에 저장
