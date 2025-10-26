# KSH Drawing AI

### 📄프로젝트 소개
---
&ensp;프롬프트를 입력하면 **내 그림체 (KSH Drawing Style)** 의 이미지를 생성하는 LoRA 기법을 활용한 Text-to-Image 그림 생성 인공지능 프로그램 제작.
</br></br></br></br>

### ✒️KSH Drawing Style
---
<p align="center">
  <img src="https://github.com/HLife15/CD_KSHD-AI/blob/main/Drawing/%EB%A3%A8%EB%B9%84.png" width="45%"/>
  <img src="https://github.com/HLife15/CD_KSHD-AI/blob/main/Drawing/%EC%B9%B4%EB%82%98.png" width="45%"/>
</p>
<p align="center">
  <img src="https://github.com/HLife15/CD_KSHD-AI/blob/main/Drawing/%ED%86%A0%EB%8F%84%EB%A1%9C%ED%82%A4.png" width="45%"/>
  <img src="https://github.com/HLife15/CD_KSHD-AI/blob/main/Drawing/%EC%9D%B4%EC%83%81.png" width="45%"/>
</p>

* 애니메이션 느낌
* 간결한 선과 명암 표현
* 여자 캐릭터는 눈이 크고 아이라인을 두껍게 표현
* 남자 캐릭터는 여자 캐릭터에 비해 눈이 작고 아이라인은 얇게 표현
</br></br></br></br>

### 🤔방법론
---
&ensp;프로젝트 진행에 있어 가장 핵심이 된 기술은 바로 **Stable Diffusion**이다. Stable Diffusion은 '**텍스트로부터 이미지를 생성하는 딥러닝 기반의 확산 모델 (Diffusion Model)**'로, Stability AI와 CompVis, Runway 등 여러 연구 기관이 협력하여 개발한 오픈소스 인공지능 모델이다. 이 모델은 노이즈가 섞인 이미지를 점진적으로 정제하면서 주어진 텍스트 프롬프트(Prompt)의 의미를 시각적으로 표현한다. 이러한 과정을 **확산 (denoising diffusion)** 이라 하며, 이미지의 생성 과정에서 **잠재 공간 (latent space)** 을 활용함으로써 연산 효율성과 메모리 사용량을 크게 줄였다.
 </br></br></br>
![Image](https://github.com/user-attachments/assets/e8d22247-ddb3-4dae-9f9f-f8e6fe24b923)
</br></br></br>
&ensp;Stable Diffusion은 이미지 생성 과정에서 **텍스트 인코더 (Text Encoder)** 와 U-Net 기반의 디노이징 네트워크, 그리고 **VAE (Variational  Autoencoder)** 를 사용한다. 텍스트 인코더는 입력된 문장의 의미를 벡터 형태로 변환하고, U-Net은 노이즈 제거 과정을 통해 점차적으로 이미지를 복원한다. 마지막으로 VAE는 잠재 공간에서 생성된 결과를 실제 이미지로 디코딩한다.
</br></br>
&ensp;이와 같은 구조 덕분에 Stable Diffusion은 사용자의 텍스트 입력을 반영하여 고해상도·고품질의 이미지를 빠르게 생성할 수 있으며, 오픈소스로 배포되어 추후 다양한 변형 모델(예: DreamBooth, LoRA, ControlNet 등)이 개발되는 계기가 되었다. 이 프로젝트에선 그 중에서 **LoRA**를 선택해서 작업을 진행했다.
</br></br></br>
![image](https://www.ibm.com/content/dam/connectedassets-adobe-cms/worldwide-content/creative-assets/s-migr/ul/g/fa/f5/lora-2.component.xl-retina.ts=1744887783463.png/content/adobe-cms/us/en/think/topics/lora/jcr:content/root/table_of_contents/body-article-8/image)
</br></br></br>
&ensp;**LoRA (Low-Rank Adaption) 기법**이란 기존의 사전 학습된 딥러닝 모델을 효율적으로 **Fine-Tuning**하는 방법이다. 새로운 데이터를 학습시키고 싶을 때, 전통적인 Fine-Tuning 기법은 사전 학습된 기존 모델을 처음부터 다시 학습하기 때문에 상당히 비효율적이다. 하지만 LoRA 기법은 기존 모델을 학습하지 않고 새로운 데이터를 기존 모델에 추가하는 식으로 학습을 진행하기 때문에 효율적이다. Stable Diffusion을 기반으로 한 기존의 이미지 생성 모델은 잘 훈련되어 있기 때문에 새로운 데이터들만 효율적으로 학습시키는 LoRA 기법을 채택하였고 이 프로젝트에서 사용한 Stable Diffusion 기반 모델은 [**Anything V5**](https://huggingface.co/stablediffusionapi/anything-v5)이다. </br></br>
&ensp;**Anything V5**는 Stable Diffusion을 기반으로 한 커스텀 파인튜닝 모델로, 특히 **애니메이션 풍 (Anime Style)** 이미지 생성에 최적화된 버전이다. 본 모델은 기본 Stable Diffusion 모델(SD 1.5 또는 SD 2.x)에 비해 다음과 같은 주요 차이점을 가진다.</br></br>
&ensp;첫 번째로 Anything V5는 대규모의 **애니메이션, 일러스트, 만화풍 이미지**를 중심으로 학습되어 인물의 윤곽선, 색감, 눈의 디테일, 채색 질감 등에서 일반 Stable Diffusion보다 훨씬 높은 표현력을 가진다. 두 번째로 **텍스트 프롬프트에 대한 응답성이 강화**되어, 세부 묘사 (Soft Lighting, Detailed Eyes, Dynamic Pose 등)를 보다 정밀하게 반영하며, 스타일 태그 (1 girl, Masterpiece, Best Quality 등)에 최적화되어 있어, 프롬프트 설계 시 높은 제어성을 제공한다. 마지막으로 Anything V5는 기존 Stable Diffusion의 모델 가중치 위에 추가적인 파인튜닝이 이루어져, 다양한 샘플러(Euler, DPM++, DDIM 등)에서도 안정적인 품질을 유지한다. 또한 VAE 및 LoRA 확장과의 호환성이 좋아, 세부 그림체나 표정 표현을 쉽게 커스터마이징할 수 있다는 장점이 있다.</br></br>
&ensp;결과적으로 Anything V5는 **Stable Diffusion의 구조적 강점은 유지하면서도, 특정 시각적 스타일(애니풍)을 극대화한 모델**로 정의할 수 있다. 따라서 애니메이션 풍의 내 그림체 학습을 진행하기에에 가장 적절한 모델이라 판단했다. 
 
</br></br></br></br>


### 🎢구현 방법
---
### 1. **Image Captioning**
</br>

&ensp;우선 데이터셋의 각 이미지들에 대해 설명하는 캡션을 생성해야 하기에 이 과정에선 [**BLIP-2**](https://huggingface.co/docs/transformers/en/model_doc/blip-2)를 이용했다.</br></br></br>

![Image](https://github.com/user-attachments/assets/bc1793e6-e5aa-4ddf-a168-f88001ca9904) </br></br></br>
&ensp;'**BLIP-2 (Bootstrapped Language-Image Pretraining 2)**'는 이미지와 텍스트를 함께 이해하고 연결하는 AI 모델이다. 이미지를 제공하면 Vision Encoder에서 그 이미지를 숫자 정보로 바꾸고, Q-Former에서 그 정보를 언어모델이 이해할 수 있는 형태로 변환한다. 마지막으로 Language Model (LLM)이 정보에 대한 대답을 자연어로 생성하는 구조로 설계되어 있다. 이를 이용해 약 4500장의 데이터셋의 이미지들에 대한 캡션을 빠르게 생성할 수 있다.</br></br>
&ensp;이렇게 생성된 캡션들은 다음 과정인 **Fine-Tuning**에 쓰일 예정이다. 각 캡션은 생성되자마자 KSH Drawing Style이 학습에 효과적으로 반영되게 하기 위해 공통적으로 뒤에 '**drawn by KSH drawing style**'이라는 태그가 추가되어 info.csv에 작성된다. 캡션 생성이 완료되어 info.csv이 만들어지면 허깅페이스 데이터셋에 업로드한다. 그렇게 만들어진 허깅페이스 데이터셋은 아래와 같다. 

</br>

![Image](https://github.com/user-attachments/assets/b2fa8d6d-56e7-4d5d-afc6-80a04c4fc2d4)

</br>

[데이터셋](https://drive.google.com/drive/folders/1iIQFhOkSXIoqaZ7ZG38tjJN79-bSCEn0?usp=sharing)
 
[허깅페이스](https://huggingface.co/datasets/HLife15/drawing)

</br></br>

### 2. **Fine-Tuning**
</br>

&ensp;Fine-Tuning 코드는 아래와 같다. 사전 학습된 모델(pretrained_model_name_or_path)로는 Stable Diffusion 기반의 [**Anything-V5**](https://huggingface.co/stablediffusionapi/anything-v5)를 사용했다.

</br>

```
accelerate launch train_text_to_image_lora.py 
--pretrained_model_name_or_path="stablediffusionapi/anything-v5" 
--dataset_name="HLife15/drawing" --caption_column='text' 
--resolution=512 --random_flip 
--train_batch_size=1 
--num_train_epochs=20 --checkpointing_steps=4000 
--learning_rate=1e-04 --lr_scheduler="cosine" --lr_warmup_steps=0 
--seed=5508 
--output_dir="D:\backup\finetuned\" 
--validation_prompt="drawn by KSH drawing style, an anime girl, blonde hair, green eyes, school uniform, smile, sunshine, outside, park, river, blue sky" 
--report_to="wandb" 
```

</br>

&ensp;이 중에서 학습을 여러번 진행하면서 조정한 항목들은 다음과 같다.

</br>

**--train_batch_size=1** : 한 번의 step에서 모델이 처리하는 데이터 샘플의 수 (1 or 2). </br>
**--num_train_epochs=20** : 학습 반복 횟수. 이 코드에서는 전체 데이터셋을 20번 반복 학습한다. </br>
**--checkpointing_steps=4000** : 일정 step마다 체크포인트를 저장한다. 이 코드에서는 4000step마다 체크포인트를 저장한다. </br>
**--validation_prompt=** : 학습 시 생성할 이미지의 프롬프트. </br>

&ensp;학습은 총 {(데이터셋 이미지 개수) * (num_train_epochs)} / (train_batch_size) step으로 진행되며 epochs 수에 따라 짧으면 5시간 (epochs = 3), 길면 60시간 (epochs = 100) 정도 걸렸다.

</br></br>

### 3. **Image-Creating**
</br>

&ensp;이미지 생성은 Custom Tkinter를 이용하여 구현한 GUI까지 포함된 **make_gui.py**에서 진행하였다. Positive Prompt와 Negative Prompt를 입력하고 Generate 버튼을 누르면 이미지가 생성되는 방식이다. 

</br>

![Image](https://github.com/user-attachments/assets/3f36d951-dc14-47eb-853e-31a9c1ba38eb)


</br></br></br></br>


### 📰결과
---
**(2025.10.24. // Dataset = 4500, batch size = 1, epochs = 20)**

</br>

Negative Prompt (공용) : wrong number of person, wrong gender, wrong hair style, wrong hair color, wrong eyes color, wrong clothe color, wrong accessory color, wrong pose, wrong background, harsh line, face closeup, no pupil, no nose, strange hand

</br></br>

<p align="center">
  <img src="https://github.com/user-attachments/assets/96bb7c5c-7bdf-4224-8961-27e8be5971f8" width="45%"/>
  <img src="https://github.com/user-attachments/assets/23e07e2b-a21f-4c8c-af9f-00f042ba692b" width="45%"/>
</p>
</br>
Positive Prompt : (drawn by KSH drawing style : 1.5), an anime girl, blonde hair, green eyes, school uniform, smile, sunshine, outside, park, river, blue sky
</br></br></br>
<p align="center">
  <img src="https://github.com/user-attachments/assets/179a8bc5-7ad0-40dd-be9c-77cda5e9b2d9" width="45%"/>
  <img src="https://github.com/user-attachments/assets/ab8031ac-c490-45a8-a5b8-5a095541cab1" width="45%"/>
</p>
</br>
Positive Prompt : (drawn by KSH drawing style : 1.5), an anime guy, blonde hair, green eyes, school uniform, smile, sunshine, outside, park, river, blue sky

</br></br></br></br>

### ✅결론
---
&ensp;이 프로젝트를 통해 LoRA 기법이 비교적 적은 자원으로도 내 그림체를 어느 정도 반영한 맞춤형 생성 모델을 구축할 수 있음을 확인할 수 있었다. 약 4500장의 데이터셋은 이미지 학습에는 부족한 양이 아닐까 생각했지만, 기존에 존재하는 Stable Diffusion 기반 모델에 LoRA 기법으로 학습을 시키자 내 그림체의 화풍과 색감이 예상보다 잘 재현되었다. 특히 내 그림체 특유의 사소한 묘사 방법 (얼굴 홍조 표현, 안광 표현 등) 역시 잘 학습된 것이 인상 깊었다.</br>
&ensp;이 프로젝트는 창작자 개인의 개성을 보존하는 AI 도구로서의 기술적인 의미를 넘어 예술적 가치도 있음을 시사한다. 향후에는 이를 이용한 만화 제작이나, Text-to-Image를 넘어 Image-to-Image 변환 등 다양한 방법으로의 확장 가능성을 기대한다.

</br></br></br></br>

### 📌참고 문헌
---
[LoRA로 StableDiffusion 그림체 학습시키기](https://pej2834.tistory.com/40)
</br>
[What is Text-to_Image? - Hugging Face](https://huggingface.co/tasks/text-to-image)
</br>
[BLIP-2 - Hugging Face](https://huggingface.co/docs/transformers/en/model_doc/blip-2)
