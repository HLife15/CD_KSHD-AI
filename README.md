# KSH Drawing AI

### 📄프로젝트 소개
---
LoRA 기법을 활영해 프롬프트를 입력하면 내 그림체(KSH Drawing Style)로 그려주는 그림 생성 인공지능 프로그램 제작. 현재는 Text-to-Image 기능만 구현되어 있으나 Image-to-Image 기능을 만드는 게 프로젝트의 최종목표이다.
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

### 🤔LoRA (Low-Rank Adaption)
---
![image](https://www.ibm.com/content/dam/connectedassets-adobe-cms/worldwide-content/creative-assets/s-migr/ul/g/fa/f5/lora-2.component.xl-retina.ts=1744887783463.png/content/adobe-cms/us/en/think/topics/lora/jcr:content/root/table_of_contents/body-article-8/image)
</br></br></br>
 **LoRA (Low-Rank Adaption) 기법**이란 기존의 사전 학습된 딥러닝 모델을 효율적으로 **Fine-Tuning**하는 방법이다. 새로운 데이터를 학습시키고 싶을 때, 전통적인 Fine-Tuning 기법은 사전 학습된 기존 모델을 처음부터 다시 학습하기 때문에 상당히 비효율적이다. 하지만 LoRA 기법은 기존 모델을 학습하지 않고 새로운 데이터를 기존 모델에 추가하는 식으로 학습을 진행하기 때문에 효율적이다. </br>
 Stable Diffusion 등 기존의 이미지 생성 모델은 잘 훈련되어 있기 때문에 새로운 데이터들만 효율적으로 학습시키는 LoRA 기법을 채택하였고 이 프로젝트에서 사용한 Stable Diffusion 기반 모델은 [**Anything V5**](https://huggingface.co/stablediffusionapi/anything-v5)이다.
</br></br></br></br>


### 🎢진행 과정
---
### 1. **Image Captioning**
</br>

 [**BLIP-2**](https://huggingface.co/docs/transformers/en/model_doc/blip-2)를 이용하여 데이터셋에 있는 이미지들의 캡션을 생성한다. 이 캡션들은 다음 과정인 **Fine-Tuning**에 쓰일 예정이다. 각 캡션은 생성되자마자 KSH Drawing Style이 학습에 효과적으로 반영되게 하기 위해 공통적으로 뒤에 '**drawn by KSH drawing style**'이라는 태그가 추가되어 info.csv에 작성된다. 캡션 생성이 완료되어 info.csv이 만들어지면 허깅페이스 데이터셋에 업로드한다. 그렇게 만들어진 허깅페이스 데이터셋은 아래 같다. 

</br>

![Image](https://github.com/user-attachments/assets/b2fa8d6d-56e7-4d5d-afc6-80a04c4fc2d4)

</br>

[데이터셋](https://drive.google.com/drive/folders/1iIQFhOkSXIoqaZ7ZG38tjJN79-bSCEn0?usp=sharing)
 
[허깅페이스](https://huggingface.co/datasets/HLife15/drawing)

</br></br>

### 2. **Fine-Tuning**
</br>

Fine-Tuning 코드는 아래와 같다. 사전 학습된 모델로는 Stable Diffusion 기반의 [**Anything-V5**](https://huggingface.co/stablediffusionapi/anything-v5)를 사용했다.

</br>

```
accelerate launch train_text_to_image_lora.py 
--pretrained_model_name_or_path="stablediffusionapi/anything-v5" 
--dataset_name="HLife15/drawing" --caption_column='text' 
--resolution=512 --random_flip 
--train_batch_size=1 
--num_train_epochs=50 --checkpointing_steps=10000 
--learning_rate=1e-04 --lr_scheduler="cosine" --lr_warmup_steps=0 
--seed=5508 
--output_dir="D:\backup\finetuned\" 
--validation_prompt="drawn by KSH drawing style, an anime girl, blonde hair, green eyes,school uniform, smile, sunshine, outside, park, river, blue sky" 
--report_to="wandb" 
```

</br>

이 중에서 학습을 여러번 진행하면서 조정한 항목들은 다음과 같다.

</br>

**--train_batch_size=1** : 한 번의 step에서 모델이 처리하는 데이터 샘플의 수 (1 or 2). </br>
**--num_train_epochs=50** : 학습 반복 횟수. 이 코드에서는 전체 데이터셋을 50번 반복 학습한다. </br>
**--checkpointing_steps=10000** : 일정 step마다 체크포인트를 저장한다. 이 코드에서는 10000step마다 체크포인트를 저장한다. </br>
**--validation_prompt=** : 학습 시 생성할 이미지의 프롬프트. </br>

학습은 총 {(데이터셋 이미지 개수) * (num_train_epochs)} / (train_batch_size) step으로 진행되며 5700장의 데이터셋을 기준으로 짧으면 5시간, 길면 60시간 정도 걸렸다.

</br></br>

### 3. **Image-Creating**
</br>

이미지 생성은 GUI까지 구현한 **make_gui.py**에서 진행하였다. Positive_prompt와 Negative_prompt를 입력하고 generate 버튼을 누르면 이미지가 생성되는 방식이다. 

</br>

![Image](https://github.com/user-attachments/assets/2d61cdf6-ac45-4841-a0b4-afc719fd7481)


</br></br></br></br>


### 📰결과
---
**(2025.06.12. // Dataset = 5700, batch size = 1, epochs = 50, checkpoint-10000)**

</br>

<p align="center">
  <img src="https://github.com/user-attachments/assets/e7b0a292-afb7-4367-be0a-8122f98ddf98" width="45%"/>
  <img src="https://github.com/user-attachments/assets/40e771f8-3882-4a2e-9681-f4d44aa51b8d" width="45%"/>
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/5c59edf2-5f5d-48c3-9339-aa9f8e213120" width="45%"/>
  <img src="https://github.com/user-attachments/assets/88b2e495-35a4-4c96-bfb3-c6bbc45c7dc4" width="45%"/>
</p>
</br>

</br></br></br></br>

### 📌참고 문헌
---
https://pej2834.tistory.com/40
</br>
https://huggingface.co/tasks/text-to-image
</br>
https://huggingface.co/docs/transformers/en/model_doc/blip-2
