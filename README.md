# KSH Drawing AI

### 📄프로젝트 소개
---
LoRA 기법을 활영해 프롬프트를 입력하면 내 그림체(KSH Drawing Style)로 그려주는 그림 인공 지능 프로그램 제작
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
