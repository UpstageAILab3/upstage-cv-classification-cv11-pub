[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/FVjNDCrt)
# Title (Please modify the title)
## Team

| ![박패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![이패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![최패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![오패캠](https://avatars.githubusercontent.com/u/156163982?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [박패캠](https://github.com/UpstageAILab)             |            [이패캠](https://github.com/UpstageAILab)             |            [최패캠](https://github.com/UpstageAILab)             |            [김패캠](https://github.com/UpstageAILab)             |            [오패캠](https://github.com/UpstageAILab)             |
|                            팀장, 담당 역할                             |                            담당 역할                             |                            담당 역할                             |                            담당 역할                             |                            담당 역할                             |

## 0. Overview
### Environment
- OS: Ubuntu 20.04
- GPU: NVIDIA RTX 3090
- Python: 3.8
- IDE: Jupyter Notebook, VSCode

#### 주요 라이브러리:
torch==1.9.0
torchvision==0.10.0
albumentations==1.0.0
pandas==1.3.3
numpy==1.21.2
timm==0.4.12
matplotlib==3.4.3
sklearn==0.24.2

### Requirements
데이터셋 처리 및 모델 학습을 위한 충분한 GPU 자원: 대용량 이미지 데이터를 빠르게 처리하고 모델을 효율적으로 학습시키기 위해 고성능 GPU가 필수적입니다.
다양한 데이터 증강 기법 적용: 데이터 증강을 통해 모델의 일반화 성능을 높이기 위해 Albumentations 라이브러리를 활용하였습니다.
모델 학습 상태 추적: 학습 과정 모니터링 및 실험 관리 도구인 wandb 설치 및 설정.
필수 라이브러리 설치: PyTorch, torchvision, albumentations, pandas, numpy, timm, matplotlib, sklearn 등을 설치하여 모델 학습 및 평가에 필요한 환경을 구성.

## 1. Competiton Info

### Overview

본 대회는 패스트캠퍼스 Upstage AI Lab 3기가 주최하는 문서 이미지 분류 대회입니다. 
다양한 종류의 문서 이미지가 주어졌을 때, 각 이미지의 문서 타입을 정확히 분류하는 것이 목표입니다. 
대회의 평가 기준은 Macro F1 Score로, 각 클래스별 F1 스코어를 계산한 후 평균을 구하여 최종 점수를 산출합니다.

총 17종의 문서 타입을 분류:
계좌번호, 자동차 번호판, 자동차 계기판, 진료비영수증, 여권, 운전면허증, 주민등록증, 자동차 등록증, 약제비 영수증, 처방전, 통원/진료 확인서, 입퇴원 확인서, 진단서, 진료비 납입 확인서, 이력서, 소견서, 건강보험 임신출산 진료비 신청서

### Timeline

2024년 7월 30일 - 시작일
2024년 8월 11일 - 최종 제출 마감일

## 2. Components

### Directory


```
├── code
│   ├── jupyter_notebooks
│   │   └── model_train.ipynb
│   │   └── TTA_SoftVoting.ipynb
│   
├── docs
│   └── pdf
│       └──  [패스트캠퍼스] Upstage AI Lab 3기_그룹 스터디 11조.pptx
│   
└── datasets
│   └── train.csv
│   └── sample_submission.csv
│   └── meta.csv
│   └── test
│       ├── images
│   └── train
│       └── images
       

```

## 3. Data descrption

### Dataset overview

#### 학습 데이터
- 총 1570장의 문서 이미지
- 총 17개의 클래스
- 각 클래스 별로 46 ~ 100장의 이미지

#### 학습 데이터 구조
- train.csv: 학습 이미지의 이름과 클래스 사이의 mapping 정보
  - ID, target
- meta.csv: 클래스 이름과 인덱스의 mapping 정보
  - target, class_name
- train/: 학습 이미지가 들어있는 directory


#### 테스트 데이터
- 총 3140장의 문서 이미지
- 총 17개의 클래스
- 대회의 난이도 조절을 위해 여러 augmentations 적용돼있음
- 실제 현실 세계의 데이터는 노이즈가 굉장히 많음 (구겨진 문서, 물에 젖은 문서, 빛번짐, …)
- 어떤 augmentations는 secret
- EDA를 통해서 파악해볼 수 있음

#### 테스트 데이터 구조

- sample_submission.csv: 예측값을 채워넣을 더미 파일
- test/: 테스트 이미지가 들어있는 directory

### EDA

- 이미지 크기 분석
이미지의 크기 분포를 파악하기 위해 히스토그램을 생성하였습니다. 이는 모델 입력 크기를 결정하는 데 중요한 정보입니다.

- 클래스 분포 분석
각 클래스의 데이터 분포를 시각화하여 클래스 불균형 여부를 확인하였습니다. 이는 데이터 증강 및 샘플링 전략을 수립하는 데 중요한 참고 자료가 됩니다.

- 특정 클래스 이미지 시각화
특정 클래스의 이미지를 추출하여 시각화함으로써 데이터의 특성을 파악하였습니다. 이를 통해 데이터 전처리 및 모델링에 필요한 추가적인 인사이트를 얻을 수 있습니다.

### Data Processing

#### 데이터 증강
- 다양한 데이터 증강 기법을 적용하여 모델의 일반화 성능을 향상시켰습니다. Albumentations 라이브러리를 사용하여 회전, 노이즈 추가, 밝기 조절 등의 기법을 적용하였습니다.

#### 데이터셋 정의
- 학습 데이터와 테스트 데이터를 로드하고, 필요한 변환을 적용하였습니다. 이를 통해 모델 학습에 최적화된 데이터셋을 구성하였습니다.

#### 데이터 로더 설정
- PyTorch의 DataLoader를 사용하여 학습 및 평가를 위한 데이터 로더를 설정하였습니다. 이를 통해 배치 단위로 데이터를 효율적으로 처리할 수 있었습니다.

#### 데이터 시각화
- Train Data와 Test Data를 시각화하여 데이터의 다양한 측면을 파악하고, 이를 통해 모델링 과정에서 발생할 수 있는 문제를 사전에 인지하고 해결하였습니다.

## 4. Modeling

### Model descrition

문서 이미지 분류를 위해 다양한 사전 학습된 모델을 활용하였습니다. 사용된 모델은 다음과 같습니다:

- EfficientNet B3
- EfficientNet B5
- EfficientNetV2 RW M
- Tiny ViT 21M
이들 모델을 선택한 이유는 사전 학습된 모델들이 이미지 분류에서 높은 성능을 보이며, 다양한 데이터 증강 기법과 결합하여 강력한 성능을 발휘하기 때문입니다.

### Modeling Process

- 데이터 증강: 다양한 증강 기법을 통해 데이터셋을 확장하고, 모델의 일반화 성능을 향상시켰습니다.
- 모델 정의: TimM 라이브러리를 활용하여 사전 학습된 모델을 불러오고, 출력 클래스 수에 맞게 마지막 레이어를 조정하였습니다.
- 손실 함수 및 옵티마이저 설정: 교차 엔트로피 손실 함수와 AdamW 옵티마이저를 사용하여 모델을 학습시켰습니다.
- 학습 루프 수행: 에포크 단위로 모델을 학습시키고, 학습 상태를 wandb를 통해 모니터링하였습니다.


## 5. Result

### Inference

#### OCR 적용
-  문서의 구조만으로는 판단하기 어려운 target 3,4,7,14에 대해 rotate, crop, contrast 등 전처리 후 EasyOCR을 적용하여 예측 성능을 높였습니다.
-  
#### TTA 적용
- Test Time Augmentation을 통해 예측 성능을 높였습니다. 다양한 증강 기법을 테스트 이미지에 적용하여 모델의 예측을 평균화하였습니다.

#### 소프트 보팅
- 여러 모델의 예측 결과를 소프트 보팅 기법으로 결합하여 최종 예측을 도출하였습니다. 이를 통해 개별 모델의 약점을 보완하고 전반적인 성능을 향상시켰습니다.


### Leader Board

- 리더보드 캡처:
- 랭크와 점수:
- Rank: 1
- Score: 0.9704

### Presentation

- _Insert your presentaion file(pdf) link_

## etc

### Meeting Log

- _Insert your meeting log link like Notion or Google Docs_


