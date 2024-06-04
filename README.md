# Food Image Classification


# Title
## CNN을 이용한 음식 이미지 분류 및 실사용 예제

# 목차 Contents

- [Members](#Members)
- I. [Proposal](#i-proposal)
- II. [Datasets](#ii-datasets)
- III. [Methodology](#iii-methodology)
- IV. [Evaluation & Analysis](#iv-evaluation--analysis)
- V. [Related Works](#v-related-works)
- VI. [Conclusion-Discussion](#vi-conclusion-discussion)

# Members
### 김도현:  데이터사이언스학부 / dhkim011030@gmail.com
### 김준환:  융합전자공학부 / junsemin@naver.com
### 심준용:  수학과 / wnsdyd029451@gmail.com
### 안성우:  융합전자공학부 / tjddn00124@gmail.com

# I. Proposal

### Motivation: Why are you doing this?
우리는 인공지능 기술의 발전을 통해 다양한 실생활 문제를 해결하려고 하고 있다. 요즘 SNS의 발달과 코로나 시기로 인하여 운동과 건강관리에 대한 욕구가 점점 증가하고 있어 자연스럽게 식단관리에 대한 관심이 증가하고 있다. 따라서 우리는 이러한 사람들의 이슈를 돕는 딥러닝 시스템이 있으면 좋을 것 같다고 생각을 하여 음식 이미지 분류에 대한 프로젝트를 진행하게 되었다.

 또한 이는 단순한 식단관리를 넘어 레스토랑, 건강관리, 요리 레시피 추천 등 여러 분야에서 도움을 줄 수 있고, 인공지능이 얼마나 정확하게 음식 이미지를 분류할 수 있는지를 확인함으로써 인공지능의 역할과 기능에 대해서 깊은 학습을 할 수 있는 프로젝트라고 생각하였다.


### What do you want to see at the end?
우선, 음식 이미지 분류를 다양한 deep learning 모델(ResNet50[4](#<4>), AlexNet, VGG)을 활용하여 성능비교(Accuracy, Error Rate 등)를 해보고, 최적의 모델을 찾아보려고 한다. 그 뒤, 해당 모델을 이용하여 본격적으로 이미지 분류 학습을 하고 CAM(Class Activation Map)분석을 통하여 가중치를 시각화 하고 어떤 가중치들이 분류에 있어서 주요했는지를 분석하고자 한다. 이를 통해서 음식 이미지 분류에 있어서 key가 되는 부분을 확인하여 궁극적으로 성능 개선을 통해 실생활에서도 오류없이 적용가능한 모델로 발전시킬 수 있는 지를 분석해본다.

# II. Datasets
### Describing your dataset
https://www.kaggle.com/datasets/harishkumardatalab/food-image-classification-dataset

kaggle의 Food Image Classification Dataset을 이용하였다. 24k개의 이미지들로 이루어져 있으며, 34개 종류의 서양, 인도 음식으로 이루어져 있다.

### (class) 
> **Baked Potato**, **Crispy Chicken**, **Donut**, **Fries**, **Hot     Dog**, **Sandwich**, **Taco**, **Taquito**, **apple_pie**, **buger**, **butter_naan**, **chai**, **chapati**, **cheesecake**, **chicken_curry**, **chole-bhature**, **dal_makhani**, **dhokla**, **fried_rice**, **ice_cream**, **idli**, **jalebi**, **kaathi_rolls**, **kadai_paneer**, **kulfi**, **masala_dosa**, **momos**, **omelette**, **paani_puri**, **pakode**, **pav_bhaji**, **pizza**, **samosa**, **shushi**


<center><img width="100%" alt="image" src="./images/Dataset/imgs.png"></center>  <br>

데이터셋을 훈련, 검증, 테스트 세트로 나누는 것은 하이퍼 파라미터를 조정하여 모델의 성능을 효과적으로 평가하고 과적합을 방지하기 위한 중요한 단계이다.


훈련 세트(Train Set):70%

검증 세트(Validation Set): 15%

테스트 세트(Test Set): 15%

<br>
데이터셋의 클래스 분포는 다음과 같다.
<center><img width="100%" alt="image" src="./images/Dataset/class_distribution.png"></center>



# III. Methodology
### Explaining your choice of algorithms (methods)
- 데이터 증강(Data Augmentation)

<center><img width="100%" alt="image" src="./images/Methodology/Data_Augmentation.png"></center>

> 데이터 증강은 모델의 일반화 성능을 향상시키기 위해 기존의 데이터 셋을 회전, 이동, 스케일링, 플리핑 등 인위적으로 증가시키는 기법이다. 이를 통해 과적합(Overfitting)을 방지하고, 데이터의 다양성도 증가시킬 수 있다.
- VGG, AlexNet, ResNet Model Comparison

## 비교 모델

**AlexNet**

<center><img width="100%" alt="image" src="./images/Methodology/AlexNet_image.png"></center>

> AlexNet은  심층 신경망 모델로, 딥러닝 분야에서 중요한 전환점을 마련했다. AlexNet은 이미지 분류 작업에서 탁월한 성능을 보여주며, 그 이후의 딥러닝 연구에 큰 영향을 미쳤습니다 신경망 구조, ReLU활성화 함수, 드롭아웃, 데이터 증강, 그리고 GPU 병렬처리를 통해 성능을 극대화했다.

**VGG**

<center><img width="100%" alt="image" src="./images/Methodology/VGG_image.png"></center>

> VGG(Visual Geometry Group)는 대규모 이미지 인식을 위한 매우 깊은 컨볼루션 네트워크이다. 16 혹은 19개의 레이어를 이용하고, 큰 커널 크기 필터를 여러 3X3 커널 크기 필터로 차례로 교체하여 AlexNet에 비해 상당한 개선을 이루었다.

**ResNet**

<center><img width="100%" alt="image" src="./images/Methodology/ResNet_image.png"></center>

> ResNet(Residual Networks)은 딥러닝 모델의 깊이를 증가시키면서도 학습이 가능하도록 설계된 모델로, '잔차학습'(Residual Learning)이라는 개념을 도입해 매우 깊은 네트워크에서도 효과적으로 학습할 수 있도록 하였다. ResNet은 성능이 매우 뛰어나며, 기울기 소실(Gradient Vanishing)문제를 효과적으로 해결하여 복잡한 문제를 해결할 수 있는 능력을 제공하고 이를 통해 이미지 분류, 객체 탐지 등 다양한 컴퓨터 비전 과제에서 높은 성능을 발휘한다.
우리는 이 ResNet50을 채택하였다.

## 시각화 기법

- CNN Filter Visualization

<center><img width="100%" alt="image" src="./images/Methodology/CNN_Filter_image.png"></center>

> CNN 필터 시각화(CNN Filter Visulaization)는 컨볼루션 신경망(CNN)의 내부 작동 방식을 이해하고, 모델이 입력 이미지에서 어떤 특징을 학습하는지 분석하는 데 사용된다. CNN의 필터는 이미지의 특정 패턴이나 특징을 감지하는 역할을 하며, 필터 시각화는 이러한 과정이 어떻게 이루어지는지 시각적으로 보여준다

- T-SNE Feature Embedding Visulaization

<center><img width="100%" alt="image" src="./images/Methodology/T-SNE_image.png"></center>

> t-SNE(t-Distributed Stochastic Neighbor Embedding)은 고차원 데이터를 저차원(주로 2차원 또는 3차원) 공간에 시각화하여 데이터의 패턴과 구조를 쉽게 이해할 수 있도록 해주는 차원 축소 기법이다. 데이터 사이의 유사성을 보존하면서 고차원 공간에서의 군집 구조를 저차원 공간에서도 잘 드러내는 특징이 있다.

- CAM (Class Activation Map) Visualization

<center><img width="100%" alt="image" src="./images/Methodology/CAM_image.png"></center>

> CAM(Class Activation Mapping) 시각화는 딥러닝 모델이 이미지의 어떤 부분을 사용하여 특정 클래스를 예측하는지 시각적으로 이해할 수 있게 해주는 기술이다. CAM은 주로 이미지 분류와 같은 컴퓨터 비전 작업에서 모델의 예측을 해석하고 디버깅하는 데 사용된다. 

# IV. Evaluation & Analysis
### Graphs, tables, any statistics (if any)

<p align="center">
    <img width="45%" alt="image" src="./images/Evaluation&Analysis/accuracy_graph.png">
    <img width="45%" alt="image" src="./images/Evaluation&Analysis/loss_graph.png">
</p>

**우리는 다음 세가지 visualization 기법을 활용하여 우리의 모델 성능을 확인해 보았다.**

- CNN Filter Visualization

<center><img width="100%" alt="image" src="./images/Evaluation&Analysis/filters.png"></center>

> 초반필터에서 간단한 패턴과 선을 잘 파악하고 있다는 것을 알 수 있다.

- T-SNE Feature Embedding Visulaization

<center><img width="100%" alt="image" src="./images/Evaluation&Analysis/feature_embedding.png"></center>

> 클래스가 많아 4개로 나누어서 시각화 하였다. 약간의 오차는 있지만 각 클래스 별로 잘 분포되어 있는 것을 볼 수 있다.

- CAM (Class Activation Map) Visualization

<center><img width="100%" alt="image" src="./images/Evaluation&Analysis/cam.png"></center>

> 각 클래스별 CAM(Class Activation Map)이다. 모델이 이미지의 음식 부분을 사용하여 특정 클래스로 잘 예측하고 있는 것을 알 수 있다.

# V. Related Works
### Tools, libraries, blogs, or any documentation that you have used to to this project.
**툴(Tool)**: Python

**라이브러리(Library)**: **PyTorch**(torch, torchvision)

**블로그(Blog)**: <1> **Kaggle**(https://www.kaggle.com/datasets/harishkumardatalab/food-image-classification-dataset) **(Dataset)**

<2> https://wikidocs.net/164796

<3> https://velog.io/@kgh732/%EB%B6%80%EC%8A%A4%ED%8A%B8%EC%BA%A0%ED%94%84-AI-Tech-U-stage.-3-3

**논문(Paper)**: <4> Deep Residual Learning for Image Recognition (https://arxiv.org/pdf/1512.03385)

<5> Very Deep Convolutional Networks for Large-Scale Image Recognition (https://arxiv.org/pdf/1409.1556)

<6> ImageNet Classification with Deep Convolutional Neural Networks (https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)


<7> Learning Deep Features for Discriminative Localization (https://arxiv.org/pdf/1512.04150)

<8> Visualizing Data using t-SNE (https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)

<9> Visualizing and Understanding Convolutional Networks (https://arxiv.org/pdf/1311.2901)

# VI. Conclusion: Discussion

