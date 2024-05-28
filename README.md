# Food Image Classification


# Title
## ResNet을 이용한 음식 이미지 분류 및 실사용 예제

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
우선, 음식 이미지 분류를 다양한 deep learning 모델(resNet50, Efficinet, VGG)을 활용하여 성능비교(accuracy, error rate 등)를 해보고, 최적의 모델을 찾아보려고 한다. 그 뒤, 해당 모델을 이용하여 본격적으로 이미지 분류 학습을 하고 CAM(class activation map)분석을 통하여 가중치를 시각화 하고 어떤 가중치들이 분류에 있어서 주요했는지를 분석하고자 한다. 이를 통해서 음식 이미지 분류에 있어서 key가 되는 부분을 확인하여 궁극적으로 성능 개선을 통해 실생활에서도 오류없이 적용가능한 모델로 발전시킬 수 있는 지를 분석해본다.

# II. Datasets
### Describing your dataset
https://www.kaggle.com/datasets/harishkumardatalab/food-image-classification-dataset

kaggle의 Food Image Classification Dataset을 이용하였다. 24k의 이미지들로 이루어져 있으며, 34개의 서양, 인도 음식으로 이루어져 있다.

### (class) 
> **Baked Potato**, **Crispy Chicken**, **Donut**, **Fries**, **Hot     Dog**, **Sandwich**, **Taco**, **Taquito**, **apple_pie**, **buger**, **butter_naan**, **chai**, **chapati**, **cheesecake**, **chicken_curry**, **chole-bhature**, **dal_makhani**, **dhokla**, **fried_rice**, **ice_cream**, **idli**, **jalebi**, **kaathi_rolls**, **kadai_paneer**, **kulfi**, **masala_dosa**, **momos**, **omelette**, **paani_puri**, **pakode**

<center><img width="100%" alt="image" src="./images/imgs.png"></center>  <br>

# III. Methodology
### Explaining your choice of algorithms (methods)

### Explaining features (if any)

# IV. Evaluation & Analysis
### Graphs, tables, any statistics (if any)

# V. Related Works (e.g., existing studies)
### Tools, libraries, blogs, or any documentation that you have used to to this project.
**Tool**:

**Library**: **PyTorch**(troch, torchvision)

**Blog**: **Kaggle**(https://www.kaggle.com/datasets/harishkumardatalab/food-image-classification-dataset) **(Dataset)**

**Paper**: Deep Residual Learning for Image Recognition (https://arxiv.org/pdf/1512.03385)

Learning Deep Features for Discriminative Localization (https://arxiv.org/pdf/1512.04150)

EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks (https://arxiv.org/pdf/1905.11946)

Very Deep Convolutional Networks for Large-Scale Image Recognitio (https://arxiv.org/pdf/1409.1556)



# VI. Conclusion: Discussion

