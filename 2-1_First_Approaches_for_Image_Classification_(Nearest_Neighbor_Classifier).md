# Lecture 2-1. First Approaches for Image Classification (Nearest Neighbor Classifier)

컴퓨터가 보는 영상/이미지 :  just a 2D matrix of intergers

### Image Classification : challenges

- scale variation : 얼마나 큰지 알 수 없다 (전체 사진, 부분 사진)
- viewpoint variation : 물체를 바라보는 각도에 따라 모양이 다름 ( 픽셀 단위로 보면 완전히 달라보임)
- background clutter : 보호색 위에 있으면 놓칠 수 있음 
- illumination : 조명  밤/낮의 사진의 픽셀이 완전 다름
- occlusion :  가려짐. 영상은 가려진 경우가 매우 많음, 사람은 뒤에 뭐가 있는지 알지만, 컴퓨터가 어떻게 알게 할 것인가? ?!?!
- deformation : 모양이 바뀌는 것(ex: 같은 사람, 같은 장소에서 완전 다른 자세), rule based로 학습시킬 수 없음(같은 사람인 것을 어떻게 학습시킬 것인가?)
- intraclass variation : 같은 차 안에도 다양한 차가 있음 (suv, convertable, color,,,)색과 상관 없이 '차'라고 어떻게 인식할까?

### An Image Classifier
image를 classify하고 싶다 
어떻게 짤까? 
시도 : 고양이를 인지시키고 싶음
- find edges(색이 바뀌는 곳에서 선을 따기) → 흑백그림 (0,1) : 훨씬 단순해짐
- find corners :고양이가 팔 하나 들면 완전히 다른 그림이 나올 것임

### Machine Learning을 도입 (Machine Learning : Data-driven Approach)
- data계속 보여줘서 알아서 패턴 찾게 (사람이 직접 묘사 안함)
1.  Collect a dataset of images and labels
2. Use machine learning algorithms to train a classifier
3. Use the classifier to predict unseen images

```python
def train(images, labels):
# some machine learning
	return model # train model
```

```python
def predict(model, image): # new image
# use the model 
	return predicted_label
```

  이후는 슬라이드 보기(14pg 이후) 
<br>
</br>

`두가지 classifier 배울거임 : nearest neighbor classifier, linear classifier`
<br>

### Nearest Neighbor Classifier
- training :  아무것도 안 한다 (메모리에 다 가지고 있다)
- predict : 기억하고 있는 것 (label)중 가장 가까운 것을 찾아서 return
- image domain에서는 어떻게 사용?
    - labeled train data 가지고 있음, query image가 나오면 가장 가까운 아이의 label찾아야함
    

### 제일 가까운 image란?

- similarity나 distance function이 필요함 (We need some similarity /distance metric between two images. But how?)
- 흑백 사진이라고 가정해서 설명해주심 (pixel마다 하나의 값만 있다고 가정) (원래는 pixel마다 3개값이 있음-RGB)
- 두 행렬간의 distance를 계산하는 문제가 됨
    - 두 장의 이미지 사진 크기가 다르면 어떻게? : 보통은 input image의 size는 같게 입력하게 설정함
    - L1 distance : absolute difference
    - L2 distance : squared difference
    - 둘이는 크게 다르지 않음

<br>
predict 시간이 많이 걸리는 것 : 매우 안 좋음
<br>

### K Nearest Neighbor Classifier

- 이전 : (1개의)가장 가까운 걸로
- k개의 가까운 점을 뽑아서, 이들의 다수결로 결정

#### Hyperparameters
- k, distance metric(L1 or L2)
- 알고리즘 자체는 똑같음, k는 몇으로 할거냐는 데이터에 따라 달라짐 (해봐야 알아)

### Nearest Neighbor Classifier Issues (문제점)

- pixel단위로 계산 : 전혀 정확한 척도가 아님
    - 좋은 distance metric : 사람이 인지한대로 distance가 바껴야함. 슬라이드에 나와있는건 사람이 봤을 때에는 다른데 l2 distance가 동일함
- test time에서 하나하나 similarity 비교 : 오래 걸림
- curse of dimensionally : 2d일 때, 적어도 16개의 데이터셋이 있어야 1d일 때 4개 데이터셋이 있을 때의 k nearest neighbor의 성능이 나온다 (차원 증가하면 필요한 데이터셋 지수함수로 증가)
- image 는 600x800x3..엄청 많음: first approach이지만 사실 안 좋은 방법이라 거의 안 씀

<br>

- normalizing
- dimension reduction : technique적용해서 600x800x3을 줄임
- etc
