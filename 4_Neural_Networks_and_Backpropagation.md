# Lecture 4. Neural Networks and Backpropagation

\<Review>

### Cross Entropy란?

- loss function
- GT(정답인 class는 1개 : 1, 나머지 0)
- 모델이 예측한 score를 1에 가까워지면 loss가 0에 가깝게, 0에 가까우면 크게
- multi class classification에서 많이 쓰임 (수업 때 많이 쓰일 것임)

### Gradient Descent

- 미분값을 이용해 loss function값을 줄이면서 update하는 방식

### Gradient Descent 한계점 3가지

1. convex(아래로 볼록)하지 않은 경우에 적용시 global optimal에 갈 수 없다 (많은 optimization에서의 한계)
2. differentiable하지 않은 function에 사용할 수 없다
    1. 해결 : survey loss function으로 해결
3. 느리다 : 전체 데이터를 가지고 gradient 추정
    1. 데이터 양 커지면 계산량 많아짐
    2. 해결 : subset에서 minibatch
    

### K-fold Cross Validation

- 하는 이유? (왜)
    - train data 전체를 training에 쓰면, 모델 성능 어떤지 잘 모름
        - train data를 evaluation에 쓰면 당연히 잘 할 수 밖에 (이미 봤던 문제니까) : 진짜 성능 아님
        - 미리 문제를 빼놓고, 안 본 문제를 얼마나 잘 푸는지 확인하기 위해
    - 어느 부분을 빼느냐에 따라 성능이 다르게 나올 것임
        - k-fold를 사용하면, 모든 부분이 공통으로 빠짐
        - 평균적인 성능 알 수 있음
        - 공정하게 evaluation을 해 모델의 성능을 좀 더 정확히 알기 위해서 사용한다.
        

### Q. k-fold cross validation할 때, evaluation을 여러개

매 실험마다 하나의 weight값이 나올텐데, 최종적인 모델의 weight값은 어떤 것으로 택하나?

- ex: 5 fold
    - same model, different hyperparameter
- 여러 방법
    1. validation  값이 가장 잘 나오게
    2. 5개 모델로 모두 예측 시키고, (weight들은 그대로)예측값을 평균시키기 : 앙상블 모델
        - 자기가 본 80%의 패턴을 학습
            - 이것 역시 noise에 overfitting
            - 서로 다른 80% 데이터에 대해서 이러는거면,
        - 1개로 했을 때보다 더 잘 나옴
        - 단점 : 계산량 많음
    
    이 외에도 여러개
    
- Sampling이 uniformly ramdomly하게 이루어지면, 모델 성능에큰 차이 없음

<본 내용>

### 지금까지

image classification

- 모델 만들어서 (linear model 배움)
- softmax loss
- 모델 업데이트 위해 stochastic gradient descent 사용한다.


### Issues w/ Linear Classifier

- W에서 학습하는 것 : template
    - class마다 이미지 하나만 배운다
    - ex) car : 앞쪽, 빨간색 차로 학습
- template이 하나밖에 없는게 한계이다
    - 한 장 안에 우리가 데이터에서 학습한 패턴을 저장하는게 어려울 것이라고 짐작할 수 있음
    - 옆모습, 노란색 차가 들어오면, 차라고 인식 못 할 수도 있음
    
- score가 다 linear하게 모델링이 될 수 밖에 없음
    - 우리 모델이 linear한 패턴만 가지고 있음
    - score가 x1,x2 input에 대해 linear :  평면이 직선으로 밖에
- class가 input x, y에 대해서 nonlinear하게 분포하는 경우도 있음 (slide pg5)
    - linear decision boundary를 어떻게 그려도 완벽하게 모델링 불가능
- 해결 : nonlinear한 것의 feature를 한번 transformation, 그 후에 linear decision boundary적용하면 가능
    - ex) pg 6
    - 해결 : polar coordinate → 이후에 linear classifer
        - 반지름 , 각도 있다면 2차원 내의 모든 점 표현 가능
- input 차원보다 훨씬 높은 차원으로 보내면 어떻게든 linear하게 separable 만들 수 있음 (수학적임) → linear classifier 사용

- image에 대해서 얘기해보자
- 기본적 원리는 같음 :
    - 이전에는 pixel 기반으로 classifier 학습
- 특징을 갖는 feature를 뽑아서 이걸로 모델링 하고 싶음
    - ex
    - 색깔로 (color histogram)
        - 옆모습이어도 같은 개구리로 인식할 것임
    - 구역별로 나눠서 gradient 방향 바뀌는지를 특징점으로
    - bag of words 미리 만들어 놓고 패턴 매칭
    

### Image Classifier with pre-extracted Features

- feature를 어떻게 뽑았든 뽑은 image feature vector
    - 이걸 이용해서 label을 예측하는 모델 만든다 ( 여기에서는 꼭 linear한 모델은 아님)
- training할 때 backpropagation 을 하는데, 모델만 training : Image Classifier with pre-extracted Features
    - z는 x로부터 정해진 함수에서 뽑았음 (미리 주어진 것임)
        - 중간과정에서 우리가 feature 만듦

### What if we can train end-to-end, such that feature extraction step also takes gradient from the classification loss?

- Is it always good? (end-to-end)
    - image가 많고 크면은, pixel로부터 모든 패턴들을 찾아내야 하므로 정말 오래 걸림
        - dataset 이 적으면, feature 뽑기에 힘들고, 계산량도 많음
            - 기존에 feature extraction(고정된 함수)으로 만들어 놓은 모델을 사용해 먼저 feature를 뽑고 시작하는 경우도 많음
                - transfer learning
    - 가능만 하다면 성능은 더 잘 나옴
- Q.  과정 전체의 classification loss 계산?
    - input x는 모델에서 고정된 것임. y도 고정 . W만 업데이트 하는 것임
    - y를 예측하게 하고, 틀린 만큼 W를 업데이트 (W : parameter / x,y: 주어진 것)
    
    - 여기에서는 z가 주어진 것임
        - z : x로부터 이미 뽑아낸 feature
    - z는 우리가 업데이트 안 함
    - 사실 z도 x의 함수
        - 함수(feature extraction)는 고정되어 있음
            - 이 function도 우리 모델의 일부로 들고 와서 loss가 그것까지 update할 수 있게 하고 싶다.
    - **가정 없이 어떤식으로 이미지 표현하는게 classification 잘할 수 있는지 조차도 data를 통해서 배우게 하고 싶다.**
    

## Neural Networks

### Perceptron

- dendrite : input 받아옴

가장 기본 모델

- 여러개의 input 이 dendrite 통해서 들어옴, cell body에서 조합이 됨, activation function 통해서 output 나감
    - 우리가 배운 linear model은 이것의 special 한 case였음! (node 1개 기준)
        - x : axon
        - Wx : weight
        - activation function
        - y  = sum of wx+b : output

### Neural Network with a Single Layer

- 이걸 여러개로 놓을 것임

그림 : (linear model)

- 우리는 3072 차원의 이미지 다룸
    - x : 3072개의 숫자가 있는 벡터
- score (output) 10개
    - class가 10개
- W = 10 x 3072인 matrix

node 형식의 그림

- 각 dimension의 input값들은 모든 output들과 연결됨
- edge에 weight 있음 : 곱해서 다 더함
- 우리가 배운 linear model임
- 1 layer

### Multilayer Perceptron (MLP)

- 중간단계 거침
    - feature 학습
    - hidden layer
- 각각의 단계(layer)는 우리가 배운 linear model임
    - linear model 중첩시킴
    - 이렇게 하는 이유?
        - 3072개가 10개로 되는 것은 동일
        - f(x) = w2(w1*x)
        - 행렬에서 w2*w1 = w라고 놓으면 f(x) = w*x
            - 중첩해서 쌓아도 하나의 linear model과 같아짐
        - 여기에 변화를 주려면 activation function 필요
            - a1, a2 : non-linear components
        - (x) = a2(w2*a1(w1*x)) : 이제 이 함수는 linear model로 치환 불가능 (non-liner function이 됨)

### Activation Functions

6강에서 자세히

자주 쓰는 것

- Sigmoid
    - 확률적인 해석이 잘 된다
    - 초창기 때에 많이 사용(요즘은 안 씀 → 이유는 6강에)
- tanh
- ReLU
    - 들어온 input이 양수면 그대로, 음수면 0으로
    - 많이 쓰임 (이유는 6강에)
    

### Implementation : 2-layer MLP

- 앞부분 : data에서 오게 될 텐데, 여기에서는 데이터 안 읽어와서 우리가 설정해야 함 (실제로 코딩할 때에는 안 할 것임)
- n : number of examples
    - 1개의 batch만 가정
- d : input dimension
    - 원래는 3072, 여기에선 1000이라고 씀
- h : hidden layer에서의 node 개수
- c : class 개수

- randn : random number

1000번 돌린다 (임의로)

밑에부터 neural network

- x.dot(w1) : w*x (linear component)
- sigmoid function (모양을 수식으로)
    - 이걸 activation function
    - input : W*x
- hidden layer * w2, activation function안 썼음
- y_pred와 y의 squared loss 계산해서 출력

뒷 부분 

- gradient descent
    - 수업 하면서, 숙제에서

### Computing Gradients

gradient 계산 방법

Stochastic gradient descent 이용해서, 얼마나 틀렸는지 이용해서 weight값 보정

- 필요한 것 : 전체 loss
- 각각의 weight값들이 이 답이 틀리는 것에 얼마나 기여했는가의 정도 : 미분값
- 이걸 알아내서 맞는 크기만큼 보정하기
- 이 미분법을 계산하기
- 이전에는 이걸 손으로 계산함..
    - LSTM은 omg..어디가 틀렸는지도 모름
    - 요즘 엄청 큰 모델은 계산 못 해..
    - backpropagation 많이 사용함

## Backpropagation : Computing Gradients

backpropagation의 원리와 구현 

### Computational Graph

- f(x, W) = Wx + b
    - forward pass : 이 값을 계산하기
    - backpropagation : 각각의 component들이 이 답이 틀리는 것에 얼마나 기여했는지를 계산
        - 뒤에서부터(loss에서 가까운) 계산하는 것이 빠름
    - foward pass → .. → backpropagation 반복

### Backpropagation Example

f(x, y, z) = (x+y)z

- suppose q = x+y
- suppose the input is x = -2, y = 5, z = -4
- q = -2+5 = 3, f = -12
- We need partial derivative of f w.r.t each variable (x,y,z)
- df/df : 맨 마지막 단의 미분값은 항상 **1**
    - 자기 자신으로 미분
- df/dz = d(qz)/dz = q
    - z: q이므로 3이어야 하는데 -4임, **3으로 변경**
- df/dq = d(qz)/dq = z
    - q에는 z인 -4로
- df/dx
    - 떨어져 있어서 바로 계산 못 함
    - chain rule (중간에 있는 q 사용)
        - df/dx = df/dq * dq/dx
        - = z*1 = z
    - x를 -4로
- df/dy
    - 마찬가지로 계산하면 y를 -4로

### Chain Rule

f는 하나의  node

input : x, y

output : z

- forward pass :  식을 계산
- backward : **upstream gradient**
    - 이 이전부터 더 뒤쪽에서 계산된 gradient값들이 전해짐
        - 시작 값은 1, 앞으로 올 때에는 항상 뒤에서 계산한걸 곱해서 들어옴
- node 안에서 **local gradient** 계산함
    - ex) f = x+y (=z)
    - dz/dx, dz/dy 같은 것 계산
        - node의 **output을 input에 대해서 미분**
    - node 안에서 일어나는 식에 대해서만, 바깥은 알 필요 없음
- 나갈 때에는 그 둘(**upstream gradient, local gradient**)을 곱한 것이 각각의 input으로 나감 : **Downstream Gradient**
    - 전체 loss를 input으로 미분한 값 (dL/dx, dL/dy)
    - 그 둘(**upstream gradient, local gradient**)을 곱한 것
- 이를 반복해서 맨 앞 node까지

### Another Example : Logistic Regression

- sigmoid function 안에 linear function있음 : logistic regression
- 많이 쓰임. 구글에서 모든게 logistic regression이었던 시절 있었음
- forward pass → backpropagation
- backpropagation
    - 맨 끝에는 upstream gradient은 **1** (always!) (다음 node 없음)
- **upstream gradient = node의 output값**
- **local gradient = f'(node의 input 값)**
- **결과 : upstream gradient * local gradient**
- **더하기(+)**
    - local gradient = 1
    - 결과 : **같은 loss값이 그대로 흘러 들어간다**
- **곱하기 (*)**
    - 미분값 : 상대방의 input 값(= local gradient)
    - 결과 : **서로 switch하고, 거기에 upstream gradient를 곱한 값**을 가지고 나감

(강의노트로 복습하기)

### Patterns in Gradient Flow

- add gate : gradient distributor
    - gradient 값: 둘 다 **upstream gradient** 값 그대로
- mul gate : swap multiplier
    - gradient 값: **상대방의 input값 * upstream gradient**
- copy gate : gradient adder
    - input 1개, output 여러개일 때
    - gradient 값: **upstream gradient의 합**
- max gate : gradient router
    - 컴퓨터비전에서 많이 쓰임
    - **max값이 선택됐던 쪽의 gradient : upstream gradient**
    - **반대쪽 gradient : 0(gradient 값 없음)**
        - 왜?
            - 미분 : 변화율
            - max가 선택이 안 되는 곳은 회로에 영향을 안 줌 (그래서 미분값이 0임)
    - 두 input node가 같은 값이라면, 두 node의 gradient는 모두 upstream gradient 값을 가짐
- Q)
    - 정수 - 인간이 이해하는 방식
    - 컴퓨터 - 유한한 소수 표현(bit수 제한-32,64)
        - 자기가 표현할 수 있는 만큼 최대한 정확하게 표현
    - bit수 많을수록 더 정확한 값을 표현 가능
        - 문제 : memory에 직결됨
        - 우리가 필요하지 않은 소수점이 중요한가? application에 따라 다름 (정확도 vs 동작만)
    - 사용할 범위만큼 잘라서 표현함

### Gradient Implementation

Gradient computation

- 처음 = 1
- 두번째 줄 : 식 단순화한 것 대입한 것
    - **sigmoid function(f(x)) 미분 값  = f(x) * (1- f(x))**

- 이렇게 계산한 gradient 값을 update에 사용
    - gradient descent에서 parameter new = parameter old - learning rate **미분 (이 gradient임)**

## Backpropagation with Vectors and Matrices

- 지금까지 : 설명 쉽게 하기 위해  input이 1차원인 scalar로 설명함
- 실제로 image input값은 vector, weight도 하나의 값이 아닌 여러 vector나 matrix 형태임
- 이럴 때에 좀 더 복잡하지만, 원리는 같음

### Vector Derivatives

- vector to scalar
    - input x : image feature (vector)
    - output : class score (scalar)
    - feature의 여러 dimension이 각각 score에 얼마나 영향을 끼쳤는지 계산하고 싶음
    - n개짜리 dimension vector가 input →미분 → n개짜리 dimension vector가 output
    - 각각 element가 짝이 됨
- vector to vector
    - x : n dim vector, y : m dim vector
    - output : 2차원 **matrix**
        - n X m
        - xi가 yj에 얼마나 영향 줬는지
        - 각각 element가 짝이 됨
    

### Backpropagation with Vectors

- Jacobian Matrix
- 아까랑 똑같이, **x,y,z가 vector**
- loss L is still scalar (우리가 예측하는 최종 값은  scalar ) →  그래서 마지막 gradient는 scalar임
- x, y, z are vectors of size x,y,z
- 벡터여도 맨 마지막의 각각의 element는 모두 1임
- **vector to scalar → gradient vector**
- dL/dz의 dimension크기도 z의 dimension크기와 같이야 한다 → upstream gradient는 z dimension (z*1)

- local gradient
    - **vector to vector → Jacobian (output : matrix)**
    - f
    - dz/dx : x*z dimension , dz/dy : y*z dimension
    - dimension 크기 = input 크기 * output 크기
    
- Downstream gradient :
    - local gradient * upstream gradient
    - dL/dx의 dimension  = (x * z)*(z*1)  = x*1
    - 확인해보면 input(x)의 dimension이 x*1으로 동일해서 맞음

- **Gradients of variables w.r.t. loss have same dimensionality as the original variable.**
    - **output의 dimension과 이걸 미분한 것의 dimension은 항상 같아야 한다**
    

### Example : Backpropagation with Vectors

### Backpropagation with Matrices

좀 더 복잡

하나의 node에서는

- 진짜 마지막 loss값은 scalar임 (1)
- upstream gradient의 dimension : z * z ( = forward pass 결과의 dimension)
- local gradient
- downstream gradient
