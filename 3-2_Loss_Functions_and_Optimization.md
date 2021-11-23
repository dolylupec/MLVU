# Lecture 3-2. Loss Functions and Optimization

### \<Review\> 

- linear classifier의 그림에서 그래서 classify 어떻게 되는 것인가?
    - variable이 2개 : W 값에 따라 기울기 결정
        
        초록면
        
        - x1에서는 큰 변화 없음( w1이 0에 가까운 수일 것이다)
        - x2에서는 마이너스 값으로 갈수록 커짐 (w2는 음수일 것이다)
        
        노란면
        
        - x1, x2 값이 증가할 수록 커짐 (w1, w2 모두 양수)
        
    
    data로 유추해서 w값을 예측해 linear classification 만듦
    

#### linear bondary 이용해서 input 예측 

방법은 여러가지임

그 중 1개 :

- 직관적으로 위에서 그림을 바라보자. (input dimension으로  projection)
- x1, x2이 결정됐을 때 맨 위에서 보이는 색으로 classify
- 판과의 경계선이 x1,x2 space로 정사영 : decision boundary
- 3개의 class중 max값으로 classify하는 것과 같음 (수업시간에 말한 것)
    - 문제 있을 수 있음 : input이 세가지 아닌 그림이 들어오면? 세가지 class 모두 낮게 나올 것임. 그래도 그 중 조금이라도 미세하게 사슴이라고 할 수 있음
        - threshold를 설정할 수 있음

또 다른 방법

- 기준치가 정해짐
- 세가지 모두 기준에 안 맞으면 아닌걸로 판별됨
- 두 class에 해당하는 것도 있을 것임
- 각 class의 threshold가 decision boundary임 (면의 접선 아님)


### \<Review Q\>

f = Wx 
- W : 우리가 학습하는 parameter
- x : input data
- W의 size : 10 x 3072 (class 크기 X input dimension)
- linear classifier에서의 bias
    - bias가 필요한 이유?
        - W부분은 data와 dependant
        - bias는 data와 independent한 패턴을 학습하기 위해 추가했음 (세상 지식)  
 
- Softmax function 이란? 그리고 왜 쓰나?
    - score가 bounded하고 확률적으로 해석하기 위해서 쓴다
- loss function이란?
    - 현재의 머신러닝 모델이 얼마나 좋은지/나쁜지 quantify하기 위해 쓰는 함수


\<본 진도\>

Loss function

margin-based loss
- ground truth가 binary classification일 때, -1,+1로 세팅

### Probabilistic Setting

- binary classification
- ground truth : 1 (true), 0(false) (softmax의 연장선)
- 한 class의 예측값 : y^, 나머지 : 1-y^
- sigmoid function : exponential한 비율 , 항상 0 이상, 다 합치면 1이 보장
- class 2개 이상으로 연장 (class가 k개)
    - k-1개의 score가 있음, 마지막은 1- 이전의 score 합들
    - softmax loss가 예시임 : 모든 class의 score를 exponential씌우고 합/해당class로 나타냄
    - loss function 필요 : GT(ground truth : 해당되는 class는1, 아니면 0), predicted 두개의 **확률분포**간의 비교 (gt에 해당하는 확률분포에 가까울 수록 맞는것임)

### Cross Entropy

- n : data point 개수
- k : class의 개수
- yik : ground truth_ i 번째 데이터의 실제 class값이 :  k번째가 참이면 1, 아니면 0 (0 또는 1)
- y^ik : 꼭 0 또는 1일 필요 없음 (softmax 쓰면 0~1 사이의 값)
- binary classification (class 2개)
    - k = 2
    - 뒤에 항이 2개밖에 없음 (yi1 + yi2 = 1)
    - 이 식을 loss function으로 쓸 수 있음 (왜 ? 밑에 설명)
    

### yik (ground truth)는 각 i에 대해서 **하나(k)만 1**이고, 나머지(k)는 0이다.

- **0이 되는 항은 다 사라짐**
- 복잡해보이지만 1인 yi번째 항만 남음
- y^ik : i번째 example 중 우리의 예측 (predicted)
    - (틀린 class의 score 안 봄)**실제 답에 해당하는 class에 대해서 어떻게 score 주는지만 볼것**임(score높을수록(1에 가깝) 잘 한것임)
- - log를 왜 씌우나?
    - (0,1)
    - 우리가 넣는 predicted probability 값은 0~1 사이의 값임
        - **1에 가까우면(맞을수록) 0에 가까워짐**
        - 그래서 cross entropy는 우리가 원하는 loss function의 형태이므로 적합함 (널리 쓰임)
        

### KL Divergence

- 엄밀하게 말해서 distance는 아닌데, distance처럼 쓰이는 거리임
    - 종종 쓰임 (참고용)
- P라는 확률분포가 Q라는 reference 분포로부터 얼마나 떨어져 있냐를 측정하는 measure
- discrete한 변수 : sigma, 연속 : 적분
- 엄밀하게 말해서 distance는 아닌데
    - distance 성립하려면 : p에서 q로의 거리와, q에서 p까지의 거리가 같아야 함
    - 근데 여기서는 다름 (둘 다 멀어지는 경우는 있지만 asymmetric = 다름)
    - distance는 항상 양수, 완전히 같으면 0
    - triangle inequality를 만족하지 않을 때가 있음
        - triangle inequality : 삼각형의 두 변의 합은 나머지 한 변의 길이보다 항상 커야한다
        
<br>

#### 지금까지:

- ML의 일반적인 흐름 : 모델 만들고, 데이터 넣고, loss function으로 모델이 잘하고 있는지 아닌지 측정
- 만약에 모델이 잘 못하고 있다면
  - 얼마나 틀렸는지 받아서 피드백 : W를 업데이트 (밑에)


### Optimization

- best element(input x, y) 찾기, criterion(기준 : loss function / object function) 최적화
    - 부등식 문제랑 같음
- x, y가 어떤 값이어야 함수가 제일 높은/낮은 값을 가지냐 : optimization
    - 한학기 분량정도 양이 진짜 많은 내용임, 수업에서는 간단하게 다룰것임

#### primitive ideas
(기초적 방법들 - 중요하지는 않음)

- exhaustive search
- random search
- visualization (문제는 우리가 다루는 input은 수만차원이라 그리기 어려움)
- (수많은 차원의 값들을 하나하나 optimization)

#### 우리가 할 방법들 : Following the slope
- 미적분학 기본 필요함
- 미분 : 어떤 점에서의 변화율
- best linear approximation
    - funtion에서 특정 점을 찍었을 때, 그 점을 선형모델로 가장 잘 설명할 수 있는 : **접선/ 기울기 (그 점에서의 접선의 기울기가 미분임)**
    - 그래프상에서 theta(input)이 어느 값을 가질 때 cost function이 가장 낮은 값을 가지는지
        - 그러기 위해 현재 점에서 어느 방향으로 값이 증가/감소하는지 알고 싶음
        - 눈 가린 상태에서 산 내려갈 때 발 짚어보면서 아래로 내려가는 방향으로 걸어가는 것과 같은 개념임
    

### Gradient Descent 
수식
- theta : linear model에서의 W값임
    - 이 값을 조금씩 조절하고 싶음
        - object function을 미분해서, 그 기울기만큼 **뺀다**
            - 최저점보다 더 큰 값에 있으면 theta값을 줄이는 방향으로 업데이트 해야함:
            - 오른쪽에 있는 값들은 양수임 (줄여야 함)
            - 왼쪽에 있는 값들은 j(theta)기울기가 음수임 (늘려야함 → 빼면 +가 되어서 theta가 오른쪽으로 이동
        - 한번만 변화하는 식이어야 성립함 (일단은 넘어감)
        
- code
    - evaluate_gradient()는 따로 또 짜야 함
    - alpha : 한 걸음의 크기를 조절 (learning rate)
        - 산이 완만하면 큰 보폭, 산이 너무 경사가 심하면 조금씩 (문제마다 alpha의 최적값은 다름)
    - threshold보다 작아지면 빠져나옴
    - 생각보다 잘 안 됨
    

### Gradient Descent Potential Problems

- 우리가 풀려는 문제들의 optimal point가 눈에 잘 안 띔(매우 복잡함)
    - input 엄청 많음, loss function 찾는 것도 어려움
- saddle point (미분값은 0이지만, 최솟값은 아닌)
    - 최소점인지 확신 없음
    - local optimum : 극솟값
- loss function을 미분해서 그 기울기만큼 따라 내려가는 것임
    - 식 자체가 미분 불가능할 때도 있음
        - zero-one loss : classification이 맞으면1, 틀리면 0 (constant)
            - 안 좋은 점 : 미분이 안 되면 내려갈 수가 없음
- converge하는 속도가 느림
    - w1x1 + w2x2를 미분해서 간다고 가정
    - w를 업데이트 (기울기를 구해서 내려가기)
    - 지형도을 다 모름(가지고 있는 데이터 포인트만큼만 안다)
        - 모든 데이터포인트 x1, x2를 넣어서 y 계산하고, 그걸로 **추론하기 (한계)**
        - 데이터가 너무 많음 : 다 미분해서 계산하려면 너무 오래 걸림 (천천히 줄어들 것임)
        - 좀 더 개선한 방법 : stochastic gradient descent

### Stochastic Gradient Descent

- idea : 데이터가 많다고 무조건 다 쓸 필요는 없다.
- 현재 모델의 W값들로 데이터를 넣어서 어느 방향으로 W를 조정할지 판단할 때, 모든 데이터 안 써도 어느정도는 맞는 방향으로 W의 방향을  이동할 수 있다.
- **minibatch**를 가지고 함 : 32/64/128/256/8192 개만을 가지고, 이걸 믿고 이동하는 것임
    - 하나마다 하면 noise할 수 있으니
    - 데이터가 저만큼만 있다고 가정해서, 무작위로 뽑으면서 방향 정하기
- 64, 128개정도로 어느정도 방향 예측할 수 있다.
- diminishing returns
    - 어느정도 개수 이상일 때, 데이터 개수를 늘려서 계산량에 비례하는만큼의 얻는 것은 적다
<br>

- W값들이 데이터가 들어올 때마다 우리가 얼마나 틀렸는지 가지고 업데이트 이루어짐

- 계속 하다보면 모델의 예측치(y^)이 ground truth(y)와 같아질 것임

- 발전이 없으면 멈추면 된다.

### Stochastic Gradient Descent 사용한 loss function 값

- 흐릿한 것 : 실제 측정값, 선명 : 스무딩한 값
- Q : Why do we see this noisy curve? (원래 값이 들쭉날쭉한 이유?)
    - 항상 똑같은 데이터만을 가지고 계산하는 것은 아님.
    - 이전의 minibatch를 토대로 방향 업데이트(100퍼센트 맞는 것은 아님)
    - 다음번에 뽑은 minibatch를 토대로 보면 또 다른 방향이 나올 수 있음
    - 그래도 전체적으로 봤을 때에는 줄어드는 방향으로 가는 것을 볼 수 있음
- Q2 : When to stop training? (언제 멈춰야?)
    - 몇번 돌리면 바닥까지 가더라..(대충) : 그만큼 돌리고 끝냄
    - loss가 더 이상 줄어들지/늘지 않는지를 보고
        - 실제로 구현하려면 어려움 (들쭉날쭉하기 때문에, 한번 나빠졌다고 바로 멈출 수는 없음. moving average 사용 - 계속해서 발전하지 않는게 20번정도 되면 멈추는 것처럼 (수업 뒤에 다시 설명)

#### \<Overall Review\>
ML based approach (모든 ML 모델의 flow) 

- ML is data-driven approach (데이터 기반으로 모델을 만들고 학습함)
- 모델을 만들고 (현재는 linear model만 배움)
- 모델 parameter(W)값을 어떤 값으로 초기화
- training data(x)를 하나씩 또는 batch 단위로 넣어서 현재 모델을 이용해 예측
- 예측된 것(y^)이 ground truth(y)와 비교해서 얼마나 좋은지/나쁜지 quantify
- 그것을 이용해 다음번에는 더 잘할 수 있도록 W를 업데이트해서 모델 training

우리가 배운 것들

- loss function 이용해서 어떻게 quantify할 것인지
- (W값어떻게 업데이트 할 것인지)

앞으로 배울 것들

- 특정 문제를 풀기 위해 어떻게 모델을 디자인해서 만들 것인지

Q : minibatch마다 가지고 있는 bias때문에 optimizing과정에서 감소함수가 아니라고 하면, full batch를 넣으면 항상 감소함수가 되나?

- no, full batch라고 해도 어차피 유한한 sample임 (noise의 최소일 뿐), 항상 감소함수라고 장담할 수 없음.

### Cross Validation

우리가 학습하고 싶은 것 : **데이터 속에 있는 일반적인 패턴(general한 패턴)**

- x값들로부터 label을 예측해내는 general한 패턴
- why?
    - 지금까지 수집한 과거 데이터뿐만 아니라 미래 데이터를 잘 예측하고 싶음
    - 일반적인 패턴 : 이전에도 그래왔고 앞으로도 그럴 것이다 - 를 학습하고 싶음

- 너무 training data에만 완벽하게 학습시키려고 하면 **overfitting 일어남**
    - 문제집만 본다고 시험을 잘 보는 것은 아님
    - training sample을 다 외운다고 잘하는 것이 아니라 패턴을 학습 시켜야 함
    

### Test dataset (eval data)
- 시험 문제를 따로 뽑아놔야 함 (그걸로 test해봐야 잘 하고 있는지 확인) 
- **ML 모델을 training하는 전체 과정에서 절대 쓰이면 안됨!!**
- 다 완성된 후에 한번 쓰기
- 우리가 가지고 있는 전체 데이터의 10~20% 따로 빼놓음
- 문제 특성에 따라 다르게 나눌 수도 있음
    - 특정 날짜 이전 데이터 : train data, 이후 데이터 : test data
    - 미래에 대해서 classification할 때 좋을 수 있음
    
- 어쨌든 test set은 따로 빼놓아야 함 - 나중에 이걸로 평가


### Choosing Hyperparameters

- train data로 train하기
- 진짜 모델 train할 때 데이터 x 넣고 y 나와서 예측하고(→ W 업데이트) train data 있어야

- 더불어 hyperparameter가 있음
    - model parameters : layer 길이,크기, activation function, regularization
    - learning parameters : learning rate(step size), 등등 정해줘야 함
    - 모델/데이터에 따라 어떤 값을 쓰는지 다름 (사람의 intuition에 따라 결정됨)
        - 여러가지 해보고 그 중 좋은 것을 고름
        - 이 때 생기는 똑같은 문제점 : training data에서 그걸 측정하면, training data에서 제일 잘 되는 값을 찾는데 이는 test dataset으로는 제일 generalize가 잘 안 될 수 있음 (거기에 있는 noise까지 고려가 되어서)
        

### Validation set
- **test와 마찬가지로, hyperparameter를 결정하기 위한 dataset을 따로 빼놔야함**
- pg 40 : 일반적으로 쓰이는 방법
- evaluation 10%, validation 20% 빼놓고 나머지 70%로 training

#### Cross Validation

- step 1 : Train a model using the training set.
    - training set : 전체 데이터의 70%로 모델 training
- step 2 : Evaluate the model using the validation set.
    - training한 모델을 validation set으로 평가
- step 3 : Repeat step1~2 with different hyperparameters, and choose the best one.
    - hyperparameter를 바꿔가면서 1,2 반봅
    - 어떤 hyperparameter 조합에서 validation set(training할 때 안 쓴 데이터셋)에서 제일 잘 되는지 평가할 수 있음. 그 중 제일 잘 되는 것을 고른다.
- step 4 : Train the model with the chosen hyperparams in step3, on training + validation sets(90% of total dataset).
    - (optional) hyperparameter 정했으면 , 데이터셋이 만약 부족했다면 validation set으로 쓴 20%도 아까움
    - 20%도 포함해서 총 90%의 데이터셋으로 다시 train함
- step 5 : Evaluate the model in step4 on the eval set.
    - 진짜 eval set을 한번 써서 평가 (성적 몇점이다)

- **제일 중요! Eval set must not be used until you finish step 4. (eval set은 어떤 방식으로도 쓰면 안 됨)**

### K-fold Cross Validation
- 좀 더 체계적으로 하는 방법 
- dataset이 너무 크지 않으연 30% 빼내는게 아까움
- 어떤 30%인지에 따라 70%의 train dataset 분포가 달라질 수 있으므로
- 공평하게 ex:5개로 나눠서 각각을 한번을 eval 또는 validation으로 쓴다.
- 하나씩 떼어놓고 나머지 (4개) 80%로 training
- 5개의 모델을 만들어서, 5개의 모델을 성과를 평균을 내는 방식으로 cross validation할 수 있음
- 장점
    - noise로부터 자유로워짐
    - 모델의 성능을 정확하게 평가 가능
- 똑같은 것을 5번이나 해야되기 때문에 시간이 오래 걸림
    - 실제로는 많이 사용되지는 않음
    - 작은 dataset으로 돌릴 때 간혹 논문에서 볼 수 있음
    

- k-fold cross validation사용하는 예시

  - (pg 42) 이전에 배운 K-nearest neighbor에서 중요한 hyperparameter중 하나 : k
  - neighbor를 몇개로 할 것이냐
      - 최적의 k는 dataset마다 다름
  - 5-fold cross validation으로 돌려봤을 때, 각 k값마다 5개의 점이 있음 - 평균으로 이어보니까 성능이 제일 잘 나올 때는 12일 때
  - 그려보면서 우리 데이터에서 k가 몇일 때 가장 좋다를 결정 가능 (이런 식으로 k-fold cross validation 사용하는 것임)
<br>

\<Question\>

- train data로 train하고, 이를 또 evaluate하면 train loss는 계속 줄어들어야하지 않냐?
    - 맞음. 단, train loss를 전체 training dataset에 대해 업데이트하고, 전체 데이터에 대해 evaluate할 때 경우에만.
    - train data는 엄청 크기 때문에 training진행될 동안 step by step동안 전체 evaluation은 시간이 오래 걸려서 잘 안 함
    - 대충 패턴을 보고 싶기 때문에 sampling을 해서 함 : 그래서 좀 튀는 경우가 생기는 것임
    
    - evaluation loss : overfitting이 되면 내려가다가 어느순간 다시 올라감
        - 다시 올라가는 시점부터는 : 전체 데이터셋에 있는 일반적인 패턴을 학습한게 아니라, training data에만 있는 noisy한 패턴을 학습하기 시작했다. (더 이상 generalize가 안 되기 시작)
    
- k-fold cross validation에서 결국 eval set이 train에 쓰인거 아닌가?
    - validation set만들 때/ eval set을 만들 때
    - **각각의 모델을 만들 때 eval set을 안 쓰는 것임**
        - 2,3,4,5로 모델을 만들 때 1을 안 보는 것임
        - 모델은 같은 모델임. 다른 데이터로 학습함
        - 그 모델이 얼마나 좋은지 평가하고 싶은 것임. 최종적으로 나오는test 정확도의 분포를 보고 모델이 얼마나 좋은지 결정
        
- saddle point / local optimum에 갇히면
    - 모멘텀 : saddle point를 탈출하기 위한 것

- supervised learning 경우 test set 또는 validation set의 label을 사용하면 안되는 건가?
    - test set : x, y 둘 다 사용하면 안 됨
    - validation set : 사실 꼭 hyperparameter 조정 용으로 사용할 필요는 없음
        - 우리가 train하고 있는 데이터 분포와 독립적으로 모델을 평가하기 위해 필요
        - x, y를 원칙적으로 쓰면은 안됨
    - 간혹 eval data의 x를 쓰는 경우가 있음 (y만 안 보면 되지 않나?) : 그것도 cheating임
    - 간혹..예외가 있을 수는 있음

- eval data = test data
- gt에서 label 값이 1이 여러개인 multi object 경우 앞에서 말한 single object처럼 train하면 되나?
    - 모델이 조금 달라질 수 있음
    - 전체적인 training과정은 같다고 볼 수 있음, 뒷부분이 다른 것임
    - 우리가 원하는 것
        - 각 class별의 score 나오게(training 전반적인 과정. 다 똑같음 )
        - score를 어떻게 해석해서 사용 (이 부분은 모델training과는 독립적)
            - class가 2개 이상이면 threshold 줘서 이상인게 여러개가 있다면 맞다, 아니면 없다
