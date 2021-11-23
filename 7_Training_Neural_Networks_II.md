# Lecture 7. Training Neural Networks II

\<Review>

### Sigmoid를 activation function으로 쓰면 안 되는 이유 3가지?

- (가장 심각) killing gradient : 네트워크 뒤로 갈수록 gradient가 0에 가까운 값이 되면서 업데이트가 안 됨 → 학습이 매우 느려짐
- sigmoid output이 0을 기준으로 분포되어 있지 않아 문제 생김
- 지수 함수가 있어 계산이 오래 걸림

### ReLU activation의 문제

- 처음에 음수 영역에 있으면 업데이트가 안 되는 영역이므로 그 뒤로 뭐가 들어와도 계속해서 업데이트가 안 된다. (영원히 죽는다)
    - 해결 : initialize를 약간의 positive 값으로 준다.

### PCA란?

- 데이터를 분산이 가장 많이 보존되는 방향으로 axis를 align해서 eigenvalue를  정보량이 많은 → 적은 순서대로 sorting됨, 그 중 앞에 몇개만 취하면 dimension reduction

### Data Augmentation이란?

목적?

- 사진을 많이 모았어도 세상에 있는 사진들 중 정말 sparse한 부분임. 이왕 가지고 있는 사진을 최대한 많이 활용하기 위해

<계속 이어서 수업 나감>

### Where we are (이전 수업)

- Data Preprocessing
- Data Augmentation
- Activation Function
- Weight Initialization

### Today (오늘 배울 것)

- update할 때 optimization어떻게 할 지
- learning rate 어떻게 정할 지?
- regularization
- batch norm

## Regularization for Neural Networks

굉장히 중요한! 꼭 NN이 아니라 머신러닝 공부한다면 꼭 알아야 하는 부분임

### Overfitting

3강에 잠깐 다룸

- 모델을 fitting할 때 general trend에만 학습해야 generalize
- training data에는 noise가 있음 (general 벗어난)
    - 이것들도 fitting시킴

- 그림에서 o,x를 classify하려고 함
    - underfit : 잘 예측 못 함
    - normal : 약간의 utlier있음. 대체로 잘 맞춤
    - overfit : training data 완벽하게 맞춤
        - 매우 복잡한 식
        - 모르는 값이 들어왔을 때 정확하게 맞출 것인가? 그렇지 않을 가능성 높다
- 정확하게 overfitting 기준은 모른다.
- training을 정상적으로 잘 시키면 training loss는 계속 준다.
    - training을 mini batch에서만 evaluate해서 들쭉날쭉할 수 있긴 함. 전체 training set에 대해서 하면 항상 gradient를 원하는 방향으로 수정하기 때문에 좋아짐
- eval/validation data (안 봤던 데이터)에 대해서는 어느 순간부터 loss가 올라간다
    - high dimensional data에서 더 심하다.
    

Curse of dimensionality

- high-dimensional 데이터로 갈수록 같은 수준의 모델링을 하기 위해 필요한 data point의 개수가 exponentially하게 늘어난다.

- high-dimensional data에서 data는 많이 필요한데, 그만큼 많지 않으면 데이터 하나가 담당하는 영역이 넓어지면서 general pattern뿐만 아니라 noise까지 차지하게 됨
- 모델이 너무 복잡하게 생겼으면 이러한 노이즈도 학습할 수 있는 능력이 생겨 더욱 더 overfitting일어날 수 있다.

Regularization으로 해결함

### Regularization

- 전통적인 방법 : adding an additional penalty term in the objective function
- 모델은 최대한 자유도를 주고, 계수값을 의미 있을 때에만 큰 값 가지게, 나머지는 매우 작은 값 가지게 강제 (regularize)

- Linear Regression
    - minimize : Y와 Xtheta가 비슷해져야한다
    - Xtheta가 Y

### Ridge Regression = L2

- regularization에서 이걸 더해줌
- theta값들의 component마다 제곱해서 더한다
- 이를 minimize한다 : theta값이 커지면 손해를 보는 구조
    - 언제 커지냐?
    

다시

ML에서 하는 일 : theta값을 정하기 위해 학습

- lambda가 커질수록 : 해당 항을 억제
- lambda가 작을수록 : 원래 regression 문제 푸는 것에 가까움
- lambda가 0이면 : regularization이 아예 없음

그림 : 2차원 문제로 가정 (theta1,theta2)

- data만으로 풀었을 때 가장 optimal한 theta값 : 빨간점
- (regularization) lambda값이 정해져 있다 : 원 안에 있는 값만 허용하겠다.
    - 밖에 있는 값이 optimal한 경우, 원 안에 있는 가장 가까운 점으로 가져오는 것이 ridge regression이 됨.
    - lambda값이 크면, 원의 크기가 더 작아지는 경우가 됨
    

regularization의 핵심 : theta 값이 어느 정도 이상 커지는 것을 방지

Q : parameter 값(theta)을 줄이는 것과 overfitting이랑 어떠한 관계가 있는 것인가?

- pg6
- normal → overfit일 때 식이 theta6,7,8,9가 새로 생김
- theta 6,7,8,9값이 0이거나 0과 가까운 아주 미묘한 값을 가진다면, overfit의 그림과 같이 복잡한 그림이 만들어지지 않고 normal처럼 단순한 그림이 만들어질 것임
    - theta값이 커질수록 해당 항의 영향이 커지면서 함수가 3차함수,4차함수,,,처럼 되면서 복잡해짐
- **따라서 단순한 모델을 만드는 것과, weight값을 작게 만드는 것은 서로 강한 상관관계가 있다.**

- 수학적으로 증명할 수 있지만, 수업에서 안 다룸 (???궁금하다!)

 

pg8, 9 : 모양이 원일 필요는 없음 

- 꼭 theta 제곱이 아니라 절댓값들의 합으로 할 수 있음
    - 제곱 :  큰 값을 많이 penalize
    - 절댓값 : linear하게 penalize이므로 큰 값이어도 상대적으로 덜 penalize

### Lasso Regularization = L1

- 뒷부분에 theta의 절댓값들의 합을 더함
- 그림이 다이아몬드 모양
- 바깥에 있는 점이 다이아몬드 영역 안에 들어오게 함
- (변에서 확장된 부분을 제외하고) **모서리**로 모이는 점이 많을 것임
    - 모서리 : **두 theta 중 하나가 0이 되는 점**
- pg 6 : 많은 theta가 0이 된다. → 항이 많이 지워짐
    - 복잡한 모델(overfit)로 시작했어도, 결국 간단한 모델(normal)로 fit될 가능성이 크다.
- Lasso encourages sparser representation of theta. (ridge보다 더 sparse하게 만들어줌)

- L2일 때 (pg 8)
    - 원 모양일 때, 모서리로 특히 모이는 것 없이 모든 곳에서 균등하게 값이 줄어든다.
    - 값이 전반적으로 작은 값을 가지게 된다.
- L1일 때 (pg 9)
    - sparser representation : theta 값들 중 상당수가 0이 되고, 0이 아닌 값이 몇개 안 남는다 → 식이 단순해짐 →  noise에 robust한 decision boundary를 만들 수 있다.

질문??? L1으로 모델의 식을 sparse하게 만드는 것과, L2로 모델 전체적 계수를 낮추는 것에서 성능적 차이는? (언제 전자/후자를 써야할지)

### Regularization for Deep Neural Networks

Q : Are deep neural networks susceptible to overfitting? (영향을 많이 받는가?)

A : Unfortunately, yes. Neural networks are particularly prone to overfitting.

따라서 neural network의 overfitting을 막기 위해 regularization을 잘 해야 한다.

Q : What should we do?

A : We will learn regularization techniques for neural nets.

- Weight decay
- Early stopping
- Dropout
- DropConnect
- Stochastic Depth

### Weight Decay

weight 값을 줄이는 것 (바로 위에서 배운 것임)

- neural network에서 weight들이 2차원으로 있음
    - **Ridge regression (L2 regularization) : 각각의 값들을 제곱해서 다 더한다**
    - **Lasso regression (L1 regularization) : 절댓값을 해서 더한다**
- cross entropy같은 object function을 minimize하는데에, 그 뒤에 **이걸** 더해주면 **weight값들이 어느정도 이상 불필요하게 커지는 것을 방지한다**.
- 이전에 말한 내용을 neural network에 적용

### Early Stopping

neural network뿐만 아니라 널리 사용되는 방법

- train data로 train하고 validation set으로 중간중간 평가를 하면서, validation loss가 다시 높아질 때 멈추면 overfitting이 거기에서 멈춘다.
    - 잘 결정해야 함. 약간씩 튀기 때문에 정말 나빠지는지 general한 trend를 봐야 함

어떤 기준을 가지고 멈춰야 할까? 기준에 따라 멈추는 시점이 다름

- loss는 우리가 직접 줄이고 있는 대상이 되는 function
- 평가하고자하는 metric이 다른 경우가 있음
    - ex) cross entropy 실제 metric

### Dropout

neural network에 특화된 기법

forward pass이후 backpropagation

**forward pass일 때 일부 node들을 매 번 랜덤하게 없애고**, 나머지 것들로 계산함

- 0.5 : 절반정도 무작위로 없앤다.
    - 주로 사용
    - 이렇게 하고 training을 하면 overfitting이 안된다.
- weight값들 학습하는게 목적 아니었어??이걸 제외하고 학습 시키는게 의미가 있나?
- 어떤 물체를 인식하기 위해 5개의 clue, 이 weighted sum
- 일부 가려도 고양이인걸 알아보면 좋겠음
- 자주 보는 사람의 얼굴은 상세한 특징 알기 때문에 마스크 써도 알아봄
- 부분적으로 가려져있는 얼굴 이미지 넣어주면 맞춰서 학습함
    - 쉬움, 많이 씀
- training할 때만  dropout
- test는 모든 노드 활성화해서 평가
- implementation

### Stochastic Depth

- depth차원에서 봄
- dropout이랑 비슷
- dl : 데이터에 맞춰서 깊게 쌓으면 좋다
- layer를 몇개씩 건너뛰면서 학습 (depth에 적용)

### Cutout

- data augmentation과 비슷
- 이미지 일부를 무작위로 잘라냄, 이럴 때에도 원하는 classification잘할 수 있게 학습 시킴
    - overfitting 덜 함
    - 한 장의 사진으로 여러개 만들어서 학습시킴
- 사진에 있는 여러가지 단서들을 가지고 학습하게 강제
- 예전에 작은 데이터 쓸 때 많이 사용함
- 모델이 크면 좋지 않음 - 중간에 전처리 많음

### Regularization in Practice

- dropout, batch normalization 은 거의 default로 하기
- 데이터 도메인에 따라 data augmentation (특히 작은 데이터셋일 때)
- early stopping같은 정규화 방법 적절히 사용하면 overfitting을 방지하고 모델을 잘 학습시킬 수 있다.

## Optimization beyond SGD

- gradient  descent
- **mini batch로 stochastic gradient descent**
    - 이것도 문제가 있음
    - 한쪽 gradient가 크게 업데이트 되면 progresses very slowly
    - 더 내려갈 수 있는 길이 있어도 gradient0이라 멈춤
    - mini batch가 메모리 크기 때문에 무한히 크게는 못 함
        - 데이터가 큰 경우에는 mini batch를 8~16처럼 작게 설정해야 함 → gradient estimation 자체가 정확하지 않을 수 있다

좀 더 자세히 봐보자

- jittering : 비효율적으로 지그재그로 업데이트 됨 (오래 걸림)
- local optimum :
    - saddle point(말안장모양) 가 문제임
        - 더 내려갈 수 있는데 멈춤
        - high dimension에서 매우 빈번하게 있는 일임
- Inaccurate Gradient Estimation
    - 데이터가 커질수록
        - 모든 데이터 학습시키면 너무 자원 많이 필요하기 때문에 데이터 전체 말고 sample을 뽑아 학습
        - 근데 이 sample이 극히 일부분이면 전체를 대표하는 값이 아니게 됨
        - 예측이 부정확해서 test
        

그래서 나온 아이디어

### SGD + Momentum

- momentum
    - 어떤 물체가 얼마나 빠르게 운동하는지 정량적으로 나타냄
    - mass와 velocity의 곱(product)
- inertia
    - 관성
    - 어떤 물체가 운동을 유지하려는 성질의 정도

- SGD 식
    - gradient와 learning rate곱한걸 빼줌
- SGD + Momentum
    - velocity를 추가해 업데이트한 것을 누적
    - roh : 0~1 사이 값
    - 좀 더 넓게넓게 이동할 수 있음
    - saddle point를 통과할 수 있음! (탈출 용도 : momentum)

### AdaGrad

- gradient를 데이터에 adaptive하게 한다
- gradient를 elememt-wise scaling을 한다
- gradient 누적한 divider 점점 커짐
    - gradient가 컸던 방향으로는 큰 값이 더해져서 나눠지므로 그만큼 작게 감
    - 작은 것으로 나눈 것은 오히려 커짐
    - 이렇게 적절한 방향으로 갈 수 있게 의도
    - 굉장히 많이 쓰임
    - 축(element)마다 다르게 scaling해서 그동안의 이동 이력(?)을 반영
- 단점
    - divider는 제곱해서 더하므로 단조증가함
    - 뒤로 갈수록 급격하게 천천히 이루어짐..

learning rate이 너무 빨리 줄어드는 것을 방지하기 위해

### RMSProp : Leaky AdaGrad

- decay rate 도입
    - 이번에 계산한 것과, 이전의 누적된 것과 가중합 (convex combination)

### Adam

- 궁극의 방법
- Ada (adaptive) + m(momentum)
- **RMSProp과 momentum**을 같이 적용
- 1차 : momentum (beta1 : decay rate)
- 2차 : AdaGrad (RMSProp) (beta2 : decay rate)

대강 concept 알기

실제로는 구현되어있는거 많이 가져다가 씀

optimizer는 많이 씀, Adam부터 많이 씀

- SGD도 많이 씀. optimizer는 여러개해보기

### First vs Second-order Optimizer

- First-order Optimization
    - 한 점에서 미분하고, 그 방향으로 얼마만큼 이동시킨다
    - 미분 : 어떤 점에서의 최선의 linear approximation (일차식)
    - 일차식에 근사/근거해서 내려감
- Second-order Optimization
    - 2차식으로 근사하는 것이다
    - taylor expansion
        - 식이 1차, 2차...로 expansion되고 끊으면 근사
        - 1차 : 1차식까지만 씀 (지금까지 한 gradient descent)
        - 2차 : scalar function을 2번 미분 : hassian matrix
        - H : 2번 미분하는 것이고, matrix이다
        
    - second -order를 쓰는 장점
        - 실제 모양에 더 가깝
        - 맨 밑에까지 **빨리** 갈 수 있음
            - 계산 횟수 대비 (100번 → 5번)
            - 계산 속도가 빠른 것은 아닐 수도 있음
    - hessian matrix
        - 제곱의 크기에 비례하게 됨
            - 수학적인 내용이므로 깊게는 x
    - H의 역행렬을 구해 업데이트
        - 역행렬 계산 :N^3 만큼 시간 필요
        - 요즘 딥러닝 N은 100만~1억
        - mini batch로 줄인다해도 계산상 너무 복잡
        - 차라리 linear하게 100번하는게 더 빠를지도
        
    - modern deep learning에서는 second-order optimization 잘 안 씀
        - 소개용이었음
    

### Optimizers in Practice

- Adam부터 시작하라.
- SGD + Momentum등도 써봐라
    - 종종 SGD 쓰는 논문 있음
    - 아마 learning rate, decay를 더 조정해야함
- Initial learning rate이 중요하다
    - 일반적 : learning rate 초기 설정 후 점점 줄어나가는 방식
    - 처음 learning rate을 잘못 잡으면 정확히 미분해도 펄쩍 뛰면서 이상한 곳으로 날라갈 수도 (다음 내용)
    

# Learning Rate Scheduling

- hyperparameter tuning에서 learning rate가 가장 흔하고 힘들게 하는 hyperparameter임..
- 너무 높게 주면 (very high learning rate)
    - 시작할 때 loss가 아예 발산해버림 (노란색 선 pg 32)
    - loss가 줄어드는 방향으로 업데이트 해도..!
- good learning rate (적절한 learning rate)
    - 잘 따라가서 local optimum까지 감
- learning rate 더 낮추면
    - 너무 조심스럽게 내려가서 오래 걸림
    
- setting : 처음에 얼마를 주고, 시간이 지날수록 decay
    - ex) 데이터셋을 한번 돌 때마다 1%씩 줄인다
    - 처음 : 넓은 지형에서 탐색
    - 어느정도 언덕 잘 찾고 내려가면 천천히 해서 잘 도착하기
    - 이것도 여러가지 방법이 있음

### 사례

- 교수님 연구 (arbitrary함) : 다음과 같이 coding으로 scheduling함
- 시작하자마자 10%의 시간은 linear하게 키음 (warm up)
- initial learning rate : 1e-4로 40%의 시간을
- learning rate1/10으로 줄여서 그 다음 25%의 시간동안
    - 이렇게 lr 줄였더니 train loss 더욱 급격하게 줄었음 (empirical result)
- learning rate 1/10 으로 또 줄여서 나머지 25%의 시간
    - loss가 더 급격하게 줄었음
    
- 몇가지 시도해봤는데 저게 제일 잘 됨

이렇게 learning rate 적절히 scheduling해주고 decay해주면

- loss도 줄고, accuracy도 증가할 수 있음


### Learning Rate Decay

- 보통 다 해보면서 어떤게 더 잘되는지 보고 결정
- Cosine
- Linear
- Inverse Sqrt
- 아니면 교수님이 하신 방식대로

### Learning Rate Decay : Initial Warmup

- 맨 처음에는 zero로 시작해 천천히 learning rate증가해서 적응한 다음, 준비가 됐다면 줄여나가는 방식
    - 안 그러면 이상한 방향으로 튐
- option임 (가진 데이터 문제에 따라 선택해서 적용)

# Batch Normalization

### Data Preprocessing

- zero centered data가 유리하다
    - gradient 계산, 업데이트 등에서
- input data를 zero centered data로 주면
    - 첫번째 layer는 zero centered data로 받음
    - 두번째부터 intermmediate layer에서는 relu 썼으므로 zero-centered 안됨
    - 뒤로갈수록 비효율
    - 어떻게 해결?

### Batch Normalization

매 layer마다 zero-mean unit-variance를 원하면 그렇게 되게 만들어라! (If you want zero-mean unit-variance activations, just make them so.)

- 전 layer에서 input받으면, 그것들의 평균으로 빼고 표준편차로 나눔
    - normalization해서 zero-centered unit-variance(분산 : 1) data로 만드는 것임
    - 평균과 분산 : data point가 축적이 되어야 계산 가능 (data 1개로 안 됨)
    

어떻게 하느냐

- mini batch에서 데이터를 N개를 한꺼번에 다룬다
- D(dimension of data) X N matrix가 input
- mini batch에서 계산
    - 거기에서 평균, 분산 계산
- Dimensionality of 평균, 분산, 새로운 output?
    - D, D , NxD

- layer마다 다음 layer로 넘어갈 때마다 값을 보정하면 activation 값 자체는 우리가 원하는 zero-centered unit variance가 되어 기울지 잘 계산하고 좋을 것 같음
    - 한가지 문제 : neural network는 서로 유기적으로 연결
        - 중간에 평균을 빼고 나누면 값이 깨지기 않나?
- 해결 : **다시 복원**
    - gamma를 곱하고 beta를 더함 (pg 40)
    - gamma가 표준편차, beta가 평균과 같으면 y는 원래 값과 비슷하게 복원 가능
    - gamma, beta  : 학습 가능한 parameter로 풀어놓고 데이터로부터 학습 가능하게 함
    - x^을 계산할 때 평균 빼고~ 하고, 다시 맞춰보라고 하면 원상태로 복귀 됨
    - gamma가 표준편차, beta가 평균과 비슷하게
- batch normalization할 때 앞의 식만 넣지 말고 뒤에 복원하는 식도 같이 넣어줘야 함

이제까지가 training할 때임

test할 때에는 어떻게?

At test time, how should we normalize

- test time에서는 mini batch에서 평균, 분산 계산 못 함
    - test에서는 데이터 하나씩 넣어서 답이 나와야 함

- 방법 : training에서 사용한 평균, 분산 값을 기억했다가 test할 때 사용
- 평균, 분산값이 고정
    - test 할 때 linear operator처럼 됨
    - 복원하는 것도 training할 때랑 동일한 값으로
    

### Why batch norm?

- train할 때, layer마다 zero-centered data가 보장됨
- neural network 학습 빠름, gradient 정확해짐, 수렴 빨라짐
- 약간의 regularization 효과  ???regularization vs regularization??
- 그럼에도 불구하고 계산량 복잡 x (linear operator 하나 더 한게 전부)
- test time 때 overhead도 없음

- batch norm 거의 표준처럼 사용한다

### 그래도  batch norm에도 문제점이 있다

(관련 새 논문 나옴)

- training할 때 평균, 분산 계산해서 기억한 후, test에서 사용한다
    - 괜찮아보임 (training과 test data가 i.i.d 샘플이면은!)
        - 같은 분포에서 나온 독립적인 샘플
- 실제 머신러닝 모델 사용하는 product에서 사용하면 training을 맨날 하는게 아님
    - 한번 training한 모델을 2년동안 사용함
    - 유튜브에 사용한다고 하면 (2년동안 모델 돌리면)
        - 2년 전의 영상 분포와 지금의 영상 분포가 다를 것음 (수, 카메라 화질, topic, 보는 패턴 등)
    - 2년 전에 만들어놓은 평균, 분산이 valid할 경우 낮음
- data distribution이 바뀌면 정확한 예측 안 될 수도 (모델 자체가 틀릴 수)

- 해결 : Batch Renormalization (논문)

### Implementation

- tf에서 batch norm 다 해줌
- 원리만 기억하고, 가져다 쓰자

- batch norm은 fully conneted or conv layer 이후, activation function 이전에 넣는게 common
    - 왜 : activation에 들어갈 때 zero-centered data가 들어가야 미분값이 효율적으로 잘 계산되기 때문
    

### More Recent Normalization

- 우리가 배운 것(Batch Norm) : channel 쪽에서 하는 것임
- 그 외
    - Layer Norm
    - Instance Norm
    - Group Norm
- tensor를 다양한 방향으로 평균 계산해서 norm하면 업데이트가 잘 됨
- 논문 읽어보기

- 실제로 모델 training할 때 그 때에 맞는 것을 사용
    - 일반적으로 batch norm 많이 쓰니 이걸로 시작해보기
    - 뭐가 좋은지는 데이터에 따라 다름~