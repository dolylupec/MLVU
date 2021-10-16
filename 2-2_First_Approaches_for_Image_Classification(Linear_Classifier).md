# Lecture 2-2. First Approaches for Image Classification (Linear Classifier)

### \<Review\>

#### Nearest Neighbor Classifier 란?

- 가까운 점들의 candidate : train dataset 전체
- train data 전체를 모델에 저장
- query가 들어오면  train data 전체를 비교해 제일 가까운 점에 해당되는 class로 classify
- time complexity
    - training : O(1)
    - inference (추론) : O(N)
        - N: 전체 training 데이터 수
    - 안 좋은 것임 :  training은 한번 하는데에 오래 걸리지만 구축하면 계속 쓸 수 있음,  한번 predict할 때마다 N번 봐야함
    - (질문) nearest neighbor- inference 할 때 train data size에 비례:  nearest neighbor가 유일
        - 이제부터 train data를 써서 무언가를 학습 (작은 모델 만들어놓기)
        - 기존에 얼마나 많은 data point썼는지 무관하게
        - dl에서 feature학습 한 후 low level에서는 nearest neighbor 쓸 수 있음

### k Nearest Neighbor Classifier

- #### k 값을 어떻게 정할지?
    - k가 너무 작을 때 (ex: k = 1) 문제점 : (k=1)제일 가까운 점의 class 를 따라감.
        - noise한 잘못 labelling된 데이터 point 때문에 섬이 생김 :  잘못 레이블링된 데이터 point에 너무 가까운  쿼리가 들어오면 그 포인트가 잘못 예측됨
        - k를 늘리면 주변의 다른 포인트 때문에 그걸 보완 가능 ( 노이즈에 좀 더 robust해질 수 있다)
- #### Nearest Neighbor에서 image classifier : pixel단위로 similarity 계산했음 (안 좋은 이유 2가지?)
    - 안 좋은 이유 1.  pixel level에서의 similarity는 high level(인간이 보는)에서의 similarity를 반영을 잘 못 한다.
    - 이유 2. image는 하나하나 비교하면 high dimensional data : 계산량이 많고 시간이 오래 걸린다
- #### Curse of Dimensionality
    - 우리가 다루는 feature들의 dimension이 늘어날 수록 그에 exponential한 수의 data point가 필요하다 (같은 수준의 modeling를 하기 위해서 : 특히 nearest neighbor 일 때)
    - high dimensional한 data 다루기 어렵다
    

## Linear Classifier

이전(knn) : 모든 training example을 기억함

### Parametric Approach

이제 :  어떤 function을 학습함 
- image를 input을 받아 연산하고 어떤 label인지 예측해주는 function 만들기
- f안에 있는 값 : 앞으로 학습할 값들(parameter(wieght)) : 어떤 값을 가지게 할 것인가(학습해야할 부분)

### Form of function f?
선형적인 모델부터 시작 (weighted sum of input pixels)

- image: 2차원 공간에 숫자들의 나열 ( 가로길이 X 세로길이 X 채널수)
    - 수업은 grayscale로 가정 (채널 1개)
- x(input), W
- we make 10 different W for each class
- 각각의 pixel값과 W를 다 곱해주면 score : 이게 가장 높게 나오면 그게 정답
<br>
- ex: input = 3072 numbers ,  output 10, cat에서 가장 큰 score 나오게
`f(x, W) = Wx + b`
  - output의 dimension : 10 x 1 (10 independent classifiers f for each class)
  - x의 dimension : 3072  x 1 (하나의  vector로)
  - W의 dimension : 10 x 3072
      - 10: class의 개수 (10개의 independent한 model)
      - 3072 : input : 각각 자기 class가 맞는지 확인
  - b의 dimension: 10 x 1
      - **input (x)와 관련 없는 (input independent)**
      - bias : affecting the output without interacting with the data x
      - ex : 세상에는 cat보다 airplane 사진이 많음
      - 더 많이 나타내면 b 값이 높음
      
### Bias Trick (pg.41)
      
- A separate variable b makes things complicated

- W'의 dimension : 10 X **3073**

- x'의 dimension : **3073** X 1

- 그럼 bias도 모델에 가져갈 수 있고, W,x로만 표현 가능
<br>
**linear model은 앞으로  Wx로 표현 (bias 있는거 가정하기)**
<br>
     
### Parametric Approach 장점
        
- W만 기억하면 됨(nearest neighbor처럼 모든 데이터 가질 필요 없음) : space efficient
- nn은 계산이 느림 : 얘는 한번에 계산 가능 (Wx)
- training 할 때 금방 안 걸림
- inference할 때는 example 개수에 비례(train data size와 비례하지 않음)
- Once trained, what we need is the weights W only. We do not need to store the huge dataet (space efficient)
- At test time, an example can be evaluated by a single matrix-vector multiplication(Wx). → Much faster than comparing against all training data(nearest neighbor)

### Illustration

- W의 각 row : 각 class


### Linear Classifier

- 2차원 공간으로 visualize
- y = ax+b에서 trick써서  y = Wx (W는 기울기와 절편 결정)
- line : decision boundary(n-1 dimension = 기울기)
- W값들이 조금씩 바뀐다 :  평행이동하거나 이동
  - W값 training : 조정해가면서


### Visual Viewpoint

- x, W 모두 쭉 펼쳐서 보고 있음
- input 값과 template(weight matrix)값 곱한다 → 다 더한다
- pixel level임
- W을 원래 모양대로 다시 배열하기
    - input (이미지)와 같은 모양임
    - W도 숫자들의 배열이니까 얘도 이미지처럼 그려보자
    - 흑백이면 단색
    - 채널이 여러개 (32x32x3 = 3072개) : R,G,B세군데에서 다 맞아야함 (각 채널에서 따로 weight 학습해야함)
    - 3가지 채널 각각 다 weight 학습해서  각 pixel들을 RGB 반영해서 그린것임
    - weight가 높다 : ex) 갈색 : red 가 높고, g,b는 낮아야 / 검은색 : 세채널 모두 낮음
- template(weight matrix) 이 input과 비슷한 패턴을 곱해야 가장 크게 나옴 (상대적임 : 그 중 가장 큰 값으로 label됨)
    - 이미지의 제곱과 비슷? : 제일 높게 나옴 (그래서 잘 classify됨)

<br>
- car 결과 보면 : 붉으스름함
  - 이 모델이 학습한 차는 train data에 있는 차임
    - 붉은게 많음, 정면샷이 많음
- deer :  배경색이 초원이라고 학습함 (다른 배경이면 안 좋은 결과 나올지도)
<br>

### 뒤집어진 말이면 W이 다르게 나오지 않을까?

- 만약 정상방향, 뒤집어진 말이 같은 비율로 있으면 형태 알아보기 어려움
    - 그런것들도 잘 학습할 수 있게 반영
    - 가지고 있는 데이터 그냥 쓰는게 아니라, 다양한 각도로 줘서 민감하지 않도록 data augmentation도함
    - 그래서 ship, truck은 다양한 방향의 data가 많아서 어떤
<br>
- training : template을 학습하는것이다
- testing : matching template
- knn과 비슷 : distance 계산해서 가장 큰 값 선택
    - 다른 점: knn은 **n개**의 data point와 비교, 그 중 제일 비슷한 애 찾아서 그것의  label이 결과
    - 10개의 template만 비교 (class의 개수(**k개**)만 비교) : 일반적으로  class개수가 n 개수보다 적음
<br>

### Bias를 쓰는 이유

- 데이터를 보기 전에의 prior knowledge 반영
- priority가 낮으면 : b가 낮음 (이class일 가능성이 낮음) : W와 input 매칭한 부분이 이를 극복할 정도로 높아야 해당 class로 예측함
- training data에서 class별로 imbalance가 있다면 bias로 보정 가능
  - 원치 않으면:
      - ex ) 의학쪽 : 암환자인지 예측, 근데 99%의 데이터는 정상 :  무조건 no라고 하면 99퍼의 성능이긴함
      - 근데 우리는 1% 잘 찾아내는 classifier를 만들고 싶음 :  data augmentation, rerating등을 통해 1%의 가중치 높혀줌 (다른 방법으로 보정 가능함)