# Lecture 3-1. Loss Functions and Optimization

Limitations of Linear Classifier

- 값의 범위가 없음 ( unbounded)
- 0~1 사이 수처럼 bounded score를 가지면 좋겠다 (그럴거임)
    - 확률로 해석하고 싶다

define how much we are confident

- 차이값에 비례하는 확률로 변환하는 operation
- 두 개의 차이 ( s1 - s2) 가
    - 양의 방향으로 커지면 class 1일 확률이 커짐
    - 음의 방향으로 커지면 class 2 ''

sigmoid : 우리가 원하는 모양의 함수임

- s1이 스코어 더 높게 나왔을 때
- s2가 스코어 더 높게 나왔을 때

 확률이기 위해

- 각 값은  0~1
- 다 더하면 1

pg5

좀 조작하면, 분모 똑같고, 분자가 자기 자신

- e^s1 >0

<Softmax Classifier>

Softmax Function

- 분모 : 각 class의 score 값을 exponential 씌워줘서 더함
- 분자 : 본인 class의 score 값을 exponential

example)

확연하게 score 차이 나게 (확률적으로해석) 해서 class 예측

How to find the Weights? <매우 중요>

pg 7

- W : data로부터 결정
- 초기에는 랜덤한 W 설정
- 우리가 예측한 것 :   y^ (y와 같으면 좋겠음)
- 얼마나 맞는지/틀렸는지 계량화 : **loss**
- **W을 업데이트**
- 원래 값과 비슷하게 예측할 때까지 업데이트

< Loss Function>

얼마나 맞았는지/틀렸는지 계량화하는 function

- input : y(ground truth), y^(우리가 예측한 것)
- 둘이 다를수록 숫자가 큼 (bad)
- 둘이 같을수록 0에 가까움

Discriminative Setting

- binary classification : +1, -1로 분류
- **margin-based loss :  y*y^에 기초로 정해짐**
    - y : +1 or -1, y^은 어떤 실수
    - 곱해서 양의 값이면 : 둘이 같은 부호(class예측이 맞았음)
    - 곱해서 음 : 둘이 같은 부호 (잘못 예측함)
    - 마이너스로 절댓값 크면 : 크게 틀렸다
    - 양의 값으로 절댓값 크면 : 아주 잘 맞음
    
    pg12
    
    - 왼쪽이 크게 틀림 :  loss 많이 주고 싶음
    - 오른쪽이 잘 맞은거임 : loss 적게 줌
    
    0/1 loss
    
    - 맞았으면 loss 없고 (0), 틀리면 1
    - 우리 모델이 얼마나 confident한지는 중요한지 않음(답을 뭐라 썼는지 중요)
    - 이 loss의 문제점 :  gradient descent로 업데이트 하는데, 미분이 안 되는 함수면 좀 힘들다
        - 미분이 잘 되는 함수 쓰자
        - 맞은거에  loss 아예 안 주면 그것도 또 문제가 될 수 있음
    
    개선한 loss function: Margin-based loss (yy^을 기준으로 만든 :  margin based)
    
    log loss
    
    - 값이 오른쪽으로 갈 수록 예측 잘 된 것임 (0에 수렴)
    - 예측 잘 안 했으면  지수함수비율로 loss 커짐
    - output이 확률로 해석될 수 있다
    - 모든 부분에서 미분 가능
    
    exponential loss
    
    - 더 극단적인
    - 조금만 틀려도 크게 반영
    - noise한 data를 다루면 robust하지 않을 수 있음
    
    hinge loss 
    
    - support vector machine의 모델임
    - margin 영역 이내 : 맞아도 약간의 loss줌, 더 확실하게 맞췄다면 loss가 0, margin 영역~틀린 것은 linear하게 증가함
    - 한 점을 제외한 곳에서 다 미분 가능 (미분값이 각 영역에서 constant한 값임 : 기억하기 편함, 계산 빨라. → 많이 사용함)
    
    정리
    
    - Exponential loss is severely affected by outliers, as it assigns extremely large loss to them. → Less suitable for noisy data
    - Hinge loss and log loss are widely used.
    - Hinge loss (SVM) is computationally more efficient. ( 각 영역에서 미분값이 constant)
    - Log loss (logistic regression) is more interpretable, as its output can be seen as p(y|x) (확률값으로 해석 가능)
