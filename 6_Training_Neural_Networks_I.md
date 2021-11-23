# Lecture 6. Training Neural Networks I

\<Review>

### Convolution을 정의하는 4가지 hyperparameter?

- filter의 크기
- filter의 개수
- stride
- padding

4가지를 정의해주면 layer가 동작하도록 되어 있음 (tensorflow/pytorch)

### Conv layer가 fully connected layer보다 **parameter 수가 획기적으로 적음. 어떤 가정(assumption)을 했기 때문?**

- positional invariance
    - 특징은 어디에서든 나타날 수 있음. filter 1개 학습해놓으면 위치에 관계 없이 똑같은 것을 쓴다
- spatial locality
    - 작은 공간에서 특징 detect, 멀리 있는걸 고려 x

### Pooling layer란?

- 해상도, 이미지 크기, parameter의 갯수를 계속 크게 가져가면 계산량이 많아지므로 이런 것들을 줄이기 위해 사용

### image 크기 32x32x3channel 가정, 2x2 max pooling, stride size 2 : 이 layer의 parameter 개수는?

- (parameter : layer에서 학습해야하는 learnable parameter, weight값의 개수)
- output은 16x16x3임
- A : **0개**
    - 함정 문제
    - pooling layer에서는 학습하지 않는다. (정해진 연산을 그냥 수행함 ex:평균, max)


<본 수업>

머신러닝 방법으로 image classification 일반적인 방법 배움

Machine learning is data-driven approach

- 머신러닝 : 인간은 모델의 형태를 만들고, 파라미터 값들을 데이터로부터 찾아서 학습시킨다
- W를 임의로 초기화
- loss 이용해서 성능 평가, parameter(W)를 업데이트해준다
- 비슷해질 때까지

모델 어떤거 쓸 수 있는지 등등을 배움

배울 것

- neural network : activation function 중에 어떤걸 골라 써야할지?
- weight(parameter) initialization?
- training data preprocessing? (학습데이터 사용 극대화 위해)
- learning rate 비롯한 모델의 hyperparameter 설정 방법, 업데이트 방법
- 언제 끝날지?(regularization 통해 overfitting 방지)

## Activation Function

non-linearity를 추구하기 위해 linear transformation 한 뒤 activation function 통과하면 neural network가 표현할 수 있는 범위가 linear function만에서 non-linear function까지 표현 가능해진다. (이전에는 이정도까지만)

- sigmoid
    - 항상 0~1 사이의 값이 나옴(보장)
    - 양수가 나오면 0.5~, 음수이면 0~0.5
- tanh
    - -1~1까지 (sigmoid와 비슷)
- ReLU
    - 0보다 작은 값은 0, 0보다 큰 값은 받은대로 나가는 function
    - non-linearity
    

각각의 장단점 보자.

### Sigmoid

- classification할 때 0또는1로 표현
- confidence 표현(softmax)
- 굉장히 매력적인 함수
- neural network 초창기일 때 무조건 이거 썼음
- layer가 깊지 않을 때 잘 동작함 (작은 scale에서는 나음 괜찮)

Reason to NOT use (deep neural network 나타나면서 문제점 3가지)

- kill the gradients
- sigmoid outputs are not zero-centered
- exp() is computationally expensive
    - computer 내부 계산에서 곱셈보다도 더 많음
    - 사실 요즘은 큰 문제는 아님. 앞에 두 문제를 더 자세히 보자

### Sigmoid Function : killed gradients

- sigmoid function = q일 때
- q' = q(1-q)
    - 재밌는 특성
    
- gradient when x=10?
    - q는1에 가깝, 1-q는 0에 가깝, gradient becomes (거의) 0
- gradient when x=-10?
    - q는 0에 가깝, gradient becomes 0
- gradient graph w.r.t. input x?

backpropagation

- upstream gradient 받아서 local gradient를 곱해서 나감
- local gradient가 input x의 절댓값이 크면 0에 가까워짐
    - upstream gradient 값과 상관 없이 0에 가까운 값이 곱해져서 나감
    - After passing this node, gradient is multiplied by around 0 if the input x is large.

- sigmoid를 통과할 때마다 값이 점점 굉장히 작아짐
    - 한번 0에 가까운 것이 나가면, 다음번에 다른 노드들이 받을 때 upstream gradient로 0에 가깝게 나오고 점점 작아질 것임
    - gradient descent를 통해 잘못한만큼 update해야돼!에 쓰임
    - 0에 가까운 값이 들어오는 것은 값이 거의 업데이트가 안 된다는 의미
    - sigmoid 통과하면서 더 이상 업데이트가 안 됨 (sigmoid를 deep neural network에 쓰면 안 되는 가장 심각한 이유)

### Sigmoid Function : Not Zero-centered Output

Consider what happens when the input is always positive (which, frequently happens in practice.)

- input data가 positive인 경우 (자주 발생함)
- ex) 추천시스템에서 평점 : 1~5의 양수값

linear transformation을 하고 sigmoid function을 씌워서 나오고 backpropagation한다 (그림)

- local gradient를 계산해보자.  q(1-q)임 (식은 8pg 파란박스)
    - sigmoid는 뭐가 들어가든 0~1 사이의 값이 나올 수 밖에 없음
    - xi가 항상 양수라고 가정
    - local gradient는 항상 양수일 수 밖에 없음
- 의미 : local gradient는 부호를 바꾸지 않는다.

Why is this a problem?

- Thus, sigmoid node does not change the sign of the upstream gradient for all wi. → All gradient elements are either all-positive or all-negative.→ Gradient update can go only to particular directions!
    - optimal gradient로 가기 위해 비효율적으로 업데이트하므로 필요 이상의 computing

- This problem is less severe than killing gradients, as final updates rely on multiple examples in batch.
    - 보통 mini batch를 사용해 여러 data point로 gradient 계산하고 그걸 평균내서 움직이므로 여기에서 보정이 됨. 완전히 지그재그로 가지는 않음
    

### Tanh Function

sigmoid와 비슷하게 생김

- output ranges : **[-1, 1]**
- 기댓값이 0
- zero-centered
    - sigmoid의 문제점 중 하나인 not zero-centered output 문제 해결

Reason to NOT use (그래도 문제점이 있음)

- saturated neurons kill the gradients.
    - 얘도 killing gradient 문제 해결 못함
    - tanh is simply a scaled sigmoid neuron
        - tanh는 본질적으로는 sigmoid이다
        - tanh(x) = 2q(2x)-1

### ReLU (Rectified Linear Unit)

- 딥러닝 들어오고 난 뒤부터 매우 많이 쓰이는 activation function
- **max(0, x)**

Reason to use

- Does not saturate (in + region).
    - saturate : 수렴
- Computationally very efficient.
    - exponentional function은 계산이 복잡
    - ReLU는 gradient를 따로 계산할 필요 없어서 계산량 줄어듦
- Converges much faster than sigmoid/tanh in practice.
    - 훨씬 더 수렴을 잘 하고 training이 빨리 된다.

But, reason to NOT use (이것도 완벽한 것은 아님)

- Not zero-centered output.
    - 항상 양으로만 감
- **Not differentiable when x=0**
    - 미분 불가능한 점이 있음
    - 둘 중 한 부분을, 또는 사이값으로 지정함
- **Dead ReLU proble**m: if an initial output is negative, it is never updated.
    - (가장 치명적인 문제임)
    - 한번 gradient가 0이 되면(output이 음수가 되면) 이후에 업데이트가 안 됨 → 해당 노드 업데이트 안 되고 죽는다
    - It may help to initialize ReLU neurons with slightly positive biases(e.g., 0.01)
        - 초기에 gradient가 0이 되어 빠지는걸 방지
        - 여전히 완전히 해결한 것은 아님

- 이를 해결 : Leaky ReLU

### Leaky ReLU (Rectified Linear Unit)

- negative 부분을 완전히 0이 아니라, 기울기 절댓값이 매우 작은 식을 넣어주자 (e.g.,0.01x)
- 완전히 0이 아니니까 약간 negative하게 나오게 해서 벗어날 수 있게
- ex) **max(0.01x, x)**

Reason to use (아까랑 장점은 비슷)

- Does not saturate (**in all regions**)
- Computationally very efficient.
- Converges much faster than sigmoid/tanh in practice.
- No Dead ReLU problem

Reason to NOT use

- An additional hyperparameter (slope where x<0)
    - 크게 어려운 문제는 아님

### ELU (Exponential Linear Unit)

- ReLU 약간 개선
- zero-centered output을 만들고 싶어서
- Leaky ReLU: negative 구간을 linear function을 쓰면 무한대로 계속 커지는 값이 나올 수 있음

ELU

- negative 구간에 saturation 식을 넣어 거의 0에 가까운 값으로 수렴하게 디자인
- 전체 기댓값이 0이 나오게 수학적으로 만든 것
- 식에서 alpha는 hyperparameter인 듯

Reason to use

- All benefits of ReLU
- Closer to zero mean outputs
- Negative saturation regime compared to (Leaky) ReLU adds some robustness to noise.

Reason to NOT use

- exp() is computationally expensive.
    - 계산량이 좀 많기는 하지만 큰 문제는 아님

### Activation Functions in Practice

실용적으로는,

- Use **ReLU**. Be careful with your learning rates.
- Try out Leaky ReLU or ELU to squeeze out some marginal gains.
    - 좀 더 잘 되는 방법 찾기 위해
- **Do NOT use sigmoid or tanh.**

 

 

## Data Preprocessing

깊이 들어가면 할 많으니 간단하게만

모든 input들이 positive한 값이 나오면 gradient가 지그재그로 (문제점임)

### Zero-centering & Normalization

**Zero-centering** 

- 아예 데이터 자체를 모두 positive 가 아니라 positive, negative 값 적절히 섞여 있게 centering해보자
- image 데이터 : 0~255 사이의 정수 (양수값) → 그대로 넣지 말고, 평균값을 빼주면 기댓값이 0이 되는 zero-centered data가 나올 것임
- 전체 평균을 빼주면 zero-centered data

**Normalization**

- 분산이 x축에는 작고, y축에서는 클 때
    - 분산이 안 컸으면 좋겠을 때도 있음
    - 기압 : 900~1100 헥토파스칼, 온도 : 0~30도 (각 단위때문에 차이 나는것임, 7배 중요는 아님 - 상관관계 없음)
        - 각각 영향 맞춰줌 (scale에 영향 받지 않게)
- 전체 std로 나눠주면 normalized data

### Why Zero-centering?

classification할 때 zero-centering하면 더 robust하다.

- zero-centering : 데이터를 평균만큼 평행이동
- 같은 linear model 학습 시
    - parameter가 살짝 바뀌면 (ex: 기울기가 바뀌면)
        - zero-centered data : 크게 영향 안 받음
        - original data (0에서 멀리 떨어진 데이터) : 벌써 틀림. classification할 때 매우 sensitive함. 틀리기 쉽다. 새로운 데이터 들어오면 쉽게 틀림. outlier가 있으면 거기에 영향 많이 받는다.

- Classification becomes less sensitive to small changes in weights.
    - see the slope change in the figure (green to orange)
- Easier to optimize.

### PCA & Whitening

PCA : Principal Component Analysis 

image 데이터(pixel 단위)에서는 잘 안 쓰지만, feature 단위에서는 많이 쓰임

ex) orignal data : x1,x2 양의 상관관계 가지고 있다.

- decorrelated data
    - decorrelate시키고 싶다 : 데이터의 분산되어있는 방향으로 axis 바꾸고 싶다.
- whitened data
    - 이후에 normalization을 시키면 정규분포화된 모양이 나올 수 있다.

### PCA

PCA 하는 방법 (앞의 그림과 매칭하기)

1. Zero-center the data:
    - X(i) : training data
    - X~ : zero-centered data
2. Estimate the covariance matrix:
    - X~(i) : dx1, X~(i)T : 1xd, X~(i)X~(i)T : dxd
3. Compute eigenvectors and eigenvalues of sigma~. Then, order them by eigenvalues in decreasing order:
    - U is a kind of ratation matrix; the covariance matrix sigma is rotated to be axis-aligned. Then, the covariance matrix becomes diagonal, meaning that each dimension is no longer correlated. Each diagonal entry lambda i indicates the variance(분산) of data points according to the i-th axis.
    - U : 일차변환을 해주는 matrix
    - UU^T = 1 → U^T : U의 역변환
    - 분산이 가장 큰(중요한 축) 순서대로  lambda1, lambda2,..로 sorting
    - 서로 decorrelation : 나머지 행이 0
4. If we choose the first k << d eigenvectors (with largest k eigenvalues) and discard the rest, we get a k-dimensional space such that the original data loses least amount of information (in terms of variance).

- PCA is one of the most widely used dimension reduction technique in ML.
- *intuitive : data에서 variance가 잘 보존되는 방향으로 k개의 dimension만 보존하고 나머지는 지워서 정보량을 줄였다.*

### Data Preprocessing in Practice

- Subtract the mean image
    - 이미지 전체의 평균을 빼서 zero-centering으로 시작
    - e.g., AlexNet
- Subtract per-channel mean
    - channel별로 평균 빼서
    - e.g., VGGNet
- Subtract per-channel mean and Divide by per-channel std
    - channel (R,G,B)별로 normalization 따로 계산
    - e.g., ResNet
- PCA and whitening
    - PCA is less common on pixel space, but widely used in a feature space.
    - e.g., YouTube 8M
    

(8강에서 모델 배울 것임)

## Data Augmentation

### Why Data Augmentation?

pixel level에서는 다른 사진, distance는 같음

어떤 모델을 training한다 : 원본이나 색 살짝 바꿔도 여자로 분류하는걸 원함

- 작은 변화에 invariant한 모델을 만들기 위해 data augmentation
- 아무리 많은 데이터가 있다 해도 전체 데이터의 일부분일 뿐
- 한 장 찍은걸로 최대한 활용해서 모델이 다양한 각도에서 학습할 수 있게
- label은 원본 이미지의 label과 동일한 것을 사용

- horizontal flip
- vertical flip(경우에 따라)
- random crops
- scaling (image 크기)

- test image도 resize함

- computational cost 많음
- 제대로만 하면 model 성능 감소는 없음

- 배경에 영향
    - 배경의 정의, independant하면 따로 떼어서 볼 수 있으면 좋긴 함 (어렵_

### Color Jitter

advanced

- filter 차이, 색을 randomize
- 본질이 흐리지 않을 정도로 조절
- training하는 데에 사용

- 문제의 domain knowledge에 따라 창의적으로 다양한 방법이 있음
- 방법 예시 1 : 그림판의 색 편집에서 "색상(hue), 채도(saturation), 명도(lightness)" 조절
    - 사람이 직관적으로 인식할 수 있게 조절 가능
    - RGB → HSL로 변환 가능한 공식 있음
- translation
- rotation
- stretching
- shearing
- random blocking
- add noise
- mix two images
- apply a filter

## Weight Initialization

Weight initialization

### Small Gaussian Random

- 0에 가까운 랜덤값을 넣어보자
- 정규분포가 되게 0을 중심, 작은 분산을 가진
- 0.01같은 작은 값을 곱해서 초기화해봤음
- tanh function 사용하면 activation이 0에 수렴하고,  gradient값이 layer 지날수록 0에 가까워져서 학습 X

### Large Gaussian Random

- 0.5를 넣어봄
- tanh 사용하면 activation은 0에 수렴하고 gradient값은 0에 수렴해서 학습 X

 초기화를 너무 작게/크게 안 됨

적당한 크기가 뭘까?

### Xavier Initialization

- 실험적으로 input값의 square root값을 취한 값을 쓰면 잘 됨
- For conv layers, d_in = F^2C
    - F : Filter size
    - C : number of channels

- 왜 동작하는가?
    - lecture note

### Weight Initialization for ReLU

We have actually told not to use tanh, use ReLU. Are we free from this initialization issue then?

x가 0과 가까운 값을 가질 때, gradient값도 0에 가까워져 학습이 안 된다!

### Kaiming/MSRA Initialization for ReLU

- 아까의 2배 : sqrt(2/d_in)
- 증명은 생략함. 이렇게 학습시키면 잘 될 것임

Initialization을 어떻게 할까?는 아직까지도 활발히 연구되고 있음.

Q : Data의 전체 평균을 빼서 zero centering

- 평균, 분산 계산 : 무조건 training data만 써야함
- test data를 기준으로 zero centering하면 안 됨!

Q : 이미지가 색이 크게 영향을 주지 않는 경우

- 흑백 이미지로 변환
    - RGB를 1/3로라던가, 1채널로 가중합으로 해서 흑백 그림
        - 그만큼 parameter값이 줄어들어서 더 robust하고 잘 되는 경우 있음

Q : PCA에서 variance가 가장 많이

차원 축소 과정에서 데이터 손실은 있을 수 밖에 없음

데이터 분포에 따라 뒷 부분에서는 분산이 거의 없는 경우가 있음

이렇게 하는 경우

- ex: 1000dim → 100dim으로
    - 데이터 차지 용량은 1/10, 한7~80%는 보존됨
    - computer power등의 기준으로 엔지니어링