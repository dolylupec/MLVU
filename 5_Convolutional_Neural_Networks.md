# Lecture 5. Convolutional Neural Networks

지금까지는 fully connected layer만 배움

- input x
- W

Fully Connected Linear Classifier

- input dimension개수만큼 있고, 그 갯수ㅁ와 같은 weight 곱해짐
- 두 벡터를 내적(inner product)
    - 일반적으로는 세로로 긴 벡터, 앞에걸 transpose해서 가로 모양으로해서, 결과는 1x1 인 score ⇒ 1개의 classifier
    - 이런 classifier가 여러개 (pg6) : 총 c개의 output이 나옴

다르게 봐보자.

input으로 들어온 모든 값들이 output에 영향을 주기

- image : output value가 모든 pixel값에 영향을 받는다
- linear classifier : featurization을 하지 않는 이상 모든 pixel이 결과에 영향을 줌

이미지에 좀 더 맞는 방식으로 다뤄보자

전통적으로 연구해온 방식

이미지에서 특정 패턴을 찾고 싶다 (ex: 눈을 찾고 싶다)

- 눈의 특성을 반영한 filter를 만든다.→패턴이 있는지 쭉 확인 (high score면 있는 것임)
    - 눈이 어떤 크기로 어디에 있는지 알 수 없음(size와 위치 일일히 다 찾아봐야함 : 일일히 스코어 확인  )

피처들을 데이터로부터 어떻게 학습할 것이냐?

- 우선, 피처가 배워져 있다고 가정해서 적용하는 코드 구현해보자

implementation

- image size : (n,n)
- filter size : (k,k)
- 흑백 그림을 가정
- filter를 쭉 스캔하면서 각각 score 계산( score : image 값과 필터값의 내적)
- 마지막에 threshold를 적용해서 어느정도 이상이면 filter에 해당하는게 있다고 output내주기(threshold가 주어져 있다고 가정하겠다)
- 4개의 루프 필요
    - 이미지의 모든 자리들
    - filter가 k by k : 각 값들 계산

### Convolutional Layer

Monotone image:

- 32x32 image로 가정
- 가장 검은색 :0, 흰색:255인 정수
- 어떤 패턴을 표현하는 필터를 쭉 돌리면서 이미지 속에서 특정 패턴을 찾는 것 : Convolution
- Convolve the filter with the image i.e., slide over the image spatially(공간적), computing dot products(곱해서 더함:내적)
- 아까는 주어진 필터. **이제는 filter를 학습해보는 것을 할 것임**
- slide over : filter를 쭉 옮겨가면서 → score 여러개 나옴

color image

- 32x32x3(channel=3)
    - r :0~255인 integer
    - g, b도 마찬가지
- filter : 3x3x3 filter
- dot product해서 합칠 것임
- 방법 : 9개의 곱이 3개가 나올 것임 (3개 독립적으로 구하고), 이걸 다 더함 → 1개의 값으로 나옴
- 여기에 bias를 넣어주어야 함(data independent하게 모델링 되게 하는 숫자)
- 총 28개의 숫자가 더해서 score 1개 나옴 : 모든 자리에서 이렇게
- 32x32를 3x3 필터로 쭉 훑으면 첫번째 줄에서 30개 나옴
- 전체를 쭉 하면 30x30이 나올 것임
- Q : For 32x32x3 image and 5x5x3 filter, what is the size of output **activation map?**
    - A: 28x28x1
    
- 이러한 필터를 여러개 만들기
- Q :  For 32x32x3 image and with 4 5x5x3 filters. what is the size of output activation map?
    - A : 28x28x4
        - 28x28 : image size
        - 4 : number of channel

- 28x28x4인 새로운 image (일종의 tensor)
- 이걸 또 convolution을 돌릴 수 있다
- Q : For 28x28x4 image and 10 5x5x4 filters, what is the size of output activation map?
    - A : 24x24x10
    - 우리가 지정한 filter의 개수만큼 output의 channel 갯수가 됨

이걸 왜 하는 것인가? 어떤 의미인가?

### Nested Conv-layers

- image로부터 feature를 한번 학습
- feature : 우리가 만든 filter image map

- high-level feature : class 구별 시 가장 핵심이 되는 패턴을 학습한다 (class와 직접적인 관계)
    - Given a fixed capacity(number of filters), high-level features learn a set of features that are most useful to distinguish different classes in the dataset.
- mid-level features
    - Given a fixed capacity(number of filters), mid-level features learn a set of features that are most useful to distinguish high-level features.
- low-level features : mid level features를 표현하기 위한 component들
    - Given a fixed capacity(number of filters), low-level features learn a set of features that are most useful to distinguish mid-level features.
- class를 구분할 수 있는 특징들을 학습시킴
    - Deep Learning : Learning Representation (표현 학습)
    

## Convolutional Layer : A Deeper Look

Two questions:

1. If the filter is larger, activation map will get smaller quicker. This may prevent us from nesting many layers. Can we avoid this?
2. If the input image is in high resolution (e.g., 4k resolution = 3840x2160), conv layers require too much computation. Any better way to deal with large image?

### Stride

ex) 7x7 input image, 3x3 filter

- 5x5 output activation map

7x7 input image, 3x3 filter, **with stride 2 : 두칸씩 뛰면서 sliding**

- 위/옆 2칸씩 띄어서 이동
- 3x3 output activation map

7x7 input image, 3x3 filter, **with stride 2** 

- doesn't fit! : cannot apply 3x3 filter on 7x7 input with stride 3.
- stride가 되는지 안 되는지 매번 고민해야되는 것인가?

- **Output size :  (N-F)/stride + 1**
    - 정수값이 나오면 되는 것임
    - 정수가 아닌 값이 나오면 안 되는 것임
    - 해결 : padding

### Padding

- **Zero Padding** : 가로/세로에 1개씩 0을 넣음 (In practice, it is common to zero pad the border.)
- 7x7 input image, 3x3 filter, 1 stride
    - 7x7 output activation map
- input과 output의 size가 같아진다. → size가 줄어드는 문제 해결 가능

- **Output size : (N-F+2P)/stride + 1**
    - **With P = (F-1)/2, the map size will be preserved**
        - F=3 → P=1
        - F=5 → P=2
        - F=7 → P=3

- padding이 없으면 이전 예시에서는 4씩 줄어들었음
    - 32x32x3 → 28x28x4 → 24x24x10
- Given an input volume of 32x32x3, apply 10 5x5 filters with stride 1, pad 2.
    - **What is the output size?**
        - **A : 32x32x10** → padding 때문에 input size와 같은 size, filter channel이 10개
    - **Number of parameters?**
        - Parameter
            - linear model에서의 parameter:학습을 하는 **Weight** (data로부터 채워짐)
            - fully connected layer에서의 parameter : **input size x output size** (input과 output 관계를 모두 modeling해야함)
            - convolutional layer에서의 parameter : **filter에 있는 숫자의 개수** (그것만 학습하면 됨, input은 주어짐)
        - **A :  10x(5x5x3 + 1) = 760**
    - **cf) Number of parameters if fully-connected?**
        - **A : 32x32x10x32x32x3 = 31457280** (inputsize x outputsize)

### Why conv?

- **It dramatically reduces the number of parameters, by assuming spatial locality and positional invariance.**
    - positional invariance : filter가 어느 위치에 있어도 같은 weight(parameter)을 쓴다.
        - 강아지가 이미지의 어느 위치에든 있을 수 있다.
    - spatial locality
        - 강아지의 코 양쪽에는 눈이 있다.
    - local 외의 것들을 다 0으로, 가까운 값만 학습함. 어느 위치에 있어도 같은 값으로 강제 - > 760개로 연산 줄어듦 (image에서는 좋은 가정)

- Does it make sense for all image tasks?
    - 이런 가정이 맞지 않는 이미지 예시
        - positional invariance : x-ray 영상은 항상 같은 위치에서 사진을 찍음, 폐는 이미 특정 위치에 있음
            - 굳이 전체를 conv를 돌릴 필요 없음
        - 증명사진에서 눈코입 : 이미 고정된 자리에 패턴이 있음
    - 이런것들 말고 대부분 성립. 따라서  **spatial locality and positional invariance을** 적용해서 모델 학습함.
    

### Padding Example 2

Q : Given an input volume of 32x32x3, appy 6 1x1 filters with stride 1, pad 0.

- What is the output size?
    - A : 32x32x6
- Number of parameters?
    - A : 6x(1x1x3 + 1)

- **Why 1x1 conv?**
    - A : Each filter performs 3-dimensional dot-products, mixing information across the channel, on the same pixel.  (한 pixel에서 모든 channel - r,g,b 등의 info를 다 섞어서 반영할 수 있음)

### Fully-connected vs Conv

- Convolutional layer is a **special case** of fully connected layer.
    - 반대로도 가능. feature size를 크게 잡아주면 됨 (note에 있음)
- Each value in the output is determined by
    - All input values with fully-connected layer
    - Values within a small region with convolutional layer
- Thus, a conv-layer is equivalent to a fully-connected layer where all other weights(outside of the filter range) are zeros.

### Convolutional Layer Summary

Given an input volume of WxHxC, a convolutional layer needs 4 hyperparameters:

- number of filters K
- the filter size F - 보통 정사각형 모양
- the stride S - 보통 위아래/좌우 stride 같게
- the zero padding P - 보통 위아래/좌우 같게

This wil produce an output of size W'xH'xK, where

- W' = (W-F+2P)/S + 1
- H' = (H-F+2P)/S + 1

Number of parameters : K(F^2C + 1)

### TensorFlow API

대표적 hyperparameter

- filters : number of filters(K)
- kernel_size : filter size(F)
- strides : stride(height-bound, width-bound) (S)
- padding : padding (P)
    - padding = 'valid' : no padding
    - padding = 'same' : make the output same as input
- input_shape = (4, 28, 28, 3)
    - 4 : 4 images
    - each image : 28x28
    - 3 : number of channels
- y = tf.keras.layers.Conv2D(2, 3, **padding='same',** activation='relu', input~~
    - 2 : number of filters
    - 3: kernel_size (3x3)
    
    note 꼭 읽어보기!
    

(다음 동영상)

### Pooling Layer

- downsampling하는 것
- 크기만 줄여주는 operator
- 하는 이유 : pixel level에서는 noise 많이 들어감
    - sensitive하게 학습하는 것을 방지
    - ex)4개 평균 : 1개가 outlier여도 나머지3개가 보정
    - overfitting 방지
    - feature map 크기 줄이면서 model을 더 작고 메모리 상에서도 문제 없게

구체적으로 pooling하는 방법

- 2x2 filter , stride 2
- 해상도가 그만큼 절반으로 나빠짐
- max pooling : 조금이라도 지나가는 곳은 다 검은색으로 칠함
- average pooling : 해상도가 나쁜 이미지를 확대한 느낌

filter F, stride S가 있을 때

- F = 2, S = 1 : 3x3
- F = 2, S = 2 : 2x2
- F=3, S=1 : 2x2인데 위에것과 다른 계산

### Pooling Layer Summary

Given an input volume of **WxHxC**, a pooling layer needs **2 hyperparameters**:

- **The spatial extent F**
- **The stride S**

This will produce an output of size **W'xH'xK**, where

- **W' = (W-F)/S + 1**
- **H' = (H-F)/S + 1**

Number of parameters in pooling layer: **0** 

- no "learning" happens
- 데이터로부터 어떤 값을 찾아내려는 것이 아니라 정해진 식을 가지고 계산

### Pooling Layer : Recent Trends

Q. Is the pooling layer necessary? Can't we just use conv layers only, but using larger stride to control #parameters?

- A. Yes, we can. Recent research supports this idea actually
- pooling layer가 점점 없어지는 추세임. pooling layer보다는 conv만을 사용해서 (특히 GAN에서)