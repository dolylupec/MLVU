# Lecture 1. Introduction to Computer Vision

인간에게는 직관 능력 있음: 딱 보면 '피겨스케이팅'이라고 알아봄
컴퓨터 : image들이 여러장(sequence) 이어져있음 

- video : image의 sequence
- image :  가로/세로로 2차원 pixel matrix. 각각의 값은 3~4 개의 인코딩된 숫자 : 숫자들 나열되어있는 것
- input : 컴퓨터가 이해하는 비디오, output : 사람이 이해하는 비디오
    - 이 둘을 연결시켜주는 수학적인 계산 모델 만들기 : Video Understanding의 핵심  


ex: 아이들이 놀고 있는 공원에서 범죄자들을 쫓는 경찰들
Video Understanding : 이해하기 위해 여러단계의 추론을 해야되어서 어려움
- Describing the content
- inferring the central topics
- describing the structure and style


### Video Understanding
- creator : 무슨 정보를 전달하고 싶어서 이를 전달하는 것일까?
- viewer : 왜 볼까? 얻고 싶은 정보가 뭘까? 뭐가 재밌을까?


### What is Computer Vision?
- digital images or videos : 컴퓨터가 비디오를 저장하는 방식 (픽셀들의 나열)
- high level understanding : 인간이 이해하는 방식


### Early works in CV
- 다른 각도에서 보면 어떻게 입체 도형이 보이는지
- tracking :  모션을 감지해서 변화를 저장/인식
- object recognition :  CV가 크게 발전하게 된 task
    - object classification : 그림 안에 뭐가 있는지
    - generic object detection (bounding box)
    - semantic segmentation : pixel 단위로 모서리 찾아내기
    - object instance segmentation : 한 덩어리가 양인지, 따로따로 5마리 양인지 구별
- face detection and recognition : 얼굴 인식하고 누구인지 알아내는 것 (중요)
- 3d reconstruction : 2d image로 통해 3d 구조 구현, 유명한 건축물들은 온갖 각도에서 찍음, 이걸 다 모아서 3d reconstruction [Building Rome in a Day]

### History of AI
- Before Deep Learning Revolution
- 1943년 (컴퓨터의 역사 eniac보다도 더 오래됨) : idea 나옴 (하나의 node - 인간이 가중치 직접 정해줌)
- 1969 : XOR problem : 수학적으로 표현할 수 없다
- 제프리 힌튼 : nn 여러개 쌓으면 어떤 function이든 다 표현 가능하다 (수학적 증명)
    - backpropagation 을 통해 XOR 해결
- 컴퓨터한테만 잘 되는 통계기법 : 2000년대 초중반 (SVM - 뇌와 상관 없고 수학적 기법임)
- 최근 : bigdata, GPU - 대규모로 ML 모델 만들 수 있기 시작함 (예전의 아이디어 드디어 구현 가능)
- Deep Neural Network (제프리 힌튼)
- ImageNet : label된 사진, 이제는 엄청난 데이터셋 아님
- AlexNet : 처음으로 Deep Neural Network 사용, 에러 엄청 낮아짐
- 더 어려운 데이터셋에 도전 중임

### Video 쪽
- Youtube 8M
    - 토픽별로 수집해서 labeling함
    - 우리는 움직이는 영상 봄. 모션 포함 다양한 정보 담고 있음, 세상을 인식할 때 가장 가까운 형태의 데이터셋
    - 데이터셋 용량 너무 큼, labeling 일일히 함

### [Machine Learning for Visual Understanding]

- Not a traditional computer vision course
- not a fundamental ml course
- learn how to apply ML for various visual understanding problems

### Tasks and Applications
- Object Recognition (image classification)
    - 가장 기본. 물체 찾아내기 (1차),  구체적 도메인에 특징 물체/현상 감지
- Action Recognition (Video Classification)
    - 사람의 동작 찾아내기
    - 좀 더 세분화해서 ( hand jesture recognition )
- Spatial & Temporal Localization (시공간적 이해)
    - image : 2d, video : 3d(시공간)
    - spatial localization : 이 물체가 어디 위치에 있다
    - temporal localization :  긴 비디오에서 특정 장면이 어디에 있는지 감지
- Segmentation
    - pixel단위로 어떤 카테고리에 속하는지
    - semantic segmentation
    - instance segmentation : 다른 사람은 다른 사람인걸로 표시해주기
- Tracking : 움직이는 물체 쫓아가기
    - pedestrian tracking : 비디오 속 시간에 따라 이전 프레임과 비슷한 곳에 있는 사람 서로 매칭 → 쭉 이으면 따라가는 것처럼 tracking 가능
    - cell tracking : 앞에꺼 응용 (자연연구)
    - hurricane tracking : 이전에는 과학모델 사용, 머신러닝 사용해 적은 연산량으로 예측 가능
- Multimodal Learning
    - video : visual 정보 + audio정보 +  semantic 정보
    - 조합해서 어떤 일이 일어나는지 예측
    - audio-visual : 영상에서 이런 물체가 나오면 이런 소리가 나온다 (패턴 찾기) 요즘 Hot! 신기한 것들 많이 됨
    - text-visual
    - image captioning : 사람이 image labeling함, 나중에 컴퓨터는 처음보는 그림에 맞는 문장을 써줌 (수업에서 해볼 것임)
    - Visual Q&A : 문장 이해를 넘어서 질문에 대한 답 찾기 (요즘 연구 많음)
- Style Transfer
    - 컨텐츠 + style섞어서 새로운 것 만듦
    

### Video Search and Discovery
- video 는 너무 많음
- content : visual 정보 (pixel),  소리 정보
- meta data : video 밖에 있지만 정보 ( 제목, 감독,설명, 댓글 등)
- viewer signals : 누가 비디오 봤는지, 이 비디오 본 사람들은 다른 것은 뭘 봤는지(비디오 관계)

### Personal Media Collection
- Lots of videos, no labels, no metadata, no viewer signals(privacy 문제도 있음)
- 정보가 부족한 상황에서 학습해야함 (어려운 문제, potential 한 문제)
    - self supervised, multimodal 활용해서 성능 높일 수 있음(ing)

### Machine Learning
- kNN
- SVM
- Tree
- Ensemble
- GMM
- K-means
- PCA
- **SGD**
- **CNN**
- **RNN**
