# GoogLeNet 정리

이번에는 GoogLeNet 논문 정리를 해보겠습니다. 코드 구현은 따로 github에 올려두었으니 확인해주시면 감사하겠습니다. 

이번 논문 정리를 하기에 앞서 논문 해석 글 사용을 허락해주신 Aroddary님께 다시한번 감사의 말씀을 드립니다.

> 출처 : https://sike6054.github.io/blog/paper/second-post/



ILSVRC 2014에서  VGGNet보다 성능이 더 좋으며, 1등을 한 GoogLeNet을 살펴보려한다.

### Abstract

이 모델의 주요 특징은 네트워크 안에서의 컴퓨팅 자원의 효율 향상이라는 점이다. 정교한 설계 때문에 이러한 특징을 가지게 되었는데, 이는 네트워크의 depth와 width를 늘려도 연산량은 증가하지 않고 유지된다는 것을 의미했다. 성능의 최적화를 위해서 Hebbian principle과 multi-scale processing을 기반으로 구조를 결정했으며, 이를 GoogLeNet이라고 부른다. GoogLeNet은 22개의 layer를 가지고 있으며, codename은 Inception이다.

> *여기서 말하는 Hebbian principle은* **“neurons that fire together, wire together”***을 말한다. 즉, 동시에 활성화 된 노드들 간에는 연관성이 있다는 의미 정도로 생각할 수 있다.*



### 1. Introduction

지난 3년간 주로 딥러닝, 구체적으로는 convolution networks의 발전때문에, 이미지 인식, 물체 감지 등의 quality는 급격한 속도로 발전했다. 이러한 발전은 단지 더 좋은 하드웨어의 성능, 더 큰 dataset, 더 큰 모델 때문이라기 보다는 새로운 아이디어와 알고리즘, 그리고 개선된 신경망 구조 때문이었다. GoogLeNet은 ILSVRC 2014 대회 제출에서 2년 전 AlexNet보다 12배나 적은 parameter를 가지고 있었다. 이러한 개선은 R-CNN에서와 같이 deep한 구조와 classic한 computer vision의 시너지 덕분에 가능했다.

또 다른 주목할만한 요소는 mobile 그리고 embedded 환경에서 지속적으로 동작하기 위해서는 알고리즘의 효율성이 중요했는데, 특히 전력과 메모리 사용 측면에서 효율적인 알고리즘이 중요성을 가진다. 이번 논문에서는 정확도의 수치보다 효율성을 더 고려하여 deep architecture를 설계했으며, 또한 추론 시에, 모델은 1.5billion 이하의 연산량을 유지하도록 설계되었다. 

> 제안하는 구조가 성능 향상의 목적을 두지 않았기에 학문적인 흥미는 떨어질 수 있으나, large dataset을 갖는 real world에서도 합리적인 비용으로 이 구조를 사용할 수 있을 것이라는 점에 중점을 뒀다.

이 논문에서, 우리는 효율적인 computer vision DNN 구조에 초점을 두었으며 codename을 Inception이라고 정했다. 왜 codename이 Inception인 유래는  Network in network 논문에서 왔다고 한다. 더 정확하게 이야기하면, "we need to go deeper"라는 유명한 인터넷 밈에서 고안했다고 한다. 

이 때 "deep"이라는 의미는 두 가지 다른 뜻을 가지고 있는데, 아래와 같다

> 1.  "Inception module"의 형태로 새로운 차원의 구조 도입
> 2.  네트워크의 depth 증가 

일반적으로, Arora 등의 이론적 연구에서 영감을 얻으면서 Inception model을 논리적 정점으로 볼 수 있다. 

> 일반적으로 Inception 모델은 Network in network의 논리로부터 영감을 받았으며, Arora의 이론적 연구가 지침이 된 것으로 볼 수 있다.



### 2. Related work

LeNet-5를 시작으로, CNN은 standard structure를 가지게 되었다. (이 구조는 convolution layer의 쌓여진 구조 + 하나 이상의 fully-connected layer로 이루어져 있다.) 이 기본 구조의 변형은 이미지 분류에서 널리 퍼졌으며, MNIST,CIFAR,ImageNet classification challenge에서 state-of-the-art 성능을 얻었다.

ImageNet과 같은 큰 dataset의 경우 최근 layer와 layer 크기를 늘리면서 dropout을 사용해 overfitting문제를 다루는 것이 트렌드였다. 비록 max-pooling layer의 결과로 정확한 공간적 정보를 잃어버린다는 걱정에도 불구하고 AlexNet은 localizaion과 object detection 및 human pose estimation 분야에 성공적으로 적용했다.

영장류의 시각 피질에 대한 신경 과학 모델에서 영감을 받은 Serre의 연구에서는 multiple scale을 다루기 위해 크기가 다른 fixed Gabor filter를 사용했다. 이와 비슷한 전략을 Inception에서 사용하나, Inception의 모든 filter가 학습한다는 점에서 차이가 있다. 또한 GoogLeNet의 경우 Inception layer가 여러번 반복되어 22-layer deep model로 구현된다.

Network-in-Network는 Lin에 의해 제안된 접근법으로, 신경망의 표현력을 높이기 위해 제안되었다. convolution layer에 적용했을 때, 이 방식은 1x1 convolution layer가 추가되고 이후 ReLU 함수가 추가되는 것 처럼 보인다. 이 접근방식을 이 논문에서 말한 구조에 많이 사용한다. 

그러나 1x1 convolution은 두 가지 목적을 가진다.

> 1. 병목 현상을 제거하기 위해 차원 축소 모듈 (dimension reduction module) 로써 사용
> 2.  depth를 늘리는 것과 동시에 성능 하락 없이 network의 width를 늘리기 위해서 사용
>
> 이를 이용하지 않으면 네트워크의 크기가 제한 될 수 있기 때문에 주로 1번 목적을 중요하게 여긴다



### 3. Motivation and High Level Considerations

Deep neural network의 성능을 향상시키는 가장 직접적인 방식은 신경망의 크기를 늘리는 것이다. 이는 depth를 늘리는 것과 width를 늘리는 것 둘 다를 말한다.

> depth -  increase the number of levels 
>
> width - increase the number of units at each level (각 level의 node 개수)

이는 특히 label된 training data가 많은 경우에 고성능 모델을 훈련시키는 안전하고 쉬운 방식이다. 그러나 이 간단한 해결책은 두 가지 단점을 가지게 되었다.

> 1. 사이즈가 커진다는 것은 parameter의 수가 많아진다는 것을 의미한다. 특히 label된 훈련 데이터의 수가 제한될 경우 Overfitting이 일어나기 쉬운 경향을 띤다.
>
>    이는 ImageNet과 같이 다양한 클래스로 세분화 된 dataset을 다루는 경우에 생기는 주요 병목현상이다. Fig 1과 같이 비슷한 생김새일 경우 사람이 사진만으로 visual category를 분류하는 것은 전문가일지라도 어려워 보인다.
>
> 2.  네트워크의 크기가 증가함에 따라, computational resource가 극도로 증가한다는 점이다
>
>    예를 들어 convolution layer가 filter의 수가 증가함에 따라 computation이 quadratic하게 증가하는 것을 볼 수 있다. 추가된 filter 대부분의 weight가 0으로 가까워지는 등 비효율적으로 사용된다면, computational resource의 낭비를 겪게 된다. computational resource는 항상 한정되어 있기에, 무차별적인 크기 증가 보다는 computing resource를 효율적으로 분배하는 것이 선호된다.

위와 같은 문제들을 해결하는 근본적인 방식은 convoltion 내부를 fully connected architecture를 sparsely connected architecture로 바꾸는 것이다.  











































































> 1. https://arxiv.org/abs/1409.4842
