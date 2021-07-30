# ResNet (Deep Residual Learning for Image Recognition)

이번에는 ResNet 논문을 정리하도록 하겠습니다. 이번 논문을 읽을 때는 최대한 제가 스스로 정리하는 쪽으로 구현해보겠습니다.

지금까지의 ILSVRC 대회 결과를 보면, **depth의 깊이**가 모델의 성능에 큰 영향을 준다는 것을 알 수 있다. 즉, depth는 매우 중요한 요소이다. 하지만, increase depth에 따라 필연적으로 **overfitting, vanishing gradient, 비용 증가** 등이 있다.

### Abstract

Neural Networks가 더 깊어질수록 훈련하기에 더 어려워진다. 이를 위해 이 논문에서는 a residual learning framework를 제안하는데 이는 이전에 사용하던 networks보다 대체로 더 깊은 네트워크를 훈련하기 쉽도록 만들어줍니다.  논문을 읽다보면 설명이 되지만, 한번도 참조한 적 없는 새로운 함수를 학습하는 것보다 input layer를 참조한 residual function을 학습하도록 layer를 재구성 했다고 한다.  이 residual networks는 optimize하기에 쉽고 network의 depth가 상당히 증가하더라도 정확도를 얻을 수 있다고 한다. ImageNet dataset에서 depth를 152 layer까지 사용하는데, 이는 VGGNet에서 사용한 depth보다 8배 깊은 depth이다. 하지만 이런 depth에도 불구하고 lower complexity를 가진다. 이런 residual net을 ensemble 해서 ImageNet test set error를 3.57% 로 줄였다.

### 1. Introduction

Deep Convolutional Neural Networks은 image classification의 돌파구와 같은 역할을 했다. Deep networks는 low / mid / high level의 특징을 classifier와 multi-layer 방식으로 통합한다. 특징들의 level은 쌓여진 layer의 수에 의해 풍부해질 수 있다.

Network depth는 엄청 중요해졌는데, 위와 같은 결과로 non-trivial visual recognition task들을 포함해 "very deep" model이 좋은 성능을 내는 것을 보여주었다.

Depth가 중요해지면서, 한가지 의문점을 낳게 되었는데, 바로 layer를 더 쌓으면 쌓을수록 network 학습이 더 쉬워지는가? 이다. 이 문제의 해답을 내놓기에는 vanishing / exploding gradient라는 악명높은 문제점이 장애물이었다. 그러나 이 문제는 SGD를 적용한 10개의 layer까지는 normalized initialization, intermediate normalization layer를 가지고 문제를 해결했다. 

> Network가 수렴하기 시작할 때 degradation 문제가 발생하기 시작한다. 
> 이는 network depth가 커지면서 정확도가 포화상태에 빠지다가 빠르게 감소하는 것을 의미한다.

<img src = "https://user-images.githubusercontent.com/78463348/127482813-985493c9-a326-44a4-ba1b-fd3889d75dad.png">

이 degradation은 overfitting에 의해 발생한 것이 아니며, 적절하게 deep한 model에 더 layer를 쌓는 것은 더 높은 training error를 만든다. 위의 Figure 1에서 이를 확인 할 수 있다.

Training accuracy의 degradation은 모든 system들이 최적화 되는 방식이 다르다는 것을 나타낸다. shallow architecture와 이 구조보다 layer를 더 추가한 깊은 architecture가 있다고 생각을 해보자.  더 deep한 모델 구조에 의해 solution이 존재한다. 

> 추가한 layer는 identity mapping이며 (입력을 그대로 전달하는 layer라고 생각하자) , 나머지 layer들은 미리 train한 shallow model을 그대로 가져오는 것이다.

즉 deep한 모델은 shallow한 모델 보다 training error가 더 높지 않게 생산해야한다는 것을 의미한다.

<img src = "https://user-images.githubusercontent.com/78463348/127482935-974a42e8-6c26-44f2-93c3-62d0524c61bd.png">

이 논문에서는 degradation 문제를 deep residual learning framework를 소개함으로써 다루고자 한다.

몇몇 쌓여진 layer들을 desired underlying mapping에 fit 하는 것 대신에, 우리는 이 layer들을 residual mapping에 fit 하기로 했다.  desired underlying  mapping을 H(x)라고 할 때, stacked nonlinear layer는 F(x) = H(x) - x 의 또 다른 mapping에 fit하기로 했다. 

우리는 original, 참조되지 않은  mapping을 optimize하는 것보다는 residual mapping을 optimize 하는 것이 더 쉽다고 가정한다. 극단적으로, 만약 identity mapping이 최적이라면, nonlinear layer들의 stack으로 identity mapping을 fit하는 것 보다 residual을 0으로 하는 것이 더 쉬울 것이다.

> 이 부분은 제가 생각하고 추측한 부분이라 틀릴 수도 있습니다.  -- original mapping이 x라고 하면 사실상 F(x)를 0으로 하면 H(x) =  x와 같아지는 것을 볼 수 있다. 수식상으로 같은데도 불구하고 극단적으로 말하면 function을 fit하는 전자의 경우보다 0으로 간단하게 fit하는 후자의 경우가 더 쉽게 fit할 수 있다고 생각한다.



Shortcut connections는 하나 또는 그 이상의 layer들을 skip하는 것을 말한다. 이 논문의 경우 Figure 2에서와 같이 간단하게 identity mapping을 수행했디. 그리고 이 output을 쌓여진 layer들의 output과 더해주었다. 

**Identity shortcut connections은 추가적인 parameter나 계산 복잡도를 추가할 필요가 없다.**

전체 network는 end to end 로 역전파와 함께 학습되며 기존 library들로 쉽게 나타낼 수 있다.

이 논문에서는 degradation 문제를 보여주고 자신들의 방식을 평가하기 위해 ImageNet을 사용한 실험을 제시한다. 여기서 보여주려고 하는 점은 두 가지이다

> **1. Our extremely deep residual nets are easy to optimize, but the counterpart  "plain" nets (that simply stack layers) exhibit higher training error when the depth increases**
>
> **2.Our deep residual nets can easily enjoy accuracy gains from greatly increased depth producing results substantially better than previous networks**

이후 내용은 ImageNet classification에서 152 layer residual net을 사용, ensemble을 적용해 3.57%의 top-5 error가 나왔으며 ILSVRC 2015 classification에서 1위를 차지하는등의 쾌거를 이뤄냈다는 이야기들이다.



### 2. Related Work

#### Residual Representations

요약하자면 **벡터 양자화에 있어 residual vector를 encoding하는 것이 original vector보다 훨씬 효과적**이라는 것이다. 

> 이 때, 벡터 양자화는 **특정 벡터 X를 클래스 벡터 Y로 mapping**하는 것을 말한다.

#### Shortcut connections

ResNet의 Shortcut connection은 다른 방식들과는 달리, **parameter가 전혀 추가되지 않으며, 0으로 수렴하지 않기에 절대 닫힐 일이 없어 항상 모든 정보가 통과된다. 따라서 지속적으로 residual function을 학습하는 것이 가능**하다고 한다.



### 3. Deep Residual Learning

#### 3.1 Residual Learning

이 부분은 1.introduction에서 residual learning에 대해 이야기 했던 부분을 다시 한번 설명하는 부분입니다. 몇몇 Stacked layer들에 의해 fit 되어지는 underlying mapping을 우리는 H(x)라 한다고 했다. (물론 전체 네트워크일 필요가 없다). 여기서 x는 이 layer들의 첫번째 input을 가리킨다. 만약 multiple nonlinear layer들은  점점 복잡한 함수에 근사화 할 수 있고, 이는 점근적으로 residual function에 근사화 할 수 있다고 가정했다 (F(x) = H(x) - x). H(x)나 F(x) - x 두 형태 모두 점근적으로 desired function에 근사화 할 수 있어야 하지만, 학습의 쉬운 정도는 다르다.

Degradation 문제는 solver가 multiple nonlinear layer들로 identity mapping을 근사화 하는 것에 어려움을 겪고 있다는 것을 암시한다. 만약 identity mapping이 최적이라면, solver는 identity mapping에 근사화 하기위해 multiple nonlinear layer의 가중치를 0으로 다가가게 만든다.

현실에서는 identity mapping이 최적인 경우가 별로 없지만, H(x) -> F(x) - x 로 바꾸는 reformation은 문제를 preconditioning 하도록 도왔다. 만약 optimal function이 preconditioning에 의해 zero mapping보다 identity mapping에 가까우면 solver가 identity mapping을 참조하여 학습하는 것이, 새로운 function을 학습하는 것보다 쉽다.

#### 3.2 Identity Mapping by Shortcuts

<img src = "https://user-images.githubusercontent.com/78463348/127521965-1e2eef56-301e-4a26-84c9-de44d82379fe.png">

> x와 y는 각각 few stacked layer들의 입력과 출력 벡터이다. <img src = "https://user-images.githubusercontent.com/78463348/127522348-17506358-7929-40c3-935e-f28c0ae46860.png">는 학습된 residual mapping을 나타낸다. <img src = "https://user-images.githubusercontent.com/78463348/127522526-95f99037-fee9-48ed-9532-0ee2710f3802.png"> 는 Figure 2에 그려진 residual block을 수식화 한 것인데, 여기서 'sigma'는 ReLU 함수이며 bias는 간단하게 만들기 위해 생략, F + x는 short connection과 element-wise addition에 의해 수행됩니다.

1.introduction에서 소개했듯이, **short connection은 parameter나 계산 복잡성을 추가하지 않는다.** 

<img src = "https://user-images.githubusercontent.com/78463348/127523971-0f5abeec-8a60-43a3-a542-fa6a34e44ff9.png" width = 50% height = 50%>

또한 F(x)와 x의 차원이 서로 같아야 하는데 만약 같지 않다면 (input/output channel이 바뀌는 경우), linear projection인 Ws를 곱해서 차원을 같게 만들 수 있다. Ws는 오직 차원을 맞추는 데에만 사용한다. 

residual network인 F 는 유연한 형태를 가진다. 이 논문에서의 실험은 F가 2~3개의 layer를 포함하며, 그보다 더 많은 layer를 포함해도 된다. 그러나 만약 F가 single layer일 경우, 1번째 식은 linear layer인 y = W1x+x 와 비슷해진다. 그리고 비록 위 표현식은 간단함을 위해 fully-connected layer에 적용했지만, 이를 multiple convolution layer에 적용해도 된다.

#### 3.3 Network Architectures

<img src = "https://user-images.githubusercontent.com/78463348/127525133-09104144-55a1-4085-87d1-601b8d1cacd7.png">

<img src = "https://user-images.githubusercontent.com/78463348/127525255-1035cb58-dd57-44d8-8143-d47b213ecf00.png">

##### **Plain Network**

Plain baseline은 Figure 3의 가운데 모델을 의미하며, 이는 VGGNet에 의해 영감을 받았습니다. 

Convolution layer들은 주로 3x3 filter를 가지고 있으며, 두 가지 simple design rule을 따릅니다.

1. 같은 output feature map 크기를 위해, layer는 filter의 수와 같은 수로 맞춘다.
2. 만약 output feature map 크기가 절반이라면, layer당 시간 복잡도를 보존하기 위해 filter의 수를 두 배로 한다.

Stride가 2인 convolution layer에 의해 downsampling 을 수행하며 network 마지막에는 global average pooling layer와 softmax 1000 way fully-connected layer를 쓴다. (원래 pooling을 사용해서 downsampling을 하지만, 여기서는 stride가 2인 convolution layer를 사용한다)

이 모델은 VGGNet보다 더 적은 복잡도와 더 적은 filter를 가지는 것에 주목할 가치가 있다. 

##### Residual Network

Plain network를 기반으로해서 shortcut connection을 추가했다. 

Input과 output의 차원이 같다면, identity shortcuts을 사용하고, 차원이 증가했을 경우, 두 가지의 option이 있다.

A) shortcut이 identity mapping을 수행함과 동시에, 차원을 키우기 위해 zero padding을 적용한다.

> 이 방식은 추가적인 parameter를 사용하지 않는다

B)  차원을 맞추기 위해 위에서 이야기 한 Projection shortcut을 사용한다. (done by 1x1 convolution)

두 option 모두 shortcut이 feature map을 2 size씩 건너 뛰기 때문에, stride를 2로 수행한다.



#### 3.4 Implementation

모델 구현

1. ImageNet을 input으로 하기 때문에, image는 짧은 쪽이 [256,480] 사이가 되도록 random하게 resize한다. 

2. 224x224 크기로 이미지를 random crop 또는 horizontal filp을 사용 & per-pixel mean substracted
3. Standard color augmentation 사용
4. 각 convolution 후 그리고 activation 전에 Batch normalization 적용
5. He initialization 적용으로, 가중치 초기화
6. SGD를 사용하며 mini-batch는 256이다.
7. 초기 learning rate는 0.1이며 학습이 정체될 때마다 1/10을 곱해준다.
8. Iteration  = 60x10^4
9. weight decay = 0.0001
10. Momentum = 0.9
11. Dropout은 사용하지 않는다.

Test시에는 10-cross validation을 사용하며 최고의 결과를 내기 위해, fully-convolutional form을 적용하며, multiple scale을 사용해 더 짧은 쪽이 {224,256,384,480,640}중 하나가 되도록 resize한 후, 평균 score로 산출



### 4. Experiments

#### 4.1 ImageNet Classification

ImageNet 2012 classification dataset으로 평가한다. 

> 1000개의 class / 1.28 million train image / 50k valid image / 100k test image

##### Plain Networks

18-layer와 34-layer를 이용해서 성능을 평가하는데, deep한 모델인 34-layer plain net이 얇은 18-layer plain net보다 높은 validation error를 가진다. 또한 training error도 높다는 것을 알 수 있다.

<img src = "https://user-images.githubusercontent.com/78463348/127591012-6589de81-1087-4092-ab55-3b02cbe08f1d.png">

<img src = "https://user-images.githubusercontent.com/78463348/127591430-ccb970bc-dd0b-4df2-a361-97dc0ffe8a1b.png">

Figure 4의 왼쪽 그림을 보면, 우리는 degradation 문제를 발견할 수 있는데, 34-layer plain net의 training/validation error를 둘 다 비교해본 결과 Table 2의 validation error만 높은 것이 아닌 training error도 높은 것으로 보아 degradation 문제로 판단했다고 한다.

이런 최적화의 어려움은 vanishing gradient에 의해 발생하는 것이 아니라고 말했는데, 왜냐하면 plain network는 Batch Normalization으로 train 했으며, 순전파 신호는 variancec가 0이 아니었으며, 역전파 기울기 또한 healthy norm을 보여주었기 때문이다. 그래서 순전파 신호와 역전파 신호 모두 사라지지 않았고, 34-layer plain net은 여전히 경쟁력 있는 정확도를 얻었다. 이로 미루어 봤을 때, deep plain net이 exponentially low convergence rate를 가지고 있고 이것이 training error의 감소에 악영향을 주었을 것으로 짐작했다.

##### Residual Networks

plain때와 마찬가지로 18-layer와 34-layer ResNet을 평가할 건데, plain net baseline architecture에서 shortcut connection 을 3x3 filter들로  추가한 것이다.

모든 shortcut에서 identity mapping을 사용하며, 차원 증가를 위해 zero-padding을 사용했따 그래서 plain과 비교했을 때, parameter의 증가가 없다. 

 Table 2와 Figure 4로 부터 3가지의 발견을 할 수 있었다.

> 1. Plain net와 상황이 반대인데, 34-layer ResNet이 18-layer ResNet보다 2.8%로 더 좋은 성능을 낸다. 더 중요한 점은 34-layer ResNet은 상당히 낮은 training error를 보였으며, validation data에 일반화 할 수 있다는 점이다. 이는 degradation문제가 이런 setting에서 해결할 수 있으며, 우리는 depth가 증가하더라도 좋은 정확도를 얻을 수 있다는 점을 의미한다.
> 2. Plain net과 비교했을 때, Table 2에서 알 수 있듯이 ResNet은 top-1 error가 3.5%정도 줄었다. 이는 residual learning이 extremely deep system에서 효율적이라는 것을 증명했다.
> 3. 마지막으로 18-layer ResNet와 plain net 둘다 accurate가 같으나, ResNet쪽이 수렴을 더 빨리 한다는 장점을 가진다. 즉 모델이 deep하지 않은 경우 , 현재 SGD solver는 여전히 plain net에서 좋은 solution을 얻을 수 있으나, ResNet이 초기에 수렴을 더 빨리 함으로써 최적화가 더 쉽다.



##### Identity vs Projection Shortcuts

<img src = "https://user-images.githubusercontent.com/78463348/127598827-2b6143a0-c5be-42f4-976f-45e48474a9ad.png">

Parameter-free와 identity shortcut은 training에 도움이 된다는 것을 보여주는데, 다음으로는 Projection shortcut에 대해 조사했다. Table 3에서는 3가지 option에 대해 비교했다고 한다

A) 차원 감소를 위해 zero-padding shortcut을 사용한 경우. 그리고 모든 shortcut은 parameter-free이다.

B) 차원 감소를 위해 Projection shortcuts을 사용한 경우. 그리고 나머지 shortcut은 identity함.

C) 모든 shortcut이 Projection shortcut을 사용한 경우.

3가지 option 모두 plain에 비해 상당히 좋은 성능을 보여준다. B가 A보다 약간 더 좋다. 왜냐하면 zero-padded 차원은 사실상 no residual learning이기 때문이다. C는 B보다 약간 더 좋다. 많은 projection shortcut에 의해 parameter가 추가되었기 때문이다.

A,B,C 모두 성능 차이가 그렇게 크지 않기에, 이는 projection shortcut이 degradation 문제를 해결하는데 있어서 필수적이지 않다는 것을 알 수 있다. 그래서 memory / time complexity와 model 크기를 줄이기 위해 이후 논문에서는 option C를 사용하지 않는다.

Identity shortcut은 특히 아래에서 설명할 bottleneck 구조의 complexity가 증가하지 않게 하기 위해 더 중요하다.

##### Deeper Bottleneck Architectures.

<img src = "https://user-images.githubusercontent.com/78463348/127597727-03474e9c-ee67-410e-bcd8-1da349a88d59.png">

ImageNet에 대해서 training time이 길어질 것을 우려해 building block을 bottleneck design으로 수정했다. 각 residual function F에서, 2-layer 대신 3-layer를 쌓는 구조를 사용했다. 3 layer는 1x1,3x3,1x1 convolution으로 구성되며, 1x1 layer는 차원을 줄이거나 늘리기 위해 사용되며 3x3 layer의 input/output 차원을 줄인 bottleneck 구조를 만들어준다. 

bottleneck 구조에서 중요한점은 parameter-free identity shortcut이다. 만약 identity shortcut이 projection shortcut으로 대체되면, shortcut이 2개의 high-dimensional과 연결되어있기 때문에 시간 복잡도와 model 크기가 2배로 늘어나는 것을 보여줄 수 있다.그래서 identity shortcut이 bottleneck design을 더 효율적인 모델로 만들어 준다

##### 50-layer ResNet

34-layer의 2-layer를 3-layer bottleneck block으로 대체해서 총 layer가 50-layer ResNet을 만들었다. 차원 증가를 위해 option B를 사용한다.

##### 101-layer and 152-layer ResNets

더 많은 3-layer block을 사용함으로써 101, 152-layer ResNet을 구성한다. depth가 상당히 증가했음에도 불구하고 152-layer ResNet은 여전히 VGG-16/19 net보다 lower complexity를 가진다.degradation 문제를 발견하지 않아서 깊은 모델로 부터 상당히 높은 정확도를 얻을 수 있었다.

##### Comparisons with State-of-the-art Methods

<img src = "https://user-images.githubusercontent.com/78463348/127598784-d9f5eed8-b9bc-4359-a090-a8db82d9d185.png">

152-layer ResNet은 single 모델로서 top-5 validation error로 4.49%를 가지는데, 이 single model은 ensemble이 적용되기 전의 다른 모든 모델의 성능보다 높았으며, ensemble을 적용한 경우 test set에 대해 3.57%의 top-5 error를 갖는다.



#### 4.2 CIFAR-10 and Analysis

CIFAR 10으로 조금 더 연구를 수행했는데, training image는 50k, testing image는 10k개 이며, 10개의 class를 가진다. 우리가  state-of-the-art result보다는 extremely deep network의 영향에 대해 초점을 맞췄기 때문에, 의도적으로 간단한 구조를 사용한다.

모델구현의 경우,ImageNet과 비슷하며, 논문 page 7에 설명되어있다. (생략하겠습니다.)

##### Analysis of Layer Responses

<img src = "https://user-images.githubusercontent.com/78463348/127599844-84adaff9-5040-4278-ae0d-c0b1333e97b9.png">

ResNet은 plain에 비해 small response를 가지는데, 이는 residual function이 non-residual function보다 일반적으로 0에 가까울 것이라고 하는 주장을 지지해준다. (depth가 깊어짐에 따라 response가 작아지는 것은 각 layer가 학습 시 signal이 변화하는 정도가 작다는 것을 의미한다.)

##### Exploring over 1000 layers

<img src = "https://user-images.githubusercontent.com/78463348/127599996-26ced7fe-9399-4027-94c3-8ffe0e3b48e9.png">

1000개 이상의 layer가 사용된 모델(1202-layer network)은 110-layer ResNet과 training error가 비슷했지만, test 결과는 좋지 못했는데, small dataset에 대해 불필요하게 network가 크기 때문이며, 이는 overfitting 때문이라고 판단된다.

참고자료

> 1. https://phil-baek.tistory.com/entry/ResNet-Deep-Residual-Learning-for-Image-Recognition-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0?category=967235 (백광록님의 논문리뷰) - 주로 related work part에서 주로 도움을 받았습니다!
> 2. [arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385) (논문)

