# VGGNet (Very Deep Convolutional Networks for Large-Scale Image Recognition) 정리

VGGNet 정리는 Time Traveler님의 VGGNet 논문 리뷰를 참고해서 만들었음을 밝힙니다.

> ###### https://89douner.tistory.com/61 (여기 진짜 논문 맛집입니다!)

###### 해당 논문 출처 : https://arxiv.org/pdf/1409.1556.pdf

AlexNet의 등장 이후 image classification분야에서 CNN 모델이 주목을 받았습니다. 이후 ILSVRC 2014년 대회에 VGGNet과 GoogLeNet 모델이 등장하게 됩니다. 특히 이 두 모델은 AlexNet보다 더 깊은 layer를 쌓아서 뛰어난 성능을 자랑합니다. ILSVRC 대회에서는 GoogLeNet보다 이미지 분류 성능은 낮았지만, 다른 연구에서는 좋은 성능을 보입니다.(Appendix).  안타깝게도 GoogLeNet에 이어 2위를 차지한 VGGNet에 대해서 정리하도록 하겠습니다.

### 0.Abstract

<left><img src = "https://user-images.githubusercontent.com/78463348/124843218-6b36ad80-dfcc-11eb-8ab6-82df7df39aa2.PNG" width = 90% height = "90%">

The effect of the convolutional network depth on its accuracy in the large-scale image recognition , Increasing depth using an architecture with very small (3X3) convolution filters 이라는 내용이 중요한데 즉, 3X3 convolution filter를 이용하고, layer의 개수를 16~19만큼 deep하게 늘려서 increasing depth를 만들었고, 이를 통해 large-scale image recognition에서 좋은 결과를 얻었다는 것을 알 수 있습니다.

### 1.Introduction

###### <img src = "https://user-images.githubusercontent.com/78463348/124843275-97522e80-dfcc-11eb-8082-b1d39ea9490a.PNG" width = 90% height = 90%>

<left><img src = "https://user-images.githubusercontent.com/78463348/124843294-9faa6980-dfcc-11eb-9b45-4ef16aadc2c6.PNG" width = 90% height = 90%>

<left><img src = "https://user-images.githubusercontent.com/78463348/124843295-a0430000-dfcc-11eb-8c7b-f1c3d155744b.PNG" width = 90% height = 90%>

ConvNet이 어느덧 computer vision 영역에서 유용한 역할을 하게 되면서, 기존 AlexNet구조를 향상시키기 위해 많은 시도들이 있었습니다.  이번 논문에서는 'depth'라는 ConvNet architecture의 중요한 측면을 다루고자 합니다. 우리는 이 구조의 다른 parameter들을 고정시키고 꾸준히 convolution layer들을 추가시키므로써 depth를 증가시켰습니다. 이는 모든 layer에 3x3 convolution filter와 같은 매우 작은 filter를 사용했기에 가능했습니다.

결과적으로 더 정확한 ConvNet architecture를 생각해냈는데, 이는 최신 ILSVRC classification과 localisation tasks를 더 정확하게 해결할 뿐만 아니라 다른 image recognition dataset에 적용가능하며, 상대적으로 simple pipelines (예를 들어, deep features classified by a linear SVM without fine-tuning)의 일부분으로 사용할 때 최선의 성능을 얻을 수 있었습니다.

논문의 나머지는 다음과 같이 구성됩니다(목차 설명). Sect 2에서는 ConvNet configuration을 설명하고, Sect 3에서 제시된 image classification training과 evaluation의 결과를 보여줍니다. Sect 4와 Sect 5에서는 ILSVRC classification task와 비교해서 구조를 설명합니다. 

### 2-1.Architecture

VGGNet의 기본설정에 대해 언급한다.

<left><img src = "https://user-images.githubusercontent.com/78463348/124843297-a0db9680-dfcc-11eb-8692-897461b7110b.PNG" width = 90% height = 90%>

ConvNet의 input은 224X224 RGB 이미지로 고정합니다. Input image(Traininng Dataset)에 대한 preprocessing은 RGB mean value만 빼주는 것만 적용합니다. (RGB mean value란?  이미지 상에 pixel들이 갖고 있는 R,G,B 각각의 값들의 평균을 의미합니다)

Image는 convolution layer들을 지나게 되는데, receptive field의 크기는 3x3의 크기를 가지고 있습니다. 

> receptive field란 filter가 한 번에 보는 영역이다. receptive field가 높으면 전체적인 특징을 잡아내는데 유용하다.
>
> 3X3을 선택한 이유는 left,right,up,down을 고려할 수 있는 최소한의 receptive field이기 때문이다.

1X1 conv filter도 사용되었는데 이는 input channels의 linear transformation 으로 보여질 수 있습니다.(여기는 사실 잘 모르겠어요)

Spatial padding이 사용되는 데, padding을 쓰는 목적은 convolution 이후에 spatial resolution을 보존하기 위해서입니다. conv filter의 stride  =1 이고 3x3 conv layer에 1 pixel padding이 적용되면  원래 해상도(이미지 크기)를 유지할 수 있습니다.

Pooling layer도 사용되었는데, Max pooling은 conv layer 다음에 적용되었으며, 총 5개의 max pooling layer로 구성됩니다. pooling 연산은 2X2 size와 stride = 2로 구성됩니다.

Convolution layer가 stack 된 이후에 FC layer가 등장하게 되는데, 총 3개의 Fully-Connected layers이 등장하며 처음 두 개의 FC layer는 4096개의 channel을 가지고 있습니다. 마지막 layer는 soft-max layer로 ILSVRC classification을 위해 1000개의 채널을 포함하고 있습니다.(class가 1000개라 이를 분류하기 위해 1000개의 channel로 이루어짐)

모든 hidden layer에는 비선형 함수인 ReLU를 가지고 있는데, 이번  Networks에서는 AlexNet에서 사용했던 Local Response Normalisation(LRN) 을 사용하지 않았습니다. 이유는, ILSVRC dataset에서 성능 향상을 가지고 있지 않는데다, 메모리 소모 및 연산량의 증가로 시간이 그만큼 소요되기 때문에 사용하지 않은 것이기 때문입니다.



### 2-2. Configurations

<img src = "https://user-images.githubusercontent.com/78463348/124843298-a1742d00-dfcc-11eb-9365-f12c9ca89a44.PNG">

configurations 에서는 A에서 E까지의 구조로 나뉠 수 있는데, 모든 구조는 2.1에서 설명한 구조를 그대로 따르되, 단지 깊이를 조금씩 변형시키면서 연구를 진행한것이라고 언급했습니다. Layer는 11 weight 부터 19 weight layer까지 구성되어 있으며 구조는 Table 1에 잘 표현 되어 있습니다.





<img src = "https://user-images.githubusercontent.com/78463348/124843303-a20cc380-dfcc-11eb-9398-25104dd62d43.PNG">



<img src = "https://user-images.githubusercontent.com/78463348/124843860-cfa63c80-dfcd-11eb-9f6c-4d2c553e6143.PNG">

<left><img src = "https://user-images.githubusercontent.com/78463348/124843299-a1742d00-dfcc-11eb-9eff-83039ab2ff0f.PNG">

<img src = "https://user-images.githubusercontent.com/78463348/124843306-a2a55a00-dfcc-11eb-8521-2c8c7e833414.PNG" >

이후 깊이를 늘렸음에도 불구하고, .weight의 개수 (Parameter의 개수) 가 더 늘어나지 않는 다는 것을 보여줍니다.



### 2-3. Discussion

<left><img src = "https://user-images.githubusercontent.com/78463348/124843300-a20cc380-dfcc-11eb-95c3-12dc98c9d06b.PNG">
​    
첫번째 convolution에서 상대적으로 large receptive fields를 쓰기 보다 3X3의 작은 receptive fields를 사용했습니다.  이유는 3X3 convolution layer를 쌓는 것이 5X5 convolution filter를 사용하는 것과 같은 효과를 가져오기 때문입니다.  즉 5X5 conv filter를 3X3 conv filter 2개로 나누어(factorizing) 사용한다고 합니다.

<left><img src = "https://user-images.githubusercontent.com/78463348/124843307-a33df080-dfcc-11eb-832f-d701c81d9033.PNG" width = 90% height = 90%>

<left><img src = "https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F59CI0%2FbtqAPdxQrQQ%2FNIX745po8xCjtzMtMcvnGK%2Fimg.png" width = 70% height = 70%>

이후 위의 그림과 같이 3X3 conv filter를 사용한 conv layer를 두번 사용하는 것이 7X7 conv filter를 1번 사용한 것과 같은 효과를 가지는 것을 볼 수 있습니다. 그럼 동등한 효과를 가지는 이 방식을 굳이 왜 사용해야하는 지는 두 가지 이유를 들어 설명합니다. 

첫번째로는 3X3 conv filter 3개와 7X7 conv filter 1개를 비교하면서 설명하는데 한 번 conv filter를 적용할 때마다 ReLU라고 하는 activation function을 적용하게 되는데 이는 non-linear한 문제에 도움을 줍니다. 결국 7X7 conv filter 1개보다 3X3 conv filter 3개를 적용하는 것이 더 많은 activation function을 적용할 수 있어 이 방식을 사용하는 것입니다. 

두번째로는 parameter의 수가 줄어둡니다. 해당 논문에도 쓰여져 있듯이  3X3 conv filter 3개가 가지는 parameter수는 총 27C^2 (여기서 C는 channel의 개수입니다.) 이에 반해 7X7 conv filter 1개가 가지는 parameter 개수는 49C^2입니다. 따라서 같은 효과를 가지면서 parameter를 줄여 overfitting을 방지하는 효과까지 가질 수 있습니다. 

왜 parameter를 줄이면 overfitting이 감소하는지는 하단의 링크를 참고해주시면 감사하겠습니다. (제가 참고한 자료를 쓰신 분이 정리한 내용입니다.)

> https://89douner.tistory.com/55



<left><img src = "https://user-images.githubusercontent.com/78463348/124843309-a33df080-dfcc-11eb-9ae7-b871a906c8ba.PNG" width = 90% height = 90%>

또한 VGG C  모델에는 1X1 conv layer도 적용하는데, 이유는 기존 receptive field에 영향을 주지않고 non-linearity를 증가시키기 위해서라고 합니다.



### 3. Classification Framework

### 3-1. Training

<left><img src = "https://user-images.githubusercontent.com/78463348/124843310-a3d68700-dfcc-11eb-9f63-ea966733b3a9.PNG" width = 90% height = 90%>

일단 hyperparameter들을 어떤 값으로 설정했는지 소개하고 있습니다.

1.cost function은 multinomial logistic regression (logistic regression을 통해 두 가지 이상의 분류 문제를 다루는 것) 를 이용했고 이는 Cross Entropy와 같다.

2.Mini-batch gradient descent를 사용하는데, 이 때 mini-batch 크기는 256으로 정해졌다.

3.Optimizer에서 Momentum = 0.9를 갖도록 hyperparameter설정

4.L2 regularization을 사용하며 L2 penalty를 5X10^(-4)으로 둔다. 

5.Dropout을 사용하며, 처음 두 FC-layer에 이용한다. dropout ratio = 0.5로 둔다.

6.Learning rate는 10^(-2)으로 설정 이후 validation set accuracy가 증가하지 않을 때 learning rate를 10만큼 나눠서 감소시킨다.

(Krizhevsky et al.,2012) = AlexNet 인데, AlexNet보다 깊이도 갚고 parameter수도 많음에도 불구하고, AlexNet보다 epoch가 적을 때 수렴했다고 밝혔다. 이는 두가지 이유를 들어 설명한다.

a)  Implicit regularisation 

- 앞선 논문중에 "This can be seen as imposing a regularisation on the 7X7 conv"라는 부분을 말하는 데, 2-3에서 7X7 conv filter 1개를 사용하는 것보다 3X3 conv filter 3개를 이용하는 것이 더 좋은 이유가 parameter의 수가 더 적어지기 때문이라는 점이었는데, 이를 implicit regularisation이라고 언급하고 있다.

b) pre-initialisation

- Pre-initialisation이란 먼저 학습하고 난 모델의 layer를 가져다가 쓰는 방식으로 진행하는 것을 말하는데, 여기서는 VGG-A 모델(16 layer) 을 학습하고, 이후 B,C,D,E 모델을 구성할 때 학습된 layer를 가져다 쓴다. 자세히 나타내면 A 모델의 처음 4개 conv layer와 마지막 3개 FC layer를 사용했다고 한다. 이 방식으로 통해 최적의 초기값을 설정해줘서 학습을 용이하게 해준다.





