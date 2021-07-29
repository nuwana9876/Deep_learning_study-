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

- 

<left><img src = "C:/Users/Park%20Jun%20Tae/Desktop/%EB%94%A5%EB%9F%AC%EB%8B%9D%20md%ED%8C%8C%EC%9D%BC/%EB%85%BC%EB%AC%B8%EC%9D%B8%EC%9A%A9%20%EC%82%AC%EC%A7%84%ED%8C%8C%EC%9D%BC%20%EB%AA%A8%EC%9D%8C/VGGNet/VGGNet%EB%85%BC%EB%AC%B8%EC%9D%B8%EC%9A%A915.PNG" width = "80%" height = "80%">



#### Training image size

VGGNet 모델을 학습할 때, 먼저 training image를 VGG모델의 input size에 맞게 바꿔줘야 한다. 

<left><img src = "C:\Users\Park Jun Tae\AppData\Roaming\Typora\typora-user-images\image-20210709105951984.png" width = 90% height = 90%>


예를 들어 S라고 하는 smallest side를 256이라고 하면, training image의 넓이 또는 높이 중에서 더 작은 side를 256으로 줄여준다. 이 때 aspect ratio를 유지하면서 나머지 side도 rescaling을 해주는데 이런 방식을 "isotropically-rescaled" 했다고 한다.

<left><img src = "C:\Users\Park Jun Tae\AppData\Roaming\Typora\typora-user-images\image-20210709110412049.png">


위와 같이 training image를 isotropically-rescaled 해줬다면, 이후에는 224X224의 크기로 random하게 crop해준다.

<left><img src = "C:\Users\Park Jun Tae\AppData\Roaming\Typora\typora-user-images\image-20210709111200783.png" width = 90% height = 90%>


S를 설정하는 방식에는 두 가지가 있는데, 첫번째로는 S를 256 or 384로 고정하는 방식이다. 이를 우리는 'Single-scale training'이라고 한다. S = 384로 설정했을 때 , .training을 빨리 하기 위해서 S가 256일 때 학습시킨 가중치로 initialize 두고 학습을 시키게 된다. 학습 시에는 learning rate를 10^(-3)으로 줄여주고 학습을 시킨다.

<left><img src = "C:\Users\Park Jun Tae\AppData\Roaming\Typora\typora-user-images\image-20210709111835805.png" width = "90%" height = 90%>


<left><img src = "C:\Users\Park Jun Tae\AppData\Roaming\Typora\typora-user-images\image-20210709111923580.png" width = 90% height = 90%>


두번째로는 'multi-scale training'이라는 방식을 사용하는데, 이는 S를 고정하지 않고 Smin에서 Smax사이의 값 중에서 random하게 설정해주는 방식이다. 객체(Object)들이 모두 같은 사이즈가 아니라 각각 다를 수가 있기 때문에 random하게 multi-scale로 학습시키면 학습 효과가 더 좋아질 수 있다. 

이러한 data augmentation 방식을 'Scale jittering'이라고 부른다.

Multi-scale training을 학습 시킬 때는 빠르게 학습시키기 위해서, 먼저 학습시킨 single scale training 방식을 이용한 모델을 가지고 이를 이용해 fine-tuning을 해준다.

<left><img src = "C:\Users\Park Jun Tae\AppData\Roaming\Typora\typora-user-images\image-20210709113339988.png" width = 90% height = 90%>
Test 할 때에도 train 할 때와 마찬가지로 rescaling을 적용한다. Training에서는 rescaling시에 기준 값을 S라고 했지만, test에서는 rescaling시에 기준 값을 Q라고 한다. S와 Q가 같을 필요는 없다. 오히려 training  image에서 설정된 S값마다 다른 Q를 적용하면 성능이 더 좋아진다고 한다. 


중요한 점은 VGGNet 모델은 training할 때랑 중간중간 overfitting을 막기위해 Testing(Validation)할 때 쓰이는 CNN 구조가 약간 다르다.

Test에서는 첫번째 FC layer를 7X7 conv layer로 바꿔주고, 마지막 FC layer 두 개를 1x1 conv layer로 바꿔준 것이 training과 다른 점이다. 이를 통해 training에서와 달리 uncropped image에 적용할 수 있다는 점이 장점이다.


<left><img src = "C:\Users\Park Jun Tae\AppData\Roaming\Typora\typora-user-images\image-20210709114223466.png">


<right><img src = "C:\Users\Park Jun Tae\AppData\Roaming\Typora\typora-user-images\image-20210709114255201.png">


제일 위의 사진은 feature map이 FC layer를 통과하기 전에, flatten 하는 순서를 나타낸 것이다. 그 밑의 사진은 feature map이 1X1 conv filter를 적용하는 사진인데, 이를 비교해 봤을 때, flatten 하는 순서에 영향이 있을 뿐, 구조 자체는 FC layer와 conv layer 모두 동일하다는 것을 알 수 있다.

<img src = "C:\Users\Park Jun Tae\AppData\Roaming\Typora\typora-user-images\image-20210709115842205.png" align = "left" width = 50% height = 50%><img src = "C:\Users\Park Jun Tae\AppData\Roaming\Typora\typora-user-images\image-20210709120034856.png" align = "right" width  = 50% height = 50%>



















(왼쪽의 사진은 Test에서의 VGGNet 구조이며, 오른쪽 사진은  Training에서의 VGGNet 구조이다. 왼쪽 이미지 상에서는 fully-connected라고 설명이 되어있지만, 실제로는 3개의 1X1 conv layer인 것이다.)

어찌보면 논문을 읽었을 때 가장 헷갈렸던 부분이 있었는데, 바로 The result is a class score map with the number of channels equal to the number of classes, and a variable spatial resolution, dependent on the input image size. Finally, to obtain a fixed-size vector of class scores for the image, the class score map is spatially averaged (sum-pooled). 라는 부분이다.

다들 공통적으로 이해가 안 갔을 거라고 나는 생각한다.(물론 아닐 수도 있다....) 

위에서 잠깐 언급했듯이 Training시에는 crop하는 방법을 사용하지만, Test시에는 FC layer를 conv layer로 바꿔줘서 uncropped image에 적용할 수 있다고 한다. 즉 , FC layer의 경우 MLP(Multi-Layer Perceptron)으로, 사실상 각 perceptron의 입력과 출력 노드가 정해져 있기 때문에 항상 입력 노드가 정해준 값과 동일해야한다. 그러나 conv 연산의 경우, 신경쓰지 않아도 된다.  *기존 crop을 통해 얻은 224X224 입력 image size가 VGG model에 입력되었을 때, classifier에 해당하는 부분 (FC layer가 1X1 conv layer로 바뀐 부분)을 거친 최종 output feature map size가 입력 image size에 따라 서로 달라집니다.* (사실 이 부분은 잘 이해되지 않는다.)

<img src = "C:\Users\Park Jun Tae\AppData\Roaming\Typora\typora-user-images\image-20210709122325547.png" align = "left">

위 그림에서 보다시피, output feature map size가 1X1이 아닌 경우가 있을 수 있다. 상단 그림의 경우, classifier 단계(1X1 conv filter가 적용되기 시작하는 부분)에서 1X1 conv filter가 feature map에 잘 적용 된 것으로 보이지만, 아래 그림의 경우 1X1 conv filter가 feature map size와 다르다는 것을 알 수 있다.

만약 큰 이미지가 입력으로 들어오게 되면 1X1 conv filter를 적용할 때, feature map size가 1X1이 아닌 7X7이 될 수도 있습니다. 즉, 1X1 conv filter를 거치고 softmax에 들어가기 전 output feature map size가 이미지 크기에 따라 달라지는데, 7X71000 output feature map size를 얻을 수도 있다는 것이다. 이 때, 7X7 output feature map을 class score map이라고 한다. 이 feature map들을 spatially averaged 해준다. (대략적으로 mean or average pooling을 한다고 해석)

이후 softmax이후에 filpped image와 original image의 평균값을 통해 최종 score를 출력하게 된다.

<img src = "C:\Users\Park Jun Tae\AppData\Roaming\Typora\typora-user-images\image-20210709124849125.png">

AlexNet의 경우, Test시에 1개의 이미지에 대해 좌상단,우상단,가운데,좌하단,우하단으로 crop하고 각각 이미지를 좌우반전시켜서 10개의 augmented images를 사용한다. 이 이미지들을 평균을 취하는 방식으로 최종 score를 출력한다.(Softmax의 결과 확률값이 나오는데 각각의 class에서 나오는 10개의 값들에 대해 평균)이로 인해 계산량이 많아져 속도가 느려진다. 그러나 FC layer를 1X1 conv layer로 대체하고 큰 이미지를 넣어 학습을 시켰고, 또한 data augmentation도 horizontal filpping 만 적용했는데도 좋은 효과가 났다고 한다.

<img src = "C:\Users\Park Jun Tae\AppData\Roaming\Typora\typora-user-images\image-20210712075924180.png">

VGGNet에서는 GoogleNet에서 사용한 multi-crop 방식이 좋은 성능을 보인다는 점을 알고, VGGNet의 dense evaluation을 보완하기 위한 방안으로 multi-crop evaluation을 같이 사용했다고 한다. 이후 두 가지 방식을 적절히 활용해서 validation한 결과 좋은 성능을 낸 것으로 확인되었다.



### 4. Classification Experiments

<img src = "C:\Users\Park Jun Tae\AppData\Roaming\Typora\typora-user-images\image-20210712080608196.png"><img src = "C:\Users\Park Jun Tae\AppData\Roaming\Typora\typora-user-images\image-20210712080655312.png">

Dataset은 ILSVRC-2012를 사용했으며, 이미지의 class는 1000개로 나뉘어 있고 dataset은 train (1.3M), valid (50K), test set (100K)으로 나뉘어 있다. 또한 classification 평가는 top-1과 top-5 error를 보고 확인 할 수 있다. top-1 error는 multi-class classificaiton error , top-5 error는 기존 ILSVRC에서 요구하는 test 기준을 사용했다고 한다.

대부분의 experiment에서 validation set을 test set으로 사용했다고 한다.



### 4.1 Single Scale Evaluation

Single scale evaluation이란 앞서 설명되었듯이, test시에 image size(scale)이 고정되어 있는 것을 의미한다. training image size를 설정해주는 방식에는 두 가지 방식이 있었는데, 첫번째는 training image size (S)를 256 or 384로 fix시켜주는 single scaling training과 두번째로는 256~512 size에서 random하게 골라 multi-scaling training을 하는 방식이 있다. S=Q라고 했을 때 test image size가 고정되고, multi scaling training방식에서는 0.5(256+512) = 384로 고정된다.



<img src = "C:\Users\Park Jun Tae\AppData\Roaming\Typora\typora-user-images\image-20210712083413135.png">



첫번째로 Local response normalisation이라는 AlexNet에서 사용된 기법을 이용한 모델과 그렇지 않은 모델 사이의 성능 향상이 두드러지지 않았기에 LRN을 사용하지 않았다.

두번째로는 ConvNet의 깊이가 깊을 수록 classificaiton error가 줄어든다는 것이다. 특이하게도 앞서 말한 내용과는 달리 1X1 conv filter를 사용하고 있는 C 모델과 3X3 conv filter를 사용하고 있는 D모델을 비교했을 때 깊이가 더 앏은 3X3 conv filter를 사용한 D모델의 성능이 더 좋다는 점을 밝히고 있는데, 이는 깊이를 깊게 쌓을 수록 non linearity를 더 잘 표현할 수 있지만, 3X3 conv filter가 spatial context (공간이나 위치정보) 의 특징을 더 잘 추출하기 때문에 3X3 conv filter를 사용하는 것이 더 좋다는 것을 언급하고 있다. 

### 4.2 Multi-scale evaluation

<img src = "C:\Users\Park Jun Tae\AppData\Roaming\Typora\typora-user-images\image-20210712085417338.png">

<img src = "C:\Users\Park Jun Tae\AppData\Roaming\Typora\typora-user-images\image-20210712085441829.png">

Multi-scale evaluation은 test 이미지를 multi-scale로 설정해서 evaluation한 방식을 말한다. 하나의 S 사이즈에 대해 여러 test image로 evaluation하는 것을 볼 수 있다. training과 testing scale 과의 많은 차이는 성능의 감소를 불러오는데, 이를 위해 Q를 training과 가깝게 S-32, S, S+32로 설정한다. test time에서 scale jittering 은 single scale과 동일 모델로 비교했을 때, 더 좋은 성능을 가져온다고 한다. 

### 4.3 Multi-crop evaluation & 4.4 ConvNet fusion

<img src = "C:\Users\Park Jun Tae\AppData\Roaming\Typora\typora-user-images\image-20210712092730208.png">

여기서는 dense evaluation 방식과 multi-crop evaluation을 이용한 validation 결과를 비교하고 있는데, multi-crop evaluation이 dense evaluation보다 살짝 더 좋으며, 저자는 두 방식의 결과의 평균을 통해서 구한 새로운 방식인 ConvNet fusion으로 validation한 결과를 이 다음에 보여주고 있다.



### 4.5 Comparison with the state of the art

<img src = "C:\Users\Park Jun Tae\AppData\Roaming\Typora\typora-user-images\image-20210712093154386.png">

<img src = "C:\Users\Park Jun Tae\AppData\Roaming\Typora\typora-user-images\image-20210712093231498.png">

ILSVRC 대회에서 제출을 위해 7개 모듈의 ensemble 기법을 적용했는데, 대회가 끝난 후 자체적으로 다시 실험한 결과 2개 모델(D,E)만 ensemble 한 결과가 더 좋았다고 한다.



