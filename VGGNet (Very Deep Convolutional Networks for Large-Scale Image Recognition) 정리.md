# VGGNet (Very Deep Convolutional Networks for Large-Scale Image Recognition) 정리

VGGNet 정리는 Time Traveler님의 VGGNet 논문 리뷰를 참고해서 만들었음을 밝힙니다.

> ###### https://89douner.tistory.com/61 (여기 진짜 논문 맛집입니다!)



AlexNet의 등장 이후 image classification분야에서 CNN 모델이 주목을 받았다. 이후 ILSVRC 2014년 대회에 VGGNet과 GoogLeNet 모델이 등장하게 됩니다. 특히 이 두 모델은 AlexNet보다 더 깊은 layer를 쌓아서 뛰어난 성능을 자랑합니다. ILSVRC 대회에서는 GoogLeNet보다 이미지 분류 성능은 낮았지만, 다른 연구에서는 좋은 성능을 보입니다.(Appendix).  안타깝게도 GoogLeNet에 이어 2위를 차지한 VGGNet에 대해서 정리하도록 하겠습니다.

###### 1.Abstract

The effect of the convolutional network depth on its accuracy in the large-scale image recognition , Increasing depth using an architecture with very small (3X3) convolution filters 이라는 내용이 중요한데 즉, 3X3 convolution filter를 이용하고, layer의 개수를 16~19만큼 deep하게 늘려서 increasing depth를 만들었고, 이를 통해 large-scale image recognition에서 좋은 결과를 얻었다는 것을 알 수 있습니다.

###### 2.Introduction

ConvNet이 어느덧 computer vision 영역에서 유용한 역할을 하게 되면서, 기존 AlexNet구조를 향상시키기 위해 많은 시도들이 있었다.  이번 논문에서는 'depth'라는 ConvNet architecture의 중요한 측면을 다루고자 한다. 우리는 이 구조의 다른 parameter들을 고정시키고 꾸준히 convolution layer들을 추가시키므로써 depth를 증가시켰다. 이는 모든 layer에 3x3 convolution filter와 같은 매우 작은 filter를 사용했기에 가능했다.

결과적으로 더 정확한 ConvNet architecture를 생각해냈는데, 이는 최신 ILSVRC classification과 localisation tasks를 더 정확하게 해결할 뿐만 아니라 다른 image recognition dataset에 적용가능하며, 상대적으로 simple pipelines (예를 들어, deep features classified by a linear SVM without fine-tuning)의 일부분으로 사용할 때 최선의 성능을 얻을 수 있었다.

논문의 나머지는 다음과 같이 구성된다(목차 설명). Sect 2에서는 ConvNet configuration을 설명하고, Sect 3에서 제시된 image classification training과 evaluation의 결과를 보여줍니다. Sect 4와 Sect 5에서는 ILSVRC classification task와 비교해서 구조를 설명합니다. 

###### 3.Architecture

VGGNet의 기본설정에 대해 언급한다.

ConvNet의 input은 224X224 RGB 이미지로 고정한다. Input image(Traininng Dataset)에 대한 preprocessing은 RGB mean value만 빼주는 것만 적용한다. (RGB mean value란?  이미지 상에 pixel들이 갖고 있는 R,G,B 각각의 값들의 평균을 의미한다)

Image는 convolution layer들을 지나게 되는데, receptive field의 크기는 3x3의 크기를 가지고 있다. 

> receptive field란 filter가 한 번에 보는 영역이다. receptive field가 높으면 전체적인 특징을 잡아내는데 유용하다.
>
> 3X3을 선택한 이유는 left,right,up,down을 고려할 수 있는 최소한의 receptive field이기 때문이다.

1X1 conv filter도 사용되었는데 이는 input channels의 linear transformation 으로 보여질 수 있다.(여기는 사실 잘 모르겠어요)

Spatial padding이 사용되는 데, 이 목적은 convolution 이후에 spatial resolution을 보존하기 위해서이다. conv filter의 stride  =1 이고 3x3 conv layer에 1 pixel padding이 적용되면  원래 해상도(이미지 크기)를 유지할 수 있다.

Pooling layer도 사용되었는데, Max pooling은 conv layer 다음에 적용되었으며, 총 5개의 max pooling layer로 구성된다. pooling 연산은 2X2 size와 stride = 2로 구성된다.

Convolution layer가 stack 된 이후에 FC layer가 등장하게 되는데, 총 3개의 Fully-Connected layers이 등장하며 처음 두 개의 FC layer는 4096개의 channel을 가지고 있다. 마지막 layer는 ILSVRC classification을 위해 1000개의 채널을 포함하고 있다.(class가 1000개라 이를 분류하기 위해 1000개의 channel로 이루어짐)















