# R-CNN (Rich feature hierarchies for accurate object detection and semantic segmentation)정리

R-CNN 논문 리뷰를 시작하도록 하겠습니다~!

제가 참고한 논문 리뷰나 출처는 가장 마지막에 적어놓도록 하겠습니다. 혹여나 적지 못한 출처의 경우 말씀해주시면 바로 찾아 적어놓도록 하겠습니다.



### Abstract

이 논문이 나오기 전 몇 년 동안 PASCAL VOC 데이터셋을 이용해서 Obejct detection을 수행하는 최고의 수행 방식은 high-level context의 복잡한 ensemble 모델이다. 이번 논문에서는 VOC 2012 데이터셋을 기준으로 object detection한 이전 모델과의 성능과 비교했을 때보다 30%를 향상시킨 간단한 알고리즘을 제안한다. 

> mean average precision (mAP) 라고 하는 object detection 성능 평가 지표가 53.3%이었다.

우리의 접근법은 두가지 key insight를 가지고 있는데, 이는 다음과 같다.

> 1. 객체를 localize 및 segment하기 위해 bottom-up 방식의 region proposal에  CNN을 적용
> 2. 데이터의 부족에 따른 domain-specific fine-tuning을 통한 supervised pre-training을 적용

R-CNN인 이유는 Regions with CNN features 이기 때문이라고 저자는 밝혔다.

### 1. Introduction

지난 10년간 다양한 visual recognition 작업에서 주로 SIFT와 HOG (gradient 기반의 특징점 추출 알고리즘)가 가장 많이 사용되었는데, 이는 2010년 ~ 2012년의 PASCAL VOC object detection에서 일반적으로 인정되는 방법이었다. 이후에 LeCun은 stochastic gradient descent (SGD) 방식을 보여주면서 backpropagation이 가능하게 되었고 이를 이용한 CNN이 등장하게 되었습니다. 

CNN은 1990년대 많은 사용에도 불구하고 CNN은 computer vision 분야에서 SVM(support vector machine)으로 인해 구식으로 여겨졌었다. 그러나 LeCun's CNN을  약간 변형해서, ReLU 함수의 사용과 Dropout regularization의 사용해 large CNN을 training 함으로써,  ILSVRC 에서 높은 이미지 분류 정확도를 보였다.이를 조금 더 확장시켜 Object Detection에 적용하면 어떨까 라는 생각을 하게 되었고, 생각을 확장시킨 결과, 이 논문은 PASCAL VOC에서  CNN을 이용한 Object detection 성능을 극적으로 향상시킨 최초의 논문이 되었다.

여기서 극적으로 성능을 향상시키는 데에는 2가지가 요구 되었다.

> 1.  이미지 classification 과는 달리 detection의 경우 이미지로 객체를 localizing 하는 것이 필요하다. 
> 2.  매우 작은 양의 주석을 가진 detection data를 가지고 high-capacity 모델을 훈련시켜야 한다

1번 문제의 경우 3가지의 방식이 제기되었다.

> 1. 문제를 regression 문제로 접근하는 방법
> 2. sliding-window detector 방식을 적용하는 것
> 3. recognition using regions 방식을 적용하는 것

결론적으로는 마지막 방식을 적용하게 되는데, 이유를 살펴보면 첫번째 방식의 문제는 실전에서 적용하기에는 정확도가 낮았는데, 이 방식을 이용해 VOC 2007에서 mAP가 30.5% 라는 결과를 냈고, 이 논문에서 사용한 방식으로 mAP가 58.5%라는 결과가 나온 점을 감안했을 때, 상당히 정확도가 좋지 않은 것을 알 수 있다. 

  첫번째 문제의 해결방안으로 sliding-window detector 방식을 사용하게 되었는데,  이 방식은 최소 20년동안 얼굴이나 보행자와 같은 category들에 제한적으로 사용되었다. 높은 공간적 해상도(high spatial resolution)를 얻기 위해 일반적으로 2개의 convolution layer와 pooling layer를 사용하며, 추가적으로 sliding-window approach 를 적용한다. 그러나 우리 network에서는 5개의 convolution layer를 사용하고  각 convolution layer마다 high receptive field (195x195 pixels)와 stride (32x32 pixels)를 가지고 있어 sliding-window paradigm내의 정확한 localization을 할 수 없고 이를 해결하는 것이 문제이다.

이를 위해 마지막 방식인 'recognition using regions'라는 방식을 사용해 CNN의 localization 문제를 해결했다. 논문에 있는 Figure 1 이미지를 보면 조금 더 이해가 쉬울 수 있다.

<img src = "https://jaehyeongan.github.io/image/rcnn.JPG">

> R-CNN process
>
> 1. input 이미지로부터 2000개의 독립적인 region proposal 생성 및 추출 (independent 하기에 중복도 가능) 
> 2. CNN을 통해 각 proposal 마다 고정된 길이의 feature vector를 추출 (CNN 적용 시 서로 다른 region shape에 영향을 받지않기 위해 image warp를 통해 fixed-size CNN input을 계산하도록 한다)
> 3. 이후 각 region 마다 category-specific linear SVM을 적용해 classification을 진행



이제 두번째 문제인 매우 작은 양의 주석을 가진 detection data를 가지고 high-capacity 모델을 훈련시켜야 한다는 점의 해결 방안이 필요하다. 이를 위해 논문에서는 'unsupervised pre-training, followed by supervised fine-tuning' 이라는 방식을 사용한다고 말한다. 이는 비지도학습의 pre-training과 이후 지도학습의 fine-tuning을 사용하는 것이다. 실질적으로는 'supervised pre-training on a large auxiliary dataset (ILSVRC), followed by domain-specific fine-tuning on a small dataset (PASCAL)' 라 할 수 있고, 이는 data가 부족한 high-capacity CNN을 학습시키는데에 효율적이다.  실험을 통해 detection을 fine-tuning시 8%의 mAP 성능향상과 더불어 VOC 2010에서 fine-tuning 후에 이전방식인 DPM방식과 비교해 33% -> 54%의 mAP를 얻을 수 있었다.

 (이 밑 문단의 경우는 잘 이해하지 못했다.)

우리의 system은 상당히 효율적이라고 할 수 있다. class-specific 연산은 상당히 작은 matrix-vector곱과 greedy non-maximum suppression뿐이다. 이 연산의 특징은 feature들은 모든 category들 공유하며, 이전에 사용된 region feature들보다 크기가 2배 더 낮은 features라는 것으로 결론이 도출된다.

중요한 것은 region에 의해 R-CNN이 작동하기 때문에 Semantic Segmentation에서도 수행할 수 있다. 또한 PASCAL 2011 test set을 가지고 segmentation task를 진행했을 때, 47.9%의 정확도를 얻었습니다.



### 2. Object detection with R-CNN

우리의 object detection system은 3가지의 모듈로 구성이 된다.

> 1. category-independent한 region proposals 생성 (이 proposal은 detector가 찾은 이용가능한 후보 detection들의 집합으로 정의한다.)
> 2. 각 region 으로부터 fixed-length feature vector를 추출하기 위한 large CNN
> 3. classification을 위한 class-specific linear SVMs

이번 section에서는 각 모듈마다 design decision, 각 모듈의 test-time usage를 보여주고, 어떻게 parameter들을 학습하는지에 대한 detail과 PASCAL VOC 2010~ 2012의 결과를 보여준다.



### 2.1 Module design

#### Region Proposal

최근 다양한 논문이나 글은 category-independent한 region proposal들을 생성하는 방법을 제공하는데, objectness, CPMC,category-independent object proposals 등 여러가지 방법이 있지만, 이 논문에서는 이전 detection 연구들과 제어(통제)된 비교를 하기위해서 selective search 방식을 사용한다.

> SelectiveSearch 방식
>
> 1. 이미지의 초기 segment를 정해, 많은 region들을 생성
> 2.  greedy(탐욕) 알고리즘을 사용해, 각 region을 기준으로 주변의 유사한 영역을 결합
> 3. 커질수 있을 만큼 결합된 region을 최종 region proposal로 제출

#### Feature extraction

caffe를 통해 Krizhevsky의 2012 논문 AlexNet CNN모델을 사용해서 각 region proposal로 부터 4096개의 feature vector를 추출한다. feature들은 mean- substracted 227x227 RGB image를 5개의 convolutional layer와 2개의 fully connected layer들을 통과시키면서 forward propagagion을 수행한다. 각 region proposal마다 feature를 계산하기 위해, 우리는 이미지를 CNN에 적용할 수 있도록, 이미지 input의 크기가 227x227 pixel이 되도록 고정한다. Region의 크기나 aspect ratio (종횡비)에 관계없이 요구되는 size에 맞추기 위해, 우리는 bounding box 주변의 모든 pixel을 warp 시켜준다. (warp란 규격에 맞게 만들기 위해서 휘게 만드는 과정이다.) 

##### (Appendix A : Object proposal transformations)

<img src = "https://t1.daumcdn.net/cfile/tistory/99F02F505CD83FC009">

위 내용을 더 잘 이해하기 위해서는 appendix A의 내용을 참고하면 좋다. CNN에 input을 넣는다는 것은 결국 fixed-size로 이미지를 바꿔주어야 한다는 점을 나타낸다. 이를 위해 이 논문에서는 transforming을 위한 2가지 방법을 제안한다. 

첫번째로는 "tightest square with context" 라는 방식으로, tightest squre안에 각 object proposal을 넣고, 해당 square에 포함된 이미지를 CNN input size로 조정한다. (isotropically => 모든 방향에 대한 정보가 고르게 분포되도록) /  이 방법에 대한 변형으로 "tightest square without context" 라는 방식이 있다. 이는 원래 object proposal에서 주변 배경을 제외하는 방법이다. 

두번째로는 "warp"라는 방식인데, 이는 anisotropical하게 각 object proposal을 CNN input으로 조정하는 방식이다. 이 때 이 논문에서는 context padding (p) 라는 내용이 등장하는데 이는 object proposal 주변 배경을 얼마나 사용할 것인지에 대한 pixel의 크기를 담고 있다고 보면 된다. 예를 들어 p = 0 의 경우 figure 7의 윗 줄에 해당하고, p = 16의 경우 figure 7의 아랫 줄에 해당한다. 이 논문에서는 p = 16을 사용한 warping이 다른 방법보다 mAP가 3~5 정도 높다고 밝혔다.

모든 방식에서, source rectangle (원본 이미지)에서 늘리고자 한 이미지 크기까지 warp시킬 때, warp한 이미지의 크기가 CNN input size와 맞지 않을 경우, 맞지 않는 부분은 이미지의 mean으로 대체한다.



### 2.2 Test-time detection

> 1. test image를 selective search 방식을 이용해 2000개의 region proposal을 추출
> 2. 각 proposal에 대해 warp시키고, 이를 CNN에 넣어 feature 추출
> 3. 각 class에 대해, 추출한 feature vector를 SVM을 이용해 score 계산
> 4.  이미지에 따른 scored region이 계산되면, greedy NMS (non-maximum suppression) 을 각 class마다 independent하게 적용하고, score가 높은 region과의 IoU가 우리가 지정한 threshold보다 큰 region의 경우 제거

#### Run-time analysis

두 개의 특성이 이 detection에 있어서 효율적으로 만들어준다.

1. 모든 CNN parameter가 모든 category를 공유한다
2. CNN에 의해 계산된 feature vector들은 low-dimensional하다는 점이다.

즉 feature를 공유한 것이 (GPU - 13s/image | CPU - 53s/image) region proposal과 feature를 계산 하는 시간을 줄여주었다. 

class-specific 계산은 오직 feature와 SVM weights 그리고 non-maximum suppression 사이의 dot product 만 있기 때문에 low-dimensional한 계산이 가능해진다.



> **NMS(Non-maximum suppresion)**
>
> 1. 예측한 bounding box들의 예측 점수를 내림차순으로 정렬
> 2. 높은 점수의 박스부터 시작하여 나머지 박스들 간의 IoU를 계산
> 3. IoU값이 지정한 threhold보다 높은 박스를 제거
> 4. 최적의 박스만 남을 떄까지 위 과정을 반복



### 2.3 Training

#### Supervised pre-training

ILSVRC 2012 large auxiliary dataset에 pre-trained 된 CNN을 이용했다.

#### Domain-specific fine-tuning

CNN을 new task (detection) 와 new domain (warped VOC windows) 에 적용하기 위해서, warped region proposals만을 이용해서 SGD를 사용해 CNN parameter를 training 시킨다.

ImageNet-specific 1000-way classification을 randomly initialized 21-way classification으로 대체한다. 이 때 class 20개는 VOC class (object class N개) 이며 나머지 1개는 background를 의미한다. 또한 CNN 구조는 바꾸지 않는다.

우리는 모든 region proposal을 ground truth box와 overlap 시켰을 때, 0.5 IoU 이상이면 해당 box의 class에 대해 positive sample, 아니라면 negative sample 이라고 정의한다. 

learning rate는 0.001이며 (초기 pre-training rate의 1/10이다) , 각 SGD iteration마다, uniform하게 mini-batch size를 positive sample 32개, negative sample 96개로 총 128개씩 구성한다. 

우리는 sampling이 positive sample에 편향되는데, 이 이유는 positive sample이 background에 비해 매우 적기 때문이다.

#### Object category classifiers

차를 detect하기 위한 binary classifier 훈련한다고 생각해보자. 차 전체를 포함하고 있는 region의 경우, positive sample이며, 차를 포함하지 않는 region일 경우 negative sample로 판단할 수 있다. 그러나 부분적으로 차와 배경이 overlap된 경우라면 어떻게 해야할까? 

이 때는 threshold를 정해줘서 positive와 negative sample을 나눠주는 것이 중요한데 

positive sample : 각 class별 object의 ground-truth bounding boxes

negative sample : 각 class별 object의 ground-truth와 IoU가 0.3미만인 region

##### (Appendix B : Positive vs. negative examples and softmax)

1.왜 CNN fine-tuninng에서와 object detection SVM training에서의 positive와 negative를 다르게 정의했는가?

이는 fine-tuning에서는 최대 IoU를 가지는 object proposal을 ground-truth에 mapping 했으며, IoU 가 0.5 이상일 때, 이를 Positive로 정의했다. 나머지 proposal에 대해서는 negative로 간주했다.

이와는 반대로, training SVM에서는, 각 class에 대해 positive sample로써 ground-truth boxes만을 가지고, IoU가 0.3 미만이라면, 이를 그 class에 대해 negative로 간주한다.  만약 0.3 IoU이상이지만 ground-truth box가 아니라면 이는 무시한다.

이렇게 사용하게 된 이유는, 처음에는 SVM을 학습할 때의  positive / negative sample을 이용했으나, 지금의 방식과 비교했을 때 성능이 좋지 않았기에 지금의 방식으로 적용했다.

2.왜 fine-tuning 후에 SVM을 training 시키는가?

이는 더 명확하게 하기 위해 , object detector로 21-way softmax regression classifier를 적용했으나 이는 2007 VOC에서 54.2% -> 50.9%로 mAP가 감소하는 것으로 나타났다. 

이는 여러요인이 합쳐져서 나온 결과인데, fine-tuning에서 positive sample의 정의가 정확한 localization을 강조하지 못하고, softmax classifier가 SVM training에 쓰인 hard negative의 subset보다는 random하게 negative sample에서 sampling한 것을 이유로 꼽았다.

위 결과는 fine-tuning 후 SVM training 없이 동일한 성능을 얻을 수 있다는 것을 보여준다.  



### 2.4 Results on PASVAL VOC 2010-12

<img src = "https://t1.daumcdn.net/cfile/tistory/99AB354F5CD83FC00A">



### 2.5 Results on ILSVRC2013 detection

<img src = "https://t1.daumcdn.net/cfile/tistory/99527A465CD83FC00A">

PASVAL VOC 보다 분류해야 할 class가 많아 mAP가 낮지만, 다른 방법보다는 좋다.



### 3. Visualization, ablation, and modes of error

### 3.1. Visualizing learned features

첫번째 layer filter는 직접적으로 visualize 할 수 있고, 쉽게 이해할 수 있습니다. 이는 방향 edges와 상대 색깔을 capture한다.  그 다음 layer에 대해 이해하는 것은 이보다는 더 어려울 것이다. 그래서 우리는 visually attractive deconvolutional approach 제시했다.

즉 region proposals에 대한 unit's activations을 계산하고 activation을 내림차순으로 정렬, 이후 NMS를 적용하고, top-scoring region을 display한다. 이 method는 "speak for itself" 할 수 있는 unit을 선택하도록 한다. 

<img src = "https://t1.daumcdn.net/cfile/tistory/99C556465CD83FC005">

Figure 4에서 두번째 줄을 보면 개와 dot array를 나타내고, 5번째 줄에서는 삼각지붕 창문집과 전차를 나타내는데, 이를 통해서 network가 shape, texture, color 그리고 물질의 특성에 영향을 받는다는 것을 알 수 있다. 이후 fully connected layer fc6 은 이 풍부한 feature들의 large set of compositions을 모델링 할 수 있다.

(이제보니 이 부분 설명을 잘 못한거 같다는 생각이...)

###  3.2 Ablation studies

#### Performance layer-by-layer, without fine-tuning

어떤 layer가 detection 성능에 중요한 layer인지 알아보기 위해, 우리는 CNN의 마지막 3개 layer애 대한 VOC 2007의 결과를 분석했다. 

Layer 5에 대해서는 3.1에서 간략하게 설명했으나 간단히 설명하면, pool 5는 9x9x256차원으로 9216차원이다.

Layer 6는 pool5와 fully connected 된 layer이다. intermediate vector는 component-wise half-wave rectified이다.(x <- max(0,x)) 또한 차원은 4096차원이다.

Layer 7은 network의 final layer 이다. 차원은 4096차원이다.

<img src = "https://t1.daumcdn.net/cfile/tistory/99CB69505CD83FC00B">

이 논문에서는 PASCAL에 대해 fine-tuning없이 ILSVRC 2012에만 pre-train된 CNN을 사용했는데, Table 2의 1~3줄을 보면 fc7이 fc6보다 성능이 떨어졌다. 이는, 29% 즉 16.8million개의 CNN parameter들을 mAP 감소 없이 제거할 수 있다는 점이다. 

여기서 더 놀라운 점은 fc7과 fc6 모두 제거하는 것이 비록 pool5가 CNN parameter의 6%만을 가지고 있음에도 불구하고 좋은 결과를 낸다는 점이다. 

CNN의 대표적인 power는 dense connected layer가 아닌 convolution layer에서 나온다. 이는 HOG 관점에서 볼 때, dense feature map을 계산하는데에 있어서 잠재적인 utility를 나타낸다. 이 representation으로 pool5 feature위에 DPM을 포함한 sliding-window detector를 수행할 수 있게 되었다.



#### Performance layer-by-layer, with fine-tuning

VOC 2007 train data로 fine-tuning된 CNN을 사용하는데, Table 2의 4~6줄을 보면 mAP가 54.2%로 8% 향상된 결과를 볼 수 있다.

이를 통해 pool5는 일반적이고, domain-specific non-linear classifier에 의해 향상이 되었다는 것을 알 수 있다.



#### Comparison to recent feature learning methods

> 이 부분은 그냥 읽어보면 될 것 같다. 이전 방법론들과 R-CNN의 성능을 수치적으로 비교해석한 것이다



### 3.3 Detection error analysis

<img src = "https://t1.daumcdn.net/cfile/tistory/998EF2435CD83FC027">

background나 object classes에 대한 confusion보다 Loc으로 나타나는 poor localization이 errors의 주요 원인이다.

<img src = "https://t1.daumcdn.net/cfile/tistory/99470C445CD83FC034">

Object detection을 하고자 하는 object의 특징들인 occlusion (occ), truncation (trn), bounding-box area (size), aspect ratio (asp), viewpoint (view), part visibility (part) 등의 문제가 있을 때의 성능을 보여준다. 이전의 방법론인 DPM보다는 R-CNN이 더 나은 것을 볼 수 있고, R-CNN 중에서도 fine-tuning, bounding-box regression을 활용했을 때 더 좋은 성능을 보임을 알 수 있다.



### 3.5 Bounding-box regression

bounding box regression은 DPM에 의해 inspire 되었으며, 논문에서는 selective search region proposal에 대한 pool5 특성을 고려해, 새로운 detection window를 예측하기 위해 linear regression 모델을 훈련시킨다.

Table 1,2 그리고 figure 4를 통해 mAP를 3~4% 증가시켰다.

#### (Appendix C : Bounding-box regression)

class-specific detection SVM을 이용해 각 selective search proposal을 scoring한 후 class-specific bounding-box regressor로 새로운 boudning box를 예측한다.

> 나머지 부분은 참고한 논문 리뷰[4] 에서 설명이 자세히 나와있으므로 이를 가져왔다.

> 여기서 N training pairs {(Pi,Gi)}i=1,...,N이고,
>
> Pi=(Pix,Piy,Piw,Pih)로 중앙점 좌표와 width, height를 나타낸다. Ground-truth bounding box는 다음과 같이 표현된다.
>
> Gi=(Gix,Giy,Giw,Gih). 목표는 P를 ground-truth box G에 매핑하는 변형을 학습하는 것이다.
> 변형을 다음의 dx(P),dy(P),dw(P),dh(P) 파라미터화 했다. 처음 두 개는 P의 bounding box의 중심의 scale-invariant 변환을 말하고, 뒤의 두 개는 P의 bounding box의 폭과 높이에 대한 로그 공간 변환을 말한다. 그리고 아래의 식을 이용하여 input proposal P를 변환하여 ground-truth box Ĝ 를 예측할 수 있다.
>
> Gx^=Pwdx(P)+Px....(1)
>
> Gy^=Phdy(P)+Py....(2)
>
> Gw^=Pwexp(dw(P))....(3)
>
> Gh^=Phexp(dh(P))....(4)
>
> d∗(P)=wT∗ϕ(P)에서 ϕ(P)는 P(proposal)의 feature vector이고, w∗ 는 각 함수 d∗(P)에 대한 ridge regression를 통해 학습되어지는 가중치 계수이다.
>
> w∗=argminŵ ∗∑iN(ti∗−wT∗ϕ5(Pi))2+λ||ŵ ∗||2
>
> L2 loss 형태에 regularization이 추가된 모습이다.
>
> regression target t∗는 다음과 같이 정의된다.
>
> tx=(Gx−Px)/Pw
>
> ty=(Gy−Py)/Py
>
> tw=log(Gw/Pw)
>
> th=log(Gh/Ph)
>
> regularization은 중요하다. 만약에 P가 ground-truth boxes와 멀리 있다면 P를 G로 변환하는 것은 말이 안된다. 이는 학습 실패로 이어질 것이다.
>
> 그리고 test-time시에 각 proposal과 predict를 딱 한 번 scoring 한다. iterating이 결과를 증진시키지 않는다는 것을 알았기 때문이다.



### 4.Semantic segmentation

#### CNN features for segmentation

<img src = "https://t1.daumcdn.net/cfile/tistory/99D6503B5CD83FC017">

이 논문에서는 retangular window를 warping한 후, CPMC region에 대한 feature를 계산하는데에 3가지 전략을 가지고 평가했다.

- full : region의 shape을 무시하고 CNN features를 warped window에 바로 연산한다. (detection에 했던 것과 같은 방식으로 )  그러나 이 feature들은 region의 non-rectangular shape를 무시한다. 

- fg : region의 가장 앞쪽 mask에만 CNN feature를 연산한다. 

  이 때, 배경을 평균으로 바꾸게 되는데, 그래야 나중에 평균을 뺐을 때, 배경이 0이 되기 때문이다.

- full + fg : 간단하게 두 가지 방식을 합친 것으로 해석된다.

#### Results on VOC 2011

<img src = "https://t1.daumcdn.net/cfile/tistory/997F55425CD83FC003">

Table.5를 보면 fc6이 항상 fc7보다 성능이 좋다.

이전의 방법론 보다는 full+fg R-CNN fc6이 성능이 더 좋다.



### 6. Conclusion

최근까지 object detection 성능은 정체 되어 있었다. 이 논문은 간단하고 scalable한 object detection 알고리즘을 제시했고, 이전 PASCAL VOC 2012의 결과에 비해 30%의 향상을 이루어냈다. 두 가지 통찰을 통해 성능의 향상이 있다고 생각하는데, 첫번째로는 Region proposal을 이용한 상향식 high-capacity CNN을 적용한 점, 두번째로는  학습 데이터가 부족해도 큰 CNN을 학습할 수 있다는 점이다. 그리고 supervised pre-training and domain-specific fine-tuning이 크게 성능을 높혔다





### 참고문헌

> 1. Girshick, Ross, et al. "**Rich feature hierarchies for accurate object detection and semantic segmentation**." Proceedings of the IEEE conference on computer vision and pattern recognition. 2014. [[pdf\]](http://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf) **(RCNN)** 
> 2. [Rich feature hierarchies for accurate object detection and semantic segmentation] https://arxiv.org/abs/1311.2524
> 3. https://jaehyeongan.github.io/2019/10/10/R-CNN/  재형님의 R-CNN 논문 리뷰
> 4. https://leechamin.tistory.com/211#----%--Detection%--error%--analysis 이차민님의 R-CNN 논문 리뷰
> 5. https://aigong.tistory.com/34  아이공님의 R-CNN 논문 full summary



> 사용을 허락해주신 이차민님과 재형님께 감사의 말씀을 드립니다.

