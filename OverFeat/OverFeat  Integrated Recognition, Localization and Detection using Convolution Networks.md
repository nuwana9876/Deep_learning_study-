# OverFeat : Integrated Recognition, Localization and Detection using Convolution Networks

이번에는 1-stage-detector의 영향을 준 'OverFeat' 라는 논문으로 사실상 R-CNN보다는 성능이 낮다는 이야기가 있지만 중요한 논문이라고 생각한다. 

### Abstract

이 논문은 classification, localization, detection을 계산하기위해 CNN을 사용한 통합 framework를 제시한다. 이 framework는 convNet으로 구현한 multiscale , sliding window approach가 얼마나 효율적인지 보여준다.Bounding box는 detection confidence를 증가시키기 위해 억제하기보다는 축적했다. Different task (=classification, localization, detection)는 single shared network를 사용해 한번에 학습 시킬 수 있다. 우리는 이를 OverFeat라고 불리는 feature extractor로써 발표했다.

### 1. Introduction

이 논문의 주요 point는 image에서 classify, locate and detect를 한번에 학습하는 것은 classification accuracy와 detection, location accuracy를 boost할 수 있다는 점이다.  이 논문의 목적은 single ConvNet으로 위의 task들을 통합한 approach를 제시하는 것이다. 또한 예측된 bounding box들을 축적하는 것으로, localization과 detection을 위한 novel approach를 소개한다. 위의 두가지 task를 수행할 때, background sample을 train에 포함하지 않으므로써, time -consuming과 복잡한 bootstrapping (= 예측값을 가지고 또다른 예측을 하는 것) 학습을 피할 수 있었으며, 전적으로 positive class에 집중하게 되므로 높은 정확도를 보였다.

이하 생략.

### 2. Vision Tasks

이 논문에서, 3개의 computer vision task를 다룬다 : (i) classification (ii) localization (iii) detection

모든 task들은 single framework와 shared feature learning base를 사용해 한꺼번에 다루지만, 이 논문에서는 각 section에 따라 나누어서 설명한다.

이 논문 전반에 걸쳐, ILSVRC 2013 대회 결과를 보여주는데, classification task에서는 각 이미지는 image의 main object에 대한 single label을 할당 받는데, multiple unlabeled image가 있을 수 있기에 정답에 가까운 5개의 class에 대해 보여준다.

Localization task에서는 classification과 비슷하게, 이미지당 5개의 guess를 나타내는데, 각 guess당 예측된 object에 대한 bounding box를 반환받는다. 예측된 box가 ground truth와 최소한 50%이상 같을 경우 정답으로 다뤘다. Localization task는 classification과 detection 사이에 편리한 중간 과정이며, detection task와 독립적으로 localization 방식을 평가할 수 있다

Detection task에서는 localization과 다르게 각 이미지당 object가 없을 수도 있고 여러개일 수도 있다.(localization의 경우 1개 이상 있어야 하는 것으로 알고있다.) 그리고 false positive는 mAP에 의해  penalize되어진다.

classification과 localization은 같은 dataset을 공유하며, 반면 detection은 object가 더 작아질 수 있는 추가적인 data를 포함한다.~~또한 detection data는 특정 object들이 없는 이미지 set을 포함한다. 이는 bootstrapping에 사용되나 이 논문에서는 그것을 사용하지 않는다.~~ 

### 3. Classification
OverFeat의 classification의 구조는 AlexNet에서의 구조와 비슷하지만 network design과 추론 과정이 더 향상된 점이 특징이다. 이는 시간 제한 때문이며, AlexNet 구조에서 학습한 특징들 중 몇몇은 탐색되지 않았는데, 그로인해 결과가 더 향상될 수 있었다고 한다.
#### 3.1 Model Design and Training
* Training data : ImageNet 2012 dataset
* smallest dimension이 256 pixels이 되도록 downsampling
* 5개의 random crops & horizontal flips (size : 221x221 pixels) : 10배로 data augmentation
* mini-batch : 128
* weight initialization : (평균, 분산 = 0, 0.1x10^ (-2)) 인 random initialize
* update by SGD & momentum : 0.6
* L2 weight decay : 1x10^(-5)
* Learning rate : 5x10^(-2) & decresed by a factor of 0.5 after (30,50,60,70,80)
* Dropout rate : 0.5 (employ at fc layer 6th and 7th)

Training시에 우리는 이 구조를 non-spatial(output map : 1x1)로 다루지만  inference step에서의 경우 spatial outputs을 생성한다. (이게 무슨 의미인ㄱ에 대해 생각한 결과, 훈련시에는 fully-connected layer를 적용하면서 non-spatial한 결과를 생성하는데 반해, 예측시에는 fc layer를 1x1 convolution layer로 대체해서 생각하므로써 spatial한 결과를 만들어 냈다.) layer 1~5의 경우 AlexNet과 비슷하며 activation funtion으로 ReLU함수를 사용하고 max-pooling을 이용한다.
차이점이라고 하면 3가지가 있는데, (i) contrast normalization을 사용하지 않는다 (contract normalization이란 이미지마다 밝기,찍은 환경이 다를텐데 이를 맞춰주기 위해 표준화를 진행하는 것을 말한다) (ii) AlexNet에서는 pooling을 할 때, overlapping을 사용했지만, OverFeat의 경우 non-overlapping이다.(iii) stride가 작기에 첫번째와 두번째 layer의 feature map이 크다 첫번째 layer의 경우 방향 edge, 패턴, 얼룩을 잡아내고, 두번째 layer에서는 형태의 다양성을 잡아내며, 나머지는 강한 선 구조나 방향 edge를 잡아낸다.

#### 3.2 Feature Extractor
feature extractor의 이름이 "OverFeat"이며, fast와 accuracy 모델로 나뉘어져있다. (두배로 많은 connection을 선택할 지(더 정확한 정확도), 더 빠른 정확도 계산을 선택할 건지에 따라 나뉘어진다고 보면 될 것 같다.)
* fast : 5 conv / pooling layer & 3 fc layer
* accuracy : 6 conv / pooling layer & 3 fc layer

#### 3.3 Multi-scale classification
첫 문단에서 말하고자 하는 바는 결국 fc layer를 conv layer로 대체한 효과에 대해 이야기하는 것으로 보인다. fc layer의 경우 input과 output의 출력이 고정되어 size가 fix된 점에 비해, conv layer의 경우 input에 따라 output이 가변적으로 변할 수 있기에 좋으며, 또한 convolution layer의 경우 겹치는 계산에 대해 공유하기에 불필요한 계산을 막아주는 효과를 가져온다.
또한 conv layer로 인한 가변성으로 인해 multi-scale에 대해 densely running network가 가능하고, sliding window에 의해 효율은 그대로면서, robustness를 향상시키는 결과를 가져오면서, 위에서 말한 spatial한 map을 가져오게 되는 것이다.
여기서 문제가 발생한다. total sampling ratio의 경우 2x3x2x3으로 총 36이 되는데, 즉 output의 1x1 pixel이  input의 36 pixel을 encode하는 것이며 이 영역을 receptive field라고 합니다. 이처럼 포괄적인 resolution은 10-view scheme에 비해 성능이 감소하는데 이유는 "network window가 이미지 내의 물체와 정렬이 잘 되어있지 않다"라고 말합니다. (사실상 이해 불가) (Fast Image Scanning 논문을 참조하자) 이 문제를 회피하기 위해 논문에서와 같은 방식을 적용하며 이를 통해 x36이 아닌 x12로 resolution loss가 줄어들게 된다.

이제 resolution augmentation이 어떻게 수행되었는가에 대해 살펴보자
(Figure 3를 보면 조금 더 이해하기 쉽다.)

* input으로 6 scale 이미지를 사용했으며 layer 5개를 통과함에 따라 unpooled  layer5 maps을 가져오게 된다. 
* unpooled map들을 3x3 max pooling 을 적용한다(non-overlapping) (repeated 3x3 times for pixel offsets of  {0,1,2})
* classifier(layer 6,7,8)은 5x5의 fixed input size를 가진다. Sliding window 를 적용해서 C-dimensional output map을 만든다.
* reshape the output maps into single 3D output map 

여기서 잠깐 Fast Image Scanning 논문에서 다룬 내용과 이 논문에서 말하고자 하는 바가 무엇인지 알아보도록 하자
Fast Image Scanning 논문 p2의 2.2.2 max-pooling layers 부분을 보면 어떻게 계산되는지를 볼 수 있다. 
이 논문에서 말하고자 하는 방향은 dense evaluation을 통해 더 세밀한 계산을 할 수 있다는 것인데,  Fast Image Scanning 논문에서 사용한 기법을 보면, 원래의 max-pooling의 경우와 다르게 모든 pixel에 대해서 동등하게 max-pooling될 기회를 줌으로써, 자세한 계산이 가능해진다는 점이었다. 예로 그림을 보면 , 6x6 input에서 Fast Image Scanning 저자가 제안한 방식으로 max-pooling을 계산하면 4개의 output Fragments가 나오게 되고, 이와 비슷하게 이 논문의 저자는 offset = {0,1,2}로 두고, 이 만큼 shifting을 통해 Fragments를 얻고 이를 sliding window 방식을 적용해서 입력이 fix된 fc layer에 넣어 나온 값을 concatenate해서 조밀한 출력을 얻겠다는 것이 이 논문의 방식이라고 이해했다.  Figure 3에서 (a)에서 unpooled map이 20개의 pixel로 구성되어 있고 이를 3x1의 non-overlapping pooling을 적용하면 6 pixel pooled map이 결과로 나온다. 이로 인해 1/3로  resolution이 줄고, 위치에 대한 정보도 그만큼 줄어드는 것을 볼 수 있다. 이를  offset = {0,1,2} 마다 classifier를 적용하고 결과를 하나로 합치게 되면 더 조밀한 출력을 얻을 수 있었다. 이를 horizontally flipped 된 image에도 그대로 적용해준다.

초기 단계에서 network는 두 개의 부분으로 나뉠 수 있는데, (i) Feature extraction layers(1-5) (ii) classifier layer (6-output) 이다. 두 부분은 정반대의 방식으로 사용되는데, (i)의 경우 전체 이미지를 한번에 convolution한다. 계산 관점에서 fixed-size feature extractor에 sliding방식 보다 훨씬 더 효율적이며 다른 location으로부터 결과를 모은다. 반면 (ii)의 경우 layer5로부터 나온 different positions and scales feature map으로부터 fixed size를 얻는다. 위의 경우 5x5 input으로 고정 feature map을 만든다.

#### 3.4 Results

결과적으로 이 방식을 VGGNet single network와 비교했을 때, top-5 error rate 13.6% < top-5 error rate 16.97% 로  small improvement를 이뤘다.

#### 3.5 ConvNets and Sliding Window Efficiency
Input의 각 window마다 전체 pipeline을 계산하는 sliding-window 방식과는 다르게, ConvNet은 겹치는 영역에 대해 공통된 계산을 공유하기 때문에 효율적이다. 논문의 Figure5를 보면 윗 줄과 아랫줄의 그림을 볼 수 있는데, 서로 겹치는 영역에 대해서는 계산을 더 할 필요가 없고, 아래 larger image에서 겹치지 않는 부분만 학습하면 된다고 이해했다. train에서의 경우, last layer에서 fully connected layer를 적용했다면, test에서의 경우 1x1 convolution layer를 적용한다.

### 4.Localization
classifier를 regressor로 대체한다. Classification과 마찬가지로 앞 부분은 같은 feature extraction layer를 사용하기에 오직 final regression layer만 학습하면 된다. Classification의 경우 결과적으로 class c에 대한 score of confidence를 계산한다면 Regression의 경우 confidence to each bounding box를 계산한다.

#### 4.1 Generating Predictions
4.Localization에서 말한 이유로 인해, bounding box 예측을 위해 classifier 계산 후에 final regression layer를 계산하면 된다. 

#### 4.2 Regressor Training
regression network의 input은 pooling된 feature map일텐데, 이를 hidden layer의 크기가 4096, 1024인 2개의 fc layer를 통과 후, output layer가 4 unit인 network를 가진다. 
* L2 loss로 regression layer를 train
* final regression layer는 1000개의 class-specific하다.
* bounding box와 비교했을 때, 50%보다 더 적게 overlap된다면 이는 train에 포함하지 않는다.

- single scale 방식으로 training 하는 것은 해당 scale에 대해서 잘 학습하고 해당 scale이 아니어도 어느정도 학습을 잘 해낸다. 그러나 multi-scale 방식으로 학습할 경우 전체 scale에 대해 정확하게 예측하며, 병합된 예측의 정확도를 exponentially 증가시킨다. 결과적으로, detection시에 많은 scales 보다 몇몇  scales에 더 잘 수행하도록 해준다.

#### 4.3 Combining Predictions
Figure 7에서 regressor bounding box에 greedy merge strategy를 적용한다.

1. Cs에 해당 scale의 spatial output에 대해 각 pixel에서 가장 높은 confidence score를 가지는 class를 해당 location에 할당한다.
2. Bs에 해당 scale의 spatial output에 bounding box 좌표를 할당한다
3. B에 Bs를 할당한다
4. 결과가 나오기 전까지 아래 병합 과정을 반복한다
    i) B에서 b1,b2를 뽑아서 match_score를 적용한 후 가장 작은 b1,b2를 b1*,b2 * *에 할당한다
    ii) 만약 match_score(b1 *,b2 *) >t 이면 멈춘다
    iii) 그렇지 않으면, B에 box_Merge(b1 * b2 *)를 b1,b2 대신에 넣는다

### 5. Detection

Detection은 classification과 비슷하게 training 하지만 spatial한 방식이라는 점에서 다르다.
Localization과 주된 차이점은 이미지에 아무 물체가 존재하지 않을 때, background class에 대해 예측이 필요하다는 점이다. Selective Search 방식과 같이 negative example을 training에 포함시키는데, 이를 bootstrapping pass에 넣는다.  training에서 small set에 대해 overfit하지 않도록 bootstrapping pass의 size가 조정될 필요가 있다. 이 문제를 해결하기 위해, random하게나 제일 상이한 negative examples을 선택함으로써, 즉석에서 negative training을 수행한다. 이 방식은 계산적으로 비싸지만, 더 간단한 과정이 된다. feature extraction은 classification에서 train되기 때문에, detection fine-tuning은 그리 오래 걸리지 않는다.

Figure 11을 보면 top 2 system과 차이가 조금 나는데, 이는 위 2개의 방식이 initial segmentation방식을 써서, 대략적으로 200000에서 2000으로 candidate window를 줄였기 때문에 OverFeat보다 성능이 높다. 이 방식은 추론과정에서 속도가 빠르며, 잠재적으로 false positive의 수를 줄여준다. 
selective search와는 반대로 dense sliding window (이는 객체의 위치와 비슷하지 않으면 이를 버리므로써 false positive를 줄이는 방식)를 사용하면 detection accuracy가 줄어든다. 
이 방식을 OverFeat에 접목시켰다면 비슷한 향상을 볼 수 있었을지도 모른다. 

### 6.Discussion

이 논문의 방식은 몇몇 방식들을 통해 향상될 지 모른다.

1. Localization에서 전체 network를 통해 back-proping하지 않는 방식
2. 최적화한 IOU 손실함수 대신에 L2loss를 사용하는 방식 (IOU와 L2loss를 바꾸는 것은 IOU가 미분가능하기에 바꿀 수 있다.)
3. bounding box를 대체 매개변수화 하는 것은 output에 비상관화하게 만들어 주는데 이는 network 학습에 도움이 된다.
