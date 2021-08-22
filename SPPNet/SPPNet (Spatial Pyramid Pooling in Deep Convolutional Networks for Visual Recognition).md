# SPPNet (Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition)
- - - -
### Abstract
기존 CNN에서의 문제점은 입력 이미지가 고정적이라는 점이었다. AlexNet의 경우 227x227 , VGGNet의 경우 224x224의 크기를 가지는 것이 일반적이었는데, 이런 제약은 인공적이며, 정확도를 낮추는 원인이었다. 이런 제약을 막기위해 spatial pyramid pooling 이라는 SPPNet 구조를 사용한다. 이는 object deformation에 강하다. 

SPPNet의 효과는 전체 image로 부터 한번에 feature map을 계산할 수 있고, fc layer를 위한 고정된 feature map을 생성하기 위해 임의의 영역에서 feature 를 pooling 할 수 있다는 점이다. 또한 R-CNN과 비교했을 때, 24~102배 더 빠른 것을 알 수 있었다.

### 1. Introduction
CNN는 잘 알려진 요구사항이 하나 있는데, 바로 input image가 고정되어야 한다는 점이다. 이는 input image의 크기나 비율을 제한한다. 이전의 방식으로 image를 임의의 크기로 뽑아내기 위해서는 crop이나 warp를 거쳐서 만들었는데 crop의 경우 이미지가 짤리고, warp의 경우 이미지 왜곡이 일어나기  때문에 정확도에 있어서 문제가 발생한다. 

그래서 이 논문의 저자는 "why do CNNs require a fixed input size"라는 질문을 던지게 된다. CNN의 경우 두 가지 part로 나눌 수 있는데, convolution layer와 fc layer다. conv layer의 경우 sliding window 방식을 통해서, spatial arrangement를 출력한다. 이 때는 image가 고정적일 필요가 없다. 그러나 fc layer의 경우 이미지의 크기나 길이가 고정적이어야 한다. 

그래서 이 논문에서는 'Spatial Pyramid Pooling (SPP)' layer를 통해 제약을 없애려고 했다. Figure 1의 그림과 같이 conv layer의 마지막 부분에 SPP layer를 붙여주는 형식이다. 이를 통해 임의의 input 이 들어오더라도 결과적으로 fix된 결과를 반환해주는 것이다.

이렇게 좋은 기능을 가졌지만 CNN의 전체적인 흐름 관점에서 이를 고려하지 않았다고 한다.

SPP의 경우 주목할만한 장점들이 있다.

1. input size가 무엇이든지 관계없이 고정된 output을 생성할 수 있다.
2. SPP는 multi-level spatial bins을 사용하는데, 이는 이전에 쓰인 sliding window 방식에서 쓰이는 하나의 window size를 쓰는 방식과는 차이를 보인다.또한 이는 object deformation에 강하다.
3. SPP는 다양한 scale에 대해 추출된 feature들을 pooling 할 수 있다.

이렇게 다양한 사이즈의 이미지를 훈련시키는 것은 scale-invariance를 늘리고, over-fitting을 줄여준다.

single network에서 다양한 input size를 받아들이기 위해, parameter들을 공유하는 multiple networks에 의해 근사화한다. 반면에 각 networks은 고정된 input size를 사용해 훈련하며, 다음 epoch마다 input size를 바꾼다.

SPP의 이점은 특정 CNN 구조와 직교한다는 점이다. 사실 이 말이 무슨 의미인지 잘 모르겠지만, 이해한대로만 말하자면, 특정 CNN 구조에서 성능 향상이 있다는 것과 같다고 생각한다. 여기서는 4개의 다른 CNN구조에서 향상이 있었는데,  특히 deep하고 large한 복잡한 구조에서 더 향상 된다고 한다.

또한 Object detection part에서도 강점을 가진다. R-CNN에서 deep CNN으로 candidate window를 생성해 정확도가 높은데 반해, time-consuming이라는 시간이 오래걸린다는 문제점을 가진다. 왜냐하면, raw pixel이나 수천개의 warp된 이미지들을 계산할 때마다 CNN을 통과시키는 작업을 반복하기 때문이다. 

반면 이 논문에서는 conv layer를 전체 이미지에 대해 한번만 사용한다. SPPNet based system의 경우, 이로 인해 R-CNN보다 100배 더 빠른 속도를 가진다. 정확도 또한 R-CNN과 같거나 더 나은 정확도를 보인다.

### 2. Deep Networks With Spatial Pyramid Pooling

#### 2.1 Convolutional Layers and Feature Maps
5개의 conv layers와 2개의 fc layer가 있다고 가정하자. conv layer 중 몇몇은 pooling layer가 동반되는데 이 또한 convolution하다고 가정한다. fc layer는 N-way softmax를 output으로 하는 layer이다. 

Conv layer의 경우 sliding filter를 사용해서 response의 정도와 spatial position을 가지면서 input과 같은 aspect ratio를 가지고 있는 것이 특징이다. 

#### 2.2 The Spatial Pyramid Pooling Layer
conv layer의 input이 다양하다는 것은 output 또한 다양한 size로 출력된다는 의미이다. classifier (SVM/Softmax) 나 fc layer의 경우는 fix된 input이 필요한데, Spatial Pyramid Pooling의 경우 한꺼번에 features을 pool하는 방식인 Bag-of-Words의 방식을 local spatial bins으로 pooling을 통해 공간적 정보를 유지할 수 있다는 점에서 향상된 모습을 보여준다.

spatial bin이 이미지의 크기와 비례하고,이미지 크기에 관계없이 bin의 수가 고정된다. 기존 sliding window와는 정반대의 개념이라고 할 수 있다. (sliding window는 input 크기에 따라서 sliding window의 수가 의존적이다.)

임의의 크기인 이미지를 network에 적용하기 위해서, 마지막 pooling layer를 SPPNet으로 대체한다. Figure 3을 보면 이해가 쉬운데, SPP의 output은 kM dimension vector를 가진다. (k : number of filters / M : number of bins) 이 고정된 input은 fc layer의 input으로 들어간다.

이는 임의의 aspect ratio와 scale의 image를 사용가능하게 했다. 또한 이미지를 임의의 크기로 변경하는 것도 가능하다.(ex) min(w,h) = 180,224, ...) SIFT와 같은 전통적인 방식에서도 input의 scale이 중요했듯이, deep CNN에서도 scale은 중요하다 (이는 나중에 밝히는 듯하다)

global average pooling은 model의 size를 줄이고, overfitting을 줄여주는데, test시에 fc layer후에 정확도 향상을 위해 global average pooling을 사용한다.

#### 2.3 Training the Network
이론적으로는 input image size에 관계없이 network를 훈련시키지만, GPU를 사용할 때에는 image size가 고정된 편이 더 효율적이다. 

* Single-size training
image를 crop해서 fixed-size input (224x224)을 만든다. 이는 data augmentation을 위해 사용했다고 한다. 

Spatial pyramid pooling에 필요한 bin size를 미리 계산 할 수 있다. conv layer 5를 지난 feature map의 크기를 axa라고 하자. nxn pyramid level로 sliding window를 수행하면 window size =  ceiling(a_n) 이고 stride = floor(a_n)으로 계산된다. L-level pyramid 라면 fc6 layer는 L output을 concatenate할 것이다.예를 들어 3-level pyramid pooling은 (3x3,2x2,1x1) 와 같은 pooling level을 가지고 있다.
single-size training의 주요 목적은 multi-level pooling을 수행하기 위함이다.

* Multi-size training

 다양한 image size를 다루기 위해서 single size에서와 같이 bin size를 먼저 구해보자. 180x180 or 224x224의 크기가 있는데, 180x180으로 crop하는 것보다 224x224 region을 180x180으로 resize해준다. 이로써, 224x224와 180x180은 단지 해상도만 다르고 내용이나 layout은 같게 된다.
  window size =  ceiling(a_n), stride = floor(a_n) 을  multi-size train에서도 똑같이 적용이 가능하고, 180-network의 SPP layer 출력 값과 224-network의 SPP layer의 출력 값이 동일한데 이는 두 network 모두 동일한 parameter를 가지는 것을 말한다. 즉 다양한 input size를 가진 image가 SPP net을 통과하면 parameter를 공유한다는 것이다.

224-network에서 180-network로 줄이기 위해, 하나의 network를 full epoch로 학습시키고, 이전 학습에서의 weight를 유지한 채로, 다음 학습시에는 network만 교체하고 다시 반복한다. 이를 통해 multi-size training이 single-size training의 결과에 수렴한다.

multi-size training을 사용하는 이유는 다양한 input size를 적용하면서 이미 존재하는 최적화된 fixed-size 구현의 이점을 유지하기 위해서이다. 

이런 single/multi size solution은 training에서만 쓰고, test시에는 어떤 size든 SPPNet 에 바로 적용한다.


### 3. SPP-Net For Image Classification (3장은 제목에 중점을 두자)

#### 3.1 Experiments on ImageNet 2012 Classification

ImageNet 2012 데이터를 사용하고, 이는 1000개의 category가 있는 데이터이다. 

1. image를 smaller dimension을 256으로 resize하고, 이를 224x224로 중심 1개, 모서리 4개 부분을 crop 
2. 여기에 data augmentation을 위해 horizontal flippin과 color 변환을 사용
3. 2개의 fc layer에 Dropout 적용
4. Learning rate : 0.01 / error plateaus 발생 시마다 1/10 적용

#### 3.1.1 Baseline Network Architectures

SPPNet의 장점은 CNN 구조에 독립적이라는 것이다. 기존의 다른 4개의 network 구조에 SPPNet을 적용하면 모든 architecture에서 정확도가 향상 되었다고 한다.
* Four architecture : ZF-5 / Convnet-5 / OverFeat-5/7 
baseline model에서 마지막 conv layer 후 pooling layer는 6x6 feature map을 만들고, 2개의 4096 dimension fc layer 이후 1000-way softmax layer를 연결한다.

#### 3.1.2 Multi-level Pooling Improves Accuracy

train과 test 크기는 둘 다 224x224이다. baseline model의 경우 conv layer는 같은 구조이며, conv layer뒤에 오는 pooling layer를  SPP layer로 대체한다. Table 2에서는 4 level pyramid를 사용하며 {6x6, 3x3, 2x2, 1x1}로 총 50bin이다. 공정한 비교를 위해, 각 view를 224x224로 crop하는 10 view prediction을 사용한다.

multi-level pooling을 하는 것은 더 많은 parameter를 가지기에 간단하지 않다. 그렇기에 더 주목할만한 가치가 있는데, 이는 object deformation과 spatial layout에 robust하다는 장점이 있기 때문이다. 
이를 위해 ZF-5 network를 사용하며 4 level pyramid : {4x4, 3x3, 2x2, 1x1} 로 총 30bin이며 이 network는 SPP를 쓰지않은 ZF-5 network보다 parameter가 더 적다. 결과적으로 SPP를 50bin을 사용한 경우와 top1/ top5 error는 비슷하며, 오히려 no-SPP보다는 더 나은 결과를 보였다.

#### 3.1.3 Multi-size Training Improves Accuracy

multi size training의 경우 training size는 224와 180을 사용한다. 이에반해 testing size는 224로 고정한다. single train때와 마찬가지로 10view prediction을 사용한다. 결과적으로 multi-size training이 single-size training보다 더 좋고, no-SPP보다는 확연히 좋은 것을 확인할 수 있었다.

#### 3.1.4 Full-Image Representations Improve Accuracy

full-image의 정확도를 조사하기 위해 smallest dimension이 256이 되도록, 그러면서 aspect ratio는 유지하도록 resize한다. 정확한 비교를 위해 하나는 full view를 SPP의 input으로 주고, 나머지 하나는 224x224를 center crop을 통해 이를 input으로 넣어준다. 
결국 이는 complete content를 유지하는 것이 중요하다는 것을 강조해주었다.

single full-image보다 multiple view의 조합이 잠재적으로 더 낫다는 것을 보여주지만, full image는 여전히 여러가지 장점을 가진다,
1. 경험적으로 12개의 multiple view(crop) 의 조합에 2개의 full image view (with flipping)가 0.2% 더 boost한다는 것을 보여준다. 이 말인 즉슨, multiple view 보다 full image가 더 나은 결과를 만들어준다는 의미
2. full-image view가 방법론적으로 이전 방식을 유지한다는 점이다. 
3. 이미지 복원과 같은 다른 분야에 적용함에 있어서 similarity ranking에 필요되어진다. (사실 이 부분이 이해가 안간다)

#### 3.1.5 Multi-view Testing on Feature Maps

1. multi view test에서 min(w,h) = s로 resize한다. 이 때, s : predefined scale(multi-view를 위해 미리 계산된 크기) 
2. convolution layer를 통해 entire image에 대해 계산. (filp된 이미지에 대해서도 똑같이 계산한다. )
3. SPP를 이용해 features을 pooling 한다.
4. fc layer를 통과시키고 softmax score를 계산한다.
5. 이후 score 평균을 낸다.
(10 view prediction사용, s = 256, 224x224 window를 사용한다.)

### 4. SPP-Net For Object Detection

R-CNN 방식을 요약하자면 selevtive search 방식으로 2000개의 candidate window를 뽑아내고, 이미지들을 fix된 크기(227x227)로 warp해준다. pre-train된 network를 통해 feature를 뽑아내고 이를 binary SVM에 넣어 훈련시킨다.
여기서 문제는 Abstract에서 이야기 했듯이 time-consuming으로 인해 문제가 되고 , Feature extraction은 time 병목현상의 중요 원인입니다.

이에 반해 SPPNet을 사용하면 전체 이미지를 한번에 feature map을 추출할 수 있으며, 각 candidate-window에 SPP를 적용한다 time-consuming convolution을 한번만 적용하기에, 속도가 더 빨라진다.

SPPNet은 feature map의 region으로부터 window-wise feature를 추출한다. 반면 R-CNN은 image region에서부터 직접 구한다. 
OverFeat detection의 경우 pre-define된 winndow-size를 구해야 했지만, SPPNet 경우 임의의 window에서 feature extraction이 가능하게 한다.  



**SPPNet 장점**

 (1) SPPnet은 CNN을 이미지에 한 번만 적용하여 속도가 빠르다.

 (2) CNN으로 생성된 feature map에서 selective search를 적용하여 생성된 window에서 특징을 추출한다.

 (3) feature map에서 임의의 크기 window로 특징 추출이 가능하다.

 (4) 입력 이미지의 scale, size, aspect ratio에 영향을 받지 않는다.



**SPPNet  동작 방식**

(1) Selective Search를 사용하여 약 2000개의 region proposals를 생성한다.

(2) 이미지를 CNN에 통과시켜 feature map을 얻는다.

(3) 각 region proposal로 경계가 제한된 feature map을 SPP layer에 전달한다.

(4) SPP layer를 적용하여 얻은 고정된 벡터 크기(representation)를 FC layer에 전달한다.

(5) SVM으로 카테고리를 분류한다.

(6) Bounding box regression으로 bounding box 크기를 조정하고 non-maximum suppression을 사용하여 최종 bounding box를 계산한다. 



#### 4.1 Detection Algorithm

거두절미하고 이 논문의 방식은 multi-scale feature extraction에 의해 성능이 향상될 수 있는데, 먼저 min(w,h) = s : {480,576,688,864,1200} 으로 resize한다. 이후 convolution layer 5개를 지나면서 feature map을 계산한다. 
한가지 전략은 feature들을 모아서 channel-by-channel로 pooling 하는 방식을 처음에 제안했는데, 이는 경험적으로 두번째 방식이 더 나은 결과를 만들었기에 두번째 방식으로 넘어가도록 하자.
두번째 방식은 각 candidate-window마다 s를 하나 선택하고,해당 크기로 scale된 candidate window는 224x224에 가까운 pixel의 수를 가지게 된다. 그 다음, 해당 window의 feature를 추출하기 위해 해당 크기로 추출된 feature map만을 사용한다. 즉, 이 방식은 candidate window의 개수에 관계없이,전체 image에서 한번만 해당 scale에 대해 feature map을 추출하는 것만 필요하다. 

(내가 이해한 방식대로만 이야기하면 결국 R-CNN과 SPP-Net의 다른 점은 feature를 추출 시에, 전체이미지로 부터 feature를 가져오는가, 아니면 feature map으로부터 Selective search 방식을 통한 window로부터 가져오는가 인데, 여기서 말하는건 feature map에서 임의의 window로 특징 추출이 가능하며, 그 window를 통해 해당 feature를 구하는 것이라 생각했으며, 전체이미지를 계속 사용하는 것이 아닌, 임의의 크기 window로 특징을 추출 할 수 있는 점이 좋다는 것을 강조하고 싶은 것 같았다. ) 

* pre-train된 network 사용. 이로 인해 앞부분의 feature extractor는 pre train 되어있으므로, 나머지 fc layer에 대해서만 fine-tune 했다.
* conv5이후 fixed-length이므로, fc6,7은 21-way (one extra negative category) + fc8 layer로 구성되어있다.
* fc8 layer의 weight는 gaussian distribution으로 표준편차가 0.01로 초기화
* learning rate를 1e-4로 고정, 마지막 3 layer에 대해서는 1e-5로 조정
* fine-tuning 시에, 50% 이상 ground truth와 overlap시에 positive sample, 10%보다 크고 50% 미만인 경우 negative sample로 정의
* 오직 fc layer를 fine-tune 하는 것이기에 250k mini-batch에서는 learning rate = 1e-4 / 50k mini-batch에서는 learning rate = 1e-5

bounding box regression에도 적용할 수 있는데, conv5를 통과한 feature 를 pooling 하고 나온 windows가 ground truth와 최소한 50% 이상 overlap되어야 regression training에 사용했다.

### 5. Conclusion

이는 논문을 그대로 옮기는 게 더 이해가 쉬운 것 같아 논문을 발췌해 올립니다.

<left><img src = "https://user-images.githubusercontent.com/78463348/130343326-e3ea2aa0-3b93-4de1-bacd-f8c9180ad233.png">



> 참고 자료
>
> 1. 논문 : https://arxiv.org/abs/1406.4729
> 2.  AI 꿈나무 님의 SPPNet 논문 리뷰 https://deep-learning-study.tistory.com/445

