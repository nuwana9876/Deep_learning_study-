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

위와 같은 문제들을 해결하는 근본적인 방식은 convoltion 내부를 fully connected architecture를 sparsely connected architecture로 바꾸는 것이다. Arora의 연구로부터 견고한 이론적 토대를 얻을 수 있는 장점이 있다.

> [Arora의 연구](https://arxiv.org/pdf/1310.6343.pdf)는 **dataset의 확률 분포를 sparse하고 큰 네트워크로 나타낼 수 있다면, 선행 layer activation들의 correlation statistic 분석 및 highly correlated output을 가진 뉴런들을 clustering 함으로써, 최적화 된 네트워크를 구성할 수 있다는 내용이라 한다**.  **(=> sparse하면서 deep neural network를 표현할 수 있다면 optimal network를 만들수 있다.)** 이에는 엄격한 condition이 따라야 하지만, Hebbian principle을 참조하면 실제로는 덜 엄격한 condition 하에서도 적용할 수 있다고 한다.
>
> 즉, sparse한 경우에는 대부분의 노드가 비활성화 되고 일부만 활성화 될텐데, 이 때 엄격한 condition에 따라 상관관계를 분석함으로써 활성화 될 노드를 정한다면 최적화 된 네트워크가 구성될 것이라는 말이다.
>
> (출처 : https://sike6054.github.io/blog/paper/second-post/)

그러나 오늘날 컴퓨터 기반환경은 불균일한 sparse data 구조에서 수행되는 계산에 대해서는 매우 비효율적이다.

현재의 수치계산은 fully-connected architectures에 효율적이기 때문에, sparesely connected architectures을 사용하는 것을 비효율 적이다. 이를 위해 Inception module을 적용해 sparese하게 하면서 내부는 dense하도록 만들었다.  

> arithmetic operation을 100배 줄였음에도 불구하고, lookup, cache misses의 overhead가 sparse matrice로 바꾼 것을 상쇄시킬만한 역효과를 낳는다. 이 차이는 더욱 커졌는데, CPU나 GPU에 최적화 된 numerical library가 개발되면서 dense matrix multiplication의 고속 연산이 가능해졌고, 이에 따라 operation의 수가 줄어듦으로 인한 이득이 점점 감소했다.

현재 머신러닝 체계가 지향하는 비전의 대부분은 convolution을 사용하는 것만으로 공간적 domain에서 sparsity를 활용한다. 그러나 convolution은 이전 layer의 patch에 대한 dense connection의 집합으로 구현된다. 

ConvNet은 대칭성을 깨고 학습을 향상시키기 위해서 LeNet 이후로 feature dimension에서 random하고 sparse한 connection을 사용했지만 더 나은 병렬 계산의 최적화를 위해 full connection으로 다시 트렌드가 변화했다. 균일한 모델 구조와 많은 filter의 수, 그리고 더 큰 batch size는 효울적인 dense computation을 활용하는 것을 가능하게 했다.

>이 밑 부분에서 4. Architectural Details 전까지의 설명은 제가 설명하는 것보다 Aroddary 님의 설명이 더욱 이해가 잘 되는 관계로 내용을 그대로 가져왔습니다.

Dense matrix 연산에 적합한 하드웨어를 활용한다는 조건 하에서, 위의 이론처럼 filter-level과 같은 중간 단계에서 sparsity을 이용할 방법이 있는가 하는 의문이 든다.

[연구1](https://graal.ens-lyon.fr/~bucar/papers/ucca2D.pdf)과 같이 sparse matrix의 계산에 대한 수많은 연구들은, sparse matrix를 상대적으로 densely한 submatrix로 clustering 하는 방법이 sparse matrix multiplication에서 쓸만한 성능을 보였다고 한다. GoogLeNet 저자는 이 연구들을 두고, 가까운 미래에 non-uniform deep learning architecture의 자동화 기법에 이와 유사한 방법이 활용 될 가능성이 있을거라 생각했다고 한다.

Inception architecture는 [Arora의 연구](https://arxiv.org/pdf/1310.6343.pdf)에서 말한 sparse structure에 대한 근사화를 포함해, dense하면서도 쉽게 사용할 수 있도록 정교하게 설계된 network topology construction 알고리즘을 평가하기 위한 사례 연구로 시작됐다.

상당한 추측에 근거한 연구였지만, [NIN](https://arxiv.org/pdf/1312.4400.pdf)에 기반하여 구현된 기준 네트워크와 비교했을 때, 이른 시점에서 약간의 성능 향상이 관찰됐다. 이는 약간의 튜닝으로도 gap이 넓어졌으며, Inception이 [R-CNN](https://arxiv.org/pdf/1311.2524.pdf)과 [Scalable object detection](https://arxiv.org/pdf/1312.2249.pdf)에서 base network로 사용되는 경우, localization과 object detection 분야에서 성능 향상 효과가 있다는 것이 증명되었다.


그럼에도 신중하게 접근할 필요가 있다. Inception architecture가 computer vision에서 성공적으로 자리 잡았음에도 불구하고, 이 architecture가 원래 의도했던 원칙에 의해 얻어진 구조인지에 대해서는 미심쩍은 부분이 있기 때문이다. 이를 확인하기 위해서는 훨씬 더 철저한 분석과 검증이 필요하다.



### 4. Architectural Details

Inception 구조의 주요 아이디어는 

1. CNN에서 최적의 local sparse structure를 근사화하는 것
2. 이를 쉽게 이용가능한 dense components로 구성하는 것

우리에게 필요한 것은 최적의 local 구성 요소를 찾고 이를 공간적으로 반복하면 된다. Arora 의 연구는 layer-by-layer construction이라는 방식을 제안하는데, 이는 마지막 layer의 correlation statistics를 분석하고 이를 high correlation을 가진 unit들로 clustering 하는 방식이다. 이 cluster들은 다음 layer의 unit으로 구성되고 이는 이전 layer의 unit으로 연결된다.

이전 layer의 각 unit은 입력 image의 일부 영역에 해당하며 이 unit들은 filter bank로 그룹화 된다고 가정한다. 즉 입력이미지와 가까운 lower layer에서는 local region (특정 영역)에 correlated unit들이 집중되어있다. 이는 무엇을 의미하냐면, **한 지역에 많은 클러스터들이 집중되어 있고, 이 클러스터들을 Network in network에서 제안했듯이, 다음 layer에서의 1x1 convolution layer로 처리 할 수 있다는 것을 의미한다.** 

**그러나 어떤 상황에서는 큰 patch size의 convolution을 사용해야 적은 수의 cluster로도 더 공간적으로 넓게 펼쳐질 수 있고, 이를 통해 patch의 수가 감소 될 수도 있다. (patch == filter). 이로 인해 correlated unit을 더 많이 뽑아낼 수 있는 것이다.****또한 patch-alignment issue를 피하기 위해 현재 Inception 구조의 실현은 filter size 1x1, 3x3, 5x5 filter size로 제한된다. 그러나 이 결정은 필요성 보다는 편의성에 더 중점을 맞춘 결정이었다.** 즉 제안된 구조는 **이러한 layer들의 output filter bank 조합이라는 것을 의미한다. 이는 다음 입력으로 연결 될 single output vector가 된다.** **( => 이 부분이 Inception 구조에서 다양한 size의 filter를 병렬로 넣고 single output vector로 다음 layer에 전달한다는 말인 것 같다.)** 

**또한 현재 성공적인 convolution network에서의 성공을 위해, pooling operation은 필수적인데, 이는 각 단계에서 alternative parallel pooling path를 추가하는 것이 추가적인 이점을 갖는다는 것을 암시한다.**

1x1, 3x3, 5x5 Convolution filter의 수는 망이 깊어짐에 따라 달라지는데, 높은 layer에서만 포착될 수 있는 높은 추상적인 특징이 있다면, 이는 공간적 집중도는 감소할 것으로 예상되기에, **높은 layer로 이동함에 따라, 3x3, 5x5 convolution layer의 비율이 증가해야한다는 것을 말한다.**

<img src = "https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fby996R%2FbtqSAR4zKGf%2FTkPxN652gdxzEJu1YtzWh1%2Fimg.png">

위 모델에서 한가지 큰 문제점이 발생하는데, 위의 figure의 (a)에서와 같이 naive한 형태(적은 수) 나 심지어 적당한 수의 5x5 convolution이어도 사용하게 되면 연산량이 많아지게 되는데 입력 feature map의 크기가 크거나 5x5 convolution layer filter의 수가 많아지면 연산량은 더욱 더 많아지게 된다. 이 문제는 pooling unit을 추가해 mix하면 문제는 더욱 저명하게 나타나게 된다.

> *Figure를 보면 Inception module은 convolution 결과들을 concatenation으로 연결하기 때문에, output 채널의 수가 많아진다는 것을 알 수 있다. 여기에 output의 채널 수가 input과 같은 pooling unit이 추가되면, 단계가 거듭 될수록 채널이 이전 layer의 2배 이상 누적되므로 치명적일 수 있다.*



이 문제를 해결하기 위해서 두번째 architecture를 제안하게 되었다. 계산량이 너무 많이 증가하게 된다면, 현명하게 차원 축소를 적용하는 것이 목적이며, 성공적인 embedding에 기반을 두었다. 저차원의 embedding은 상대적으로 큰 image patch에 대한 많은 정보를 담고 있다. 하지만 이런 embedding의 경우는 정보가 고밀도로 압축된 형태이며, 압축된 정보는 처리하기 더욱 어려워진다. 

Arora의 연구의 조건을 만족하기 위해, 대부분의 위치에서 sparse하도록 유지해야하며,오직 총 집합해야 할 때마다 신호를 압축해야 한다. 즉, 1x1 convolution은 3x3, 5x5 convolution 전에 계산량을 감소시키기 위해 사용한다. 이 때 ReLU를 activation으로 사용하면서 두 가지의 목적을 가진다.

일반적으로 Inception network는 상기 타입의 module이 서로 쌓여서 구성된 네트워크인데, 가끔 feature map의 크기를 반으로 줄이기 위해 stride가 2인 max pooling layer를 사용한다. training시에 메모리 효율과 같은 기술적인 이유로 인해, 상위 layer에서만 inception module를 사용하고 하위 layer의 경우는 전통적인 convolution network를 유지하는 방법이 효율적이다. 물로 이는 필수적인 것이 아니라 비효율 적인 인프라를 고려한 것일뿐이다.

이 구조의 주요 장점은 계산 복잡도의 통제 불가능한 폭발적인 증가없이 각 단계에서 unit의 수를 증가하는 것이 가능하게 되었다. 이는 더 큰 filter size를 갖는 convolution 전에, 이전 layer의 많은 수의 input filter가 다음 layer로 넘어가는 것을 막기 위해 그들의 차원을 축소시켜면서 위와 같은 장점이 가능했다.

또 다른 실질적인 유용한 측면은  시각적 정보를 다양한 scale 상에서 처리된 후에 종합함으로써, 다음 stage에서 서로 다른 scale로부터 feature를 동시에 추상화 한다는 타당한 직관을 따른다. (=> 1x1,3x3,5x5 convolution을 한꺼번에 처리하는 것이 장점이라는 이야기이다)

Inception 구조로 인한 계산 자원의 효율적인 사용이 가능해지면서 각 단계에서 width가 늘어나는 것 뿐만 아니라, depth 또한 늘어나는 것이 가능해졌다. inception 구조의 다른 활용은 약간 성능은 떨어지지만, 비용을 저렴하게 만들어 주는 것 또한 가능하다. 



### 5. GoogLeNet

<img src = "https://sike6054.github.io/blog/images/GoogLeNet,%20Table.1(removed).png">

> **Table.1**
> *ILSVRC 2014 competition에 제출됐던 GoogLeNet의 구조이다. “#3x3 reduce”와 “#5x5 reduce”는 각각 3x3, 5x5 convolution 전에 사용 된 reduction layer의 1x1 filter 개수를 나타낸다. Inception 모듈에서의 “pool proj” 열 값은 max-pooling 뒤에 따라오는 projection layer의 1x1 filter 개수를 나타낸다. *



GoogLeNet은 ILSVRC 2014 competition에 제출한 Inception 구조의 특정 형태를 말한다. 저자는 더 깊고 널븐 Inception network를 사용했는데,  이 quality는 조금 낮지만, ensemble 기법을 사용해서 약간 성능을 향상시켰다. 경험에 따르면, 정확한 architectural parameter의 영향이 상대적으로 미미하기 때문에, detail한 network 설명은 생략했다. Table 1은 가장 일반적인 구조를 나타낸다. image sampling method에 따라 학습되었다는 점만 다르고 나머지는 같은 네트워크 연결방식을 사용한 이 네트워크는 ensemble에서 7개 모델 중 6개에 사용되었다. 

GoogLeNet은 Inception module 내부의 reduction / projection layer를 포함한 모든 convolution에는 ReLU를 사용했다. 네트워크의 receptive field의 크기는 224x224 이며 이는 평균을 뺀 RGB channel을 가진다.

이 네트워크는 계산적 효율성과 실용성을 고안해 만들었기에, 특히 메모리가 적은 경우를 포함해 제한적인 conputational resource를 가진 device에서도 inference를 할 수 있다. 

GoogLeNet을 나누어 살펴보면 총 4가지 부분으로 생각 할 수 있다.

Part 1.

<img src = "https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbiTAD1%2FbtqSELQkwgt%2FkJPKmvCoM19ph9Jon1zGb0%2Fimg.png">



Part 1은 **입력 이미지와 가까운 낮은 레이어가 위치해 있는 부분**이다.

이는 위에서 설명했듯이 **효율적인 메모리 사용을 위해 낮은 layer에서는 기본적인 CNN 모델을 적용**하고, 높은 layer에서 Inception module을 사용하라고 하였기에 Inception module이 사용되지 않은 것을 볼 수 있다.

**Part 2**



![img](https://blog.kakaocdn.net/dn/bTGs3P/btqSxBVdZ5u/3cyfHeBiEGEScc5zdpYnl1/img.png)



Part 2는 **Inception module**로서 **다양한 특징을 추출하기 위해 1 x 1, 3 x 3, 5 x 5 Convolutional layer가 병렬적으로 연산을 수행**하고 있으며, **차원을 축소하여 연산량을 줄이기 위해 1 x 1 Convolutional layer가 적용**되어 있는 것을 확인할 수 있다.

**Part 3**

 



![img](https://blog.kakaocdn.net/dn/b0wn0v/btqSmAiDErs/oTQTSFOERoyEwPDs2u0alk/img.png)



Part 3는 **auxiliary classifier가 적용된 부분**이다.

모델의 **깊이가 매우 깊을 경우, 기울기가 0으로 수렴하는 gradient vanishing 문제가 발생**할 수 있다. 이때, 상대적으로 얕은 신경망의 강한 성능을 통해 신경망의 중간 layer에서 생성된 특징이 매우 차별적이라는 것을 알 수 있다. 따라서 **중간 layer에 auxiliary classifier를 추가하여, 중간중간에 결과를 출력해 추가적인 역전파를 일으켜 gradient가 전달될** 수 있게끔 하면서도 **정규화 효과**가 나타나도록 하였다.

추가로, 지나치게 영향을 주는 것을 막기 위해 **auxiliary classifier의 loss에 0.3을 곱**하였고, **실제 테스트 시에는 auxiliary classifier를 제거** 후, 제일 끝단의 softmax만을 사용하였다.

 

**Part 4**

Part 4는 **예측 결과가 나오는 모델의 끝 부분**이다.

여기서 **최종 Classifier 이전에 average pooling layer를 사용**하고 있는데 이는 **GAP** (Global Average Pooling)가 적용된 것으로 **이전 layer에서 추출된 feature map을 각각 평균 낸 것을 이어 1차원 벡터로 만들어 준다.** 이는 1차원 벡터로 만들어줘야 최종적으로 이미지 분류를 위한 softmax layer와 연결할 수 있기 때문이다.

<img src = "https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FevD2RB%2FbtqSATBmCEB%2FpkrTC9WFiqbkYiqZUxUgq1%2Fimg.png">

이렇게 **평균하여 1차원 벡터로 만들면 가중치의 개수를 상당히 많이 줄여주는데**, FC 방식을 이용할 경우에는 가중치의 개수가 7 x 7 x 1024 x 1024 = 51.3M이지만, **GAP를 사용하면 단 1개의 가중치도 필요하지 않다.** 또한 GAP를 적용할 시, **fine tuning**을 하기 쉽게 만든다.



<img src = "https://sike6054.github.io/blog/images/GoogLeNet,%20Fig.3(removed).png">

- Filter size가 5x5이고 strides가 3인 average pooling layer. 출력의 shape은 (4a)와 (4d)에서 각각 4x4x512와 4x4x528이다
- Dimension reduction을 위한 1x1 conv layer(128 filters) 및 ReLU
- FC layer(1024 nodes) 및 ReLU
- Dropout layer (0.7)
- Linear layer에 softmax를 사용한 1000-class classifier.



### 6. Training Methodology

GoogLeNet은 분산 학습 프레임 워크를 이용해 적당한 양의 모델과 데이터 병렬성을 이용해서 학습시켰다. 비록 GoogLeNet의 학습은 CPU 기반으로 진행되었지만, high-en GPU를 사용하면 일주일 안에 수렴할 것이라고 대략적인 추정을 제안했다. 

asynchronous SGD를 사용했으며, momentum은 0.9를 사용했다. 또한 learning rate schedule을 사용했는데 8 epoch마다 4%의 learning rate를 감소하도록 고정했다. 추론에 사용될 최종 모델을 만들기 위해 Polyak averaging을 사용했다. 

> [cs231n 강의 슬라이드](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture7.pdf)에서는 Polyak averaging의 동작을 다음의 한 줄로 표현했음. 부가 설명은 [강의 동영상](https://www.youtube.com/watch?v=_JB0AO7QxSA) 참조
>
> **INSTEAD of using actual parameter vector, keep a moving average of the parameter vector and use that at test time.**

이미지 샘플링 방법은 ILSVRC 2014 competition까지의 몇 달 동안 크게 변화됐었다. 이미 수렴한 모델들은 다른 옵션을 사용하여 학습됐으며, 때로는 dropout이나 learning rate 등의 hyperparameter를 변경하기도 했다.

> 그래서 이 네트워크를 학습하는 가장 효과적인 방법에 대한 가이드를 제공하긴 어렵다고 함.

문제를 더 복잡하게하기 위해, 모델 중 일부는 상대적으로 작은 크기의 crop 상에서 주로 학습했고, 다른 모델들은 더 큰 크기의 crop 상에서 학습했다.

> 이는 [Andrew Howard의 연구](https://arxiv.org/ftp/arxiv/papers/1312/1312.5402.pdf)에서 영감을 얻은 방법이라 한다.

Competition 이후에는, 종횡비를 [3/4, 4/3]로 제한하여 8% ~ 100%의 크기에서 균등 분포로 patch sampling 하는 것이 매우 잘 작동한다는 것을 발견했다. 또한, [Andrew Howard의 연구](https://arxiv.org/ftp/arxiv/papers/1312/1312.5402.pdf)의 ‘photometric distortion’이 overfitting 방지에 유용하다는 것을 발견했다.



참고자료

> 1. https://arxiv.org/abs/1409.4842
> 2. https://sike6054.github.io/blog/paper/second-post/
> 3. https://phil-baek.tistory.com/entry/3-GoogLeNet-Going-deeper-with-convolutions-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0
> 4. https://deep-learning-study.tistory.com/389?category=963091
> 5. https://velog.io/@whgurwns2003/Network-In-NetworkNIN-%EC%A0%95%EB%A6%AC
> 6. https://arclab.tistory.com/162
> 7. https://leedakyeong.tistory.com/entry/%EB%85%BC%EB%AC%B8-GoogleNet-Inception-%EB%A6%AC%EB%B7%B0-Going-deeper-with-convolutions-1

 











































































> 1. https://arxiv.org/abs/1409.4842
