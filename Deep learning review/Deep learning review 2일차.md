# Deep learning review

###### LeCun, Yann, Yoshua Bengio 및 Geoffrey Hinton.

- 2문단

지도학습(Supervised learning)

Deep 하던지 Deep 하지 않던지 간에 , 대부분의 Machine learning 형태는 supervised learning이다.  Supervised learning에 대해 설명하자면, 많은 양의 데이터를 모으고, 각각 category에 맞게 labeling을 해준다. training을 하면서 machine은 vector of scores를 output으로 주는데, 이를 통해 가장 높은 score와 관련있는 category를  그 이미지의 category로 추정한다.  우리는 이 과정에서 output score 와 desired pattern of scores의 error를 측정하기 위해 objective function (목적함수 = loss function)을 이용하고, 그런 다음 error를 줄이는 방향으로 내부 adjustable한 parameter들을 조절한다.이를 weight라고 하는데 input-output관계를 정의해주는 real number라 보면 된다. weight vector를 조절하기 위해 gradient vector를 계산해야 하는데, tiny하게 weight가 증가 했을 때 error가 얼마나 증가하는 지를 나타내는 vector이다. weight vector는 gradient vector와는 반대 방향으로 조정된다. gradient vector는 minimum에 가깝도록 계산되며 output error 가 낮아지는 방향으로 변한다. 대부분의 practitioner들은 이를 stochastic gradient descent라고 하는데 , output과 error를 계산하고 이를통해 평균 기울기를 구하며, weight들을 조절한다.

여기서 stochastic이란 확률적이라는 뜻으로, 전체 데이터를 small set으로 나누어서 계산하기에, 평균 gradient에 noise를 주지만 장점으로는 정교한 최적화 기술에서 계산하는 것에  비해서는 엄청 빠르게 good set의 weight들을 계산 할 수 있다. 이후에 train set 과는 전혀 다른 test set을 이용해서 performance를 평가한다. 이를 통해 machine은 새로운 데이터에 대한 분별력을 기를 수 있다.

machine learning을 사용하는 이유 중 하나인 linear classifier가 있는데, feature vector component의 weighted sum을 계산해서 이 값이 임계치를 넘는다면 특정 카테고리로 분류하는 방식을 나타낸다. 그러나 linear classfier의 경우 input을 half space으로 나누는 등 너무 단순하게 분류를 하다보니 , 정작 classifier에서 필요로 하는 input에 대한 불필요한 변수들 (position, background,orientation,pitch)등에 민감하지 않고 input에 대한 특정 미세한 변수들에 민감해야 하는 특징을 가지지 못한다. 이 글에서 예로 samoyed와 white wolf의 그림을 classification 하는 것으로 예시를 들었는데, 1. 두 samoyed가 다른 위치, 다른 배경을 가진 사진,2. samoyed와 white wolf 가 같은 포즈와 비슷한 배경을 가진 사진을 classification 하는데 linear classifier (shallow classifier)는 전자를 같은 카테고리로 분류했으며, 후자를 구분하지 못하는 결과를 가져왔다. 이는 위에서 이야기한 irrelevant 한 측면에 민감하지 않고, selective 한 측면에 민감해야 하는 특징을 linear classifier에서는 가지지 못했다는 것이다. 즉 이를 해결하기 위해서 good feature extractor가 꼭 필요하다. 또한 classifier를 더 powerful하게 하기 위해서는 non-linear한 feature들 (kernal method)이 사용된다. 기존의 option의 경우 good feature extractor들을 직접 design 했지만 만약 general-purpose learning procedure로 계산될 수 있는 경우에는 이를 피한다. → 이는 deep-learning 의 장점

Deep learning의 구조는 simple module이 multi stack으로 쌓여 있는 구조인데, 모든 module은 learn 될 수 있으며, 대부분의 module은 non-linear로 계산한다.특히 각 module은 selectivity와 invariance를 모두 증가시키는 방향으로 바꾸는데, 여기서 non-linear한 layer 때문에 특정 미세한 특징에 민감하게, 불필요한 변수들에 대해서는 민감하지 않게 나타내어진다.