# Deep learning review

###### LeCun, Yann, Yoshua Bengio 및 Geoffrey Hinton.

- 1문단

Deep Learning은  현대 사회의 많은 측면에서 사용되는데 예를 들어 최신의 speech recognition, visual object recognition, object detection 및 drug discovery 나 genomics 같은 분야에서 사용된다.

Deep learning은 backpropagation(역전파) 알고리즘을 활용해서 이전 layer의 표현으로 부터 다음 layer의 표현을 계산하는 것으로, 어떻게 machine이 내부 parameter(weight, bias)들을 바꿔야 하는지 나타냄으로써  large data set에서 복잡한 구조를 발견하는 것이다. 이는 뒤에 자세히 나온다.

이 review에서 Deep learning method를 표현하는데 있어서 <u>Representation learning</u> 이라는 방식으로 Deep learning을 표현할 수 있는데, <u>이는 simple하고 non-linear한 module을 사용해서 multiple level로 표현</u>하는 것이다.  즉 one-level(raw input)을 higher하고 더 추상적인 level들의 표현으로 변환하는 것이다.  그로인해 classification의 경우 input 측면에서 자세히 나타낸 higher layer는 input과 관련 없는 변수들을 막고, 식별하는 데에 있어서 중요한 역할을 한다.

Deep learning의 중요한 측면은 engineers에 의해 layers of features 가 결정되는 것이 아니라, general-purpose learning 절차를 사용해서 데이터에 의해 학습이 되어진다는 것이다. 또한 Deep learning은 문제를 푸는데에 있어서 주요한 성장을 만들었는데, AI community의 최선의 시도들에 대해 저항하면서(계속 최선의 선택을 하기위해 반복하면서)  높은 차원의 데이터에서의 복잡한 구조를 찾아낼 수 있었다. 그러므로 많은 분야에서 문제 해결을 위해 사용되었다. deep learning은 직접 engineering하는 것이 거의 없어서, 쉽게 많은 데이터의 계산에 대해 이득을 본다. 새로운 알고리즘과 구조들은 deep neural network의 처리를 가속화 시켜준다.

