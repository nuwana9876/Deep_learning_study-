

# CNN 정리 

CNN에 관한 정리는 TAEWAN.KIM 블로그를 바탕으로 정리하며 만들었음을 밝힙니다.

> http://taewan.kim/post/cnn/

### CNN 용어 정리

1. Convolution (합성곱)
2. 채널 (channel)
3. 필터 (filter)
4. 커널 (kernel)
5. 스트라이드 (stride)
6. 패딩 (Padding)
7. 피처 맵 (Feature map)
8. 액티베이션 맵 (Activation map)
9. 풀링 (Pooling)



###### Convolution

convolution은 신호와 시스템 과목에서 배웠듯이, 합성곱 연산은 두 함수 f,g 중 하나의 함수를 반전(reverse) 하고 이동(shift) 한 후, 나머지 다른 함수와 곱한 결과를 적분하는 것을 의미한다.

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/445f13a390d6e35ef67aa8d1e1099ab898d3bcb6)[^1]

이를 그림으로 보게 되면 [^2]

<left><img src = "https://upload.wikimedia.org/wikipedia/commons/9/97/Convolution3.PNG" width ="50%" height = "50%"></left>

이를 2차원 입력데이터를 1개의 filter로 합성곱 연산을 수행하는 과정을 통해 Feature Map을 만들 수 있다. 

> 출처 : 위키피디아

<img src = "http://deeplearning.stanford.edu/wiki/images/6/6c/Convolution_schematic.gif" width ="50%" height = "50%">   [^3]

###### Channel

이미지 픽셀은 각각 실수이다. 컬러사진의 경우 RGB 3개의 실수를 통해 표현한 3차원 데이터이다. 흑백사진의 경우에는 2차원 데이터로 1개의 채널로 구성된다. Convolution layer에 들어가는 입력 데이터는 1개 이상의 channel을 필요로 하고, 만약 n개의 필터가 convolution layer에 적용된다면 , 출력 데이터는 n개의 채널을 가지게 된다.

###### Convolution Layer 출력 데이터 크기 산정

- 입력 데이터 높이  : H
- 입력 데이터 폭 : W
- 필터 높이 : FH
- 필터 폭 : FW
- stride : S
- padding : P

 [^4]

<left><img src = "https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbgnKVA%2FbtqxTK6I4u3%2FwJrmZv2JzCxLRg5BjgWON0%2Fimg.jpg" width = 50% height = 50%> 



###### Filter & Stride

위의 convolution part에서 노란색 영역을 filter라고 부른다. Filter는 다른 말로는 Kernel이라고도 하는데 이는 해당하는 특징이 이미지에 있는지 없는지를 찾아내기 위해 사용된다. 입력데이터가 있으면 이를 stride라는 지정된 간격만큼 이동하면서 채널 마다 convolution을 진행하고,  각 채널마다 계산된 convolution의 결과를 다 합하여 출력데이터인 Feature map이 됩니다.

<img src = "https://taewanmerepo.github.io/2018/01/cnn/conv2.jpg" width = 50%, height = 50%>[^5]

###### Activation map

Convolution layer의 입력 데이터를 filter가 이동하면서 convolution 한 결과를 Feature map이라고 하는데 이 Feature map은 convolution으로 인해 만들어진 행렬이며 Activation map의 경우 Feature map에 활성화 함수를 적용한 결과이다. 즉 convolution layer의 최종 출력 결과가 Activation map이 되는 것이다.



###### Padding

Padding 이란 convolution을 할 때, stride에 따라 출력 행렬의 크기가 달라지는데, stride가 큰 경우 출력 데이터가 크게 줄어들기 마련이다. 이 때, 출력 데이터가 줄어드는 것을 방지하기 위해 입력 데이터 주위에 지정된 크기만큼 특정 값으로 채워 넣는 것을 Padding이라고 하고, 보통의 경우는 0으로 채워넣는다.

padding을 쓰는 이유에 대해서는 위에서 설명했다시피, 이미지 데이터의 축소를 막기 위해서도 있지만, Edge pixel data를 충분히 활용하기 위해서도 있다. 예를 들어 7X7의 입력 데이터가 주어졌다고 할 때, 중요한 데이터 성분(행렬의 한 pixel값)이 만약 입력 데이터의 외곽에 존재한다면, convolution을 진행했을 때 그 pixel은 안쪽에 존재하는 pixel보다 덜 사용되게 된다. (convolution을 하게 되면 안쪽의 pixel의 경우 여러번 겹쳐서 이용되는데 반해, 외곽에 존재하는 pixel은 그보다는 적게 이용된다.) 이로인해 중요한 정보가 덜 사용되게 되는데, 이를 방지하기 위해서 모서리를 0으로 둘러싸줄 경우, 중요한 정보가 padding을 하기 이전보다 많이 사용될 수 있다.



###### Pooling layer

pooling layer의 경우 convolution layer의 출력 결과를 입력으로 받아, Activation map의 크기를 줄이거나 특정 데이터를 강조하는 용도로 사용된다.  pooling layer 처리 방식으로는 Max pooling , Average pooling , Min pooling이 있으며 , CNN에서는 Max pooling을 주로 사용한다.  (- Pooling은 Activation function 마다 매번 적용하는 것이 아니라, 데이터의 크기를 줄이고 싶을 때 선택적으로 이용하는 것이다.) Max pooling의 경우 해당 receptive field에서 가장 큰 값을 고른 것을 말한다. Average pooling의 경우 해당 receptive field안에 존재하는 parameter의 평균값만 계산한 것이다.

Pooling Layer 특징

- 학습대상 parameter가 없다.
- pooling layer를 통과하면 행렬의 크기가 감소한다.
- pooling layer를 통해서 채널의 수가 바뀌지 않는다.
- 입력 데이터의 변화에 Robust하다.



Pooling을 쓰는 이유는 앞선 layer인 Conv layer나 Activation등을 거치고 나온 feature map의 모든 값이 전부 필요하지 않기 때문이다. 즉, 추론을 하는데에 있어 적당량의 데이터 (해당 이미지의 특징을 나타내는 중요한 값)만 필요하기에 Activation map의 크기를 줄이면서 특징을 나타내는 값만 가져오는 것이다.

Pooling layer의 효과는 

1. parameter를 줄이기에, network의 overfitting을 억제한다.
2. parameter가 줄어들어 그만큼 computation(계산)이 줄어들기에, hardware resource를 절약하고 속도가 빨라진다.



Pooling 레이어에서 일반적인 Pooling 사이즈는 정사각형이다. Pooling 사이즈를 Stride 같은 크기로 만들어서, 모든 요소가 한번씩 Pooling되도록 만든다. 입력 데이터의 행 크기와 열 크기는 Pooling 사이즈의 배수(나누어 떨어지는 수)여야 한다. 결과적으로 Pooling 레이어의 출력 데이터의 크기는 행과 열의 크기를 Pooling 사이즈로 나눈 몫이다.

<left><img src = "https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FQy3bk%2FbtqxZqeppVo%2F7JCpwGtJhcs4JmWzqzTTh1%2Fimg.png" width = 50% height = 50%>



###### CNN 전체 구성

CNN은 Convolution layer와 Max pooling layer를 반복적으로 stack을 쌓는 특징 추출 부분과 Fully connected layer를 구성하고 마지막 출력층에는 softmax함수를 이용한 분류 부분으로 나뉜다.

CNN을 구성하면서 중요한 점은 filter나 stride, padding의 크기를 조절해서 각 layer의 입력과 출력 부분의 크기를 잘 조절해야 한다는 점이다.

<img src = "https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fdi5m1K%2FbtqxXI08ZvE%2F8rtI9KVUPNj97J7kybx1ak%2Fimg.png" >

 [^6]

- 코드는 github에 올려져 있으며, pytorch로 구현되어 있다.



###### CNN 입출력 parameter 계산

밑의 나오는 조건과 같은 이미지를 학습하는 CNN의 각 layer 별 입력과 출력 데이터의 shape를 계산하고, 네트워크가 학습시키는 parameter의 개수를 계산하면 다음과 같다.

- 입력 데이터 shape : 39 X 31 X 1
- 분류 데이터  : 100

###### convolution 의 학습 parameter 수는 "input channel X filter width X filter height X output channel " 로 계산된다.

<img src = "https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FYx9YE%2FbtqxYWEDVAQ%2F8MTuToOvvW0KYvItm1HIkk%2Fimg.jpg" width = 100% height =  70%>

각 layer마다의 shape 계산은 생략하고, 중요한 부분만 따로 다루겠다.



Convolution layer를 다루는데 Activation map의 크기를 계산하는 식은 다음과 같다.

<left><img src = "https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbUVAFV%2FbtqxZRXCnfV%2FPtcR0yDfaue7xTJMQZXX10%2Fimg.png" width = 50% height = 50%>



###### Flatten layer shape

Flatten Layer의 경우 Fully connected neural network로 바꿔주는 역할을 한다. layer에는 parameter가 존재하지않고, 입력 데이터의 shape 변경만을 수행한다.



요약하자면 다음과 같이 표현 가능하다.

<img src = "https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbZ9xTN%2FbtqxZpAl1Q9%2FYQBhZLtMAnoFHGNrFqYcF1%2Fimg.png" >









---

[^1]:출처 : https://ko.wikipedia.org/wiki/%ED%95%A9%EC%84%B1%EA%B3%B1 
 
[^2]: 출처 : https://ko.wikipedia.org/wiki/%ED%95%A9%EC%84%B1%EA%B3%B1
 
[^3]:출처: http://deeplearning.stanford.edu/wiki/index.php/Feature_extraction_using_convolution*

[^4]:출처 : http://taewan.kim/post/cnn/
 
[^ 5]:출처: https://taewanmerepo.github.io/2018/01/cnn/conv2.jpg
 
[^ 6]:출처 : https://www.researchgate.net/figure/Architecture-of-our-unsupervised-CNN-Network-contains-three-stages-each-of-which_283433254

