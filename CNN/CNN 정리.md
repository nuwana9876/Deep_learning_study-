

# CNN 정리 

CNN에 관한 정리는 TAEWAN.KIM 블로그를 바탕으로 만들었음을 밝힙니다.

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

<img src = "http://deeplearning.stanford.edu/wiki/images/6/6c/Convolution_schematic.gif" width ="50%" height = "50%">   [^3]

###### Channel

이미지 픽셀은 각각 실수이다. 컬러사진의 경우 RGB 3개의 실수를 통해 표현한 3차원 데이터이다. 흑백사진의 경우에는 2차원 데이터로 1개의 채널로 구성된다. Convolution layer에 들어가는 입력 데이터는 1개 이상의 channel을 필요로 하고, 만약 n개의 필터가 convolution layer에 적용된다면 , 출력 데이터는 n개의 채널을 가지게 된다.



###### Filter & Stride













---

[^1]:출처 : https://ko.wikipedia.org/wiki/%ED%95%A9%EC%84%B1%EA%B3%B1 
[^2]: 출처 : https://ko.wikipedia.org/wiki/%ED%95%A9%EC%84%B1%EA%B3%B1
[^3]:출처: http://deeplearning.stanford.edu/wiki/index.php/Feature_extraction_using_convolution*

