## Deep learning review

###### LeCun, Yann, Yoshua Bengio 및 Geoffrey Hinton.

7문단

Recurrent neural networks

speech나 language  같은 sequential 입력을 받을 때 RNN을 쓰는 것이 더 효과적이라고 한다. RNN은 시간에 따라 input으로 들어온 element를  하나 하나를 처리한다. 이 때 RNN은 hidden unit을 가지고 있는데 이를 'state vector'라고 하고 입력으로 들어온 element들의 모든 과거 history에 대한 정보를 포함하고 있는 vector이다.

RNN은 상당히 강력한 network임에도 불구하고 몇몇 문제를 가지고 있는데, backpropagate된 gradient들이 각 time step마다 늘고, 줄어들고 하는 것이 반복된다는 것이다. 그런 반면에 장점을 말해보자면 다음에 나올 문자나 다음 문자들의 순서를 예측하는 것을 잘 수행하고, 복잡한 문제를 해결하는데 사용될 수 있다는 점이다. 예를 들어 영어 문장을 입력 받아 그 의미를 나타내는 표현 vector로 network를 만들고 이를 입력으로 다시 받아 프랑스어로 그 표현을 반환하는 복잡한 문제를 해결할 수 있다는 것을 보여준다. 마치 encoder와 decoder로써 번역을 한 것과 같은 느낌을 준다.

그리고 이를 영어를 프랑스어로 번역하는 것을 다른 관점으로 보면 이미지에 담긴 의미를 영어로 번역하는 것으로도 만들 수 있다. ConvNet을 encoder로 써서 나온 active vector 를 RNN을 decoder로 써서 구한 결과로 표현 해 줄 수 있다는 것이다. RNN을 펼쳐서 나타내보면 매우 깊은 순방향 network가 보이는데 이 layer들은 같은 weight를 공유하고 있다는 점이 특징이다. 비록 이 목적이 매우 long- term한 것을 학습하는 것이지만, 경험적으로 봤을 때 정보를 매우 뒤쪽의 layer로 보내는 것이 거의 어렵다는 것이다. 이와 같은 점을 고치기 위해서 explicit memory로 network를 늘리는 것이 하나의 방법인데, 이에 관한 첫번째 제안은 LSTM(Long short-term memory) network를 이용하는 것이다. memory cell이라는 특별한 unit을 이용해서 기존의 데이터를 축적하거나 삭제하거나 하는 방식을 이용한다. (자세한 LSTM의 내용은 나중에 다루기로 한다.) 요약하면, memory cell을 이용해 더 장기적으로 input을 다음으로 보내주는 것이라고 생각하면 된다.

그 후로 LSTM network는 기존의 RNN에 비해 효율적이라는것이 증명되었는데, 특히 여러 layer를 통과하게 되는 경우에 효과가 더 좋다. 과거 몇 년동안 수 많은 작가들은 memory module로 RNN을 증가시키는 방법을 제안했는데, 첫번째로는 Neural Turing Machine을 쓰는 것이다. 이 network는 tape 같은 memory로, RNN이 읽어 들이거나 쓰는 것을 선택할 수 있도록 하는 memory에 의해 network가 증가되었다.  두번째로는 Memory-network로, standard question-answering 벤치마크에서 상당히 좋은 성적을 거뒀는데, 이 memory는 나중에 network가 질문에 대답하도록 요청되어진 이야기를 기억하는데에 사용된다.

간단한 memorization을 넘어서, Neural Turing Machine과 Memory-network는 task를 수행하기 위해 추론과 symbol을 다루는 것이 필요로 되어지는데, Neural Turing Machine은 algorithm을 배울 수 있고 무엇보다도, 정렬되지 않은 input을 정렬된 symbol들로 output을 표현할 수 있게 된다. 이 때 각각의 symbol들은 list에서 우선순위를 나타내는 실수 값을 가지고 있다. Memory network의 경우, 텍스트 모험 게임과 비슷한 환경에서 세계의 상태에 대해 알거나, 이야기를 읽은 후에 복잡한 추론을 필요로 하는 질문에 대답할 수 있다. 예를 들어 , 'The Lord of the Rings'의 15개 문장 버전을 network에 보여준 후, 'Where is Frodo now?'같은 질문들에 정확하게 대답할 수 있다.