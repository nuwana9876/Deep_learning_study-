# AlexNet 

> 출처 : https://deep-learning-study.tistory.com/376

Maxpooling의 목적

1.input size를 줄인다

2.불필요한 parameter들을 줄여서 overfitting을 조절한다.

3.pooling 할 때 결국 특징을 제일 잘 나타내는 건 feature map에서 가장 큰 수이며 가장 큰 수가 특징에 제일 관련이 있다는 의미.

----------------------------------------------------------------------------------------------------------------------------------------#

super()라는 함수는 자식 클래스에서 부모클래스의 내용을 사용하고 싶을 경우에 사용한다. 

특히 overriding 문제때문에 더더욱 super함수를 써야 하는데,

```python
class Person:

  def greeting(self):

​    print('안녕하세요.')

 

class Student(Person):

  def greeting(self):

​    super().greeting()  # 기반 클래스의 메서드 호출하여 중복을 줄임

​    print('저는 파이썬 코딩 도장 학생입니다.')

james = Student()

james.greeting()
```

> 출처 : https://dojang.io/mod/page/view.php?id=2387

위 예제를 보면 super를 통해 부모 클래스에서 상속받아 greeting을 사용하고 

자식 클래스에서 이름은 같지만 print('저는 파이썬 코딩 도장 학생입니다')를 덧붙여서 새로운 함수를 만들어냈다.

메서드 오버라이딩을 통해 원래 기능을 유지하면서 새로운 기능을 덧붙일 때 사용한다.

----------------------------------------------------------------------------------------------------------------------------------------#



ReLU의 <code>inplace = True</code>> 는 결과값을 새로운 변수에 저장하는 것이 아닌 기존 데이터를 결과값으로 덮어쓰는 것이다.

----------------------------------------------------------------------------------------------------------------------------------------#



__name__ 이란 현재 모듈의 이름을 담고있는 내장 변수이다. 이 변수는 직접 실행되는 경우(인터프리터를 통해 실행되는

경우) 에는 if문 안에 있는 내용을 실행하고, 그렇지 않으면 (다른 곳으로 import되어서 쓰이는 경우) else문을 실행한다.\



만약 직접 실행되는 경우는 __main__이 되며, 임포트 되는 경우는 파일의 이름 juntae.py의 juntae를 반환한다.



----------------------------------------------------------------------------------------------------------------------------------------#



random을 적용할 때 쓰는 함수들

```python
import torch

import numpy as np

import random

torch.manual_seed(random_seed)

torch.cuda.manual_seed(random_seed)

torch.cuda.manual_seed_all(random_seed) # if use multi-GPU

torch.backends.cudnn.deterministic = True

torch.backends.cudnn.benchmark = False

np.random.seed(random_seed)

random.seed(random_seed)
```



----------------------------------------------------------------------------------------------------------------------------------------#