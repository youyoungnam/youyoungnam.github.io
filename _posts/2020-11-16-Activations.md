---
title: ReLU함수
categories: Activation_함수
author_profile: true
---



## Activation함수 

**Activation함수는 어떤 신호를 입력받아 이를 적절히 처리해 출력해주는 함수를 의미하고 MLP에서 기본적으로 시그모이드 함수를 사용한다.
그런데, Back Propagation 과정 중에 시그모이드를 미분한 값을 계속 곱해주면서 Gradient값이 앞 단의 Layer로 올수록 0으로 수렴하는 현상이 발생. 이를 Gradient Vanishing이라고 한다. 
Gradient Vanishing은 Hidden_layer가 깊어질수록 심해지기 때문에 Hidden Layer를 길게 쌓아 복잡한 모델을 만들 수 있다는 장점이 의미가 없게 된다.**



## ReLU함수 

**ReLU(Rectified Linear unit)함수는 기존의 시그모이드 함수와 같은 비선형 활성 함수가 지니고 있는 문제점을 어느 정도 해결한 활성 함수이다. 활성 함수 ReLU는 F(x) = max(0, x)와 같이
정의된다.**


<img src = "https://pytorch.org/docs/stable/_images/ReLU.png">

**image 출처**
https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html


**입력 값이 0 이상이면 이 값을 그대로 출력하고 0 이하이면 0으로 출력하는 함수이다. 이 활성 함수를 미분할 때 입력 값이 0이상인 부분은 기울기가 1 입력값이 0이하인 부분은
0이 된다. 즉, Back Propagation 과정 중에 곱해지는 Activation 미분 값이 0 또는 1이 되기때문에 아예 없애거나 완전히 살리는 것으로 해석된다. 이를 통해 Hidden Layer가 깊어져도 
Gradient Vanishing이 일어나는 것을 완화시키며 Layer를 깊게 쌓아 복잡한 모형을 만들 수 있게 된다.**



**ReLU함수가 나오면서 RELU의 변형 함수가 많이 나오기 시작했다. Leaky Relu, ELU, parametric relu, SELU , SERLU등 다양한 Activation 함수가 나오고 있다 각 함수의 형태의 CIFAR10 데이터에
각 활성화 함수를 적용한 성능은 SERLU가 성능이 가장 좋게 나왔다고 할 수 있다. 하지만 모든 Task에 대해 이 활성 함수가 항상 가장 좋다고 말하기는 어렵다. 실제로 많은 논문의 코드를 살펴보면
다양한 활성 함수를 사용하는 것을 알 수 있다. 즉, 활성 함수 내에서 어느 정도 일반화된 성능 차이는 있을 수 있지만, 딥러닝을 적용하는 분야에 따라 조금씩 성능의 차이는 있다**
