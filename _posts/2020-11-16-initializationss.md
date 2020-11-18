---
title: Pytorch를 이용해서 MNIST에 Dropout + ReLU + Batch_Normalization + He Uniform Initialization 적용하기
categories: Pytorch
author_profile: true
---



**이전에 MNIST데이터 셋을 가지고 설계했던 MLP에서 간단한 함수 하나만 추가해주면 된다. 밑에 코드를 보자.**

```python
import torch.nn.init as init

def weight_init(m):                    # (1)
    if isinstance(m, nn.Linear):
        init.kaiming_uniform_m(m.weight.data)


model = Net().to(DEVICE)
model.apply(weight_init)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum= 0.5)
criterion = nn.CrossEntropyLoss()

```
## (1)설명

**Weight, Bias 등 딥러닝 모델에서 초깃값으로 설졍되는 요소에 대한 모듈인 init를 임포트한다. MLP 모델 내의 weight를  초기화할 부분을 설정하기위해 weight_init 함수를 정의한다.
MLP 모델을 구성하고 있는 파라미터 중 nn.Linear에 해당하는 파라미터 값에 대해서만 지정한다. nn.Linear에 해당하는 파라미터 값에 대해 he_initialization을 이용해 파라미터 값을 초기화 한다.**


**weight_init함수를 Net()클래스의 인스턴스인 model에 적용한다. 지금까지 다룬 예제가 Class 내 모델을 설계하는 영역에서 설정했다면 이번에는 모델을 정의하는 부분에서 설정을 바꿔주게 된다.
우선 model을 정의한 후 apply를 이용해 모델의 파라미터를 초기화한다. 초기화를 진행할 때 정의된 weight_init함수를 보면 모델 내 파라미터 값중 nn.Linear인스턴스에 대해서는 Kaming_uniform을 이용해
초기화하는 것으로 설정되어 있다. 여기서 kaming_uniform은 He Inititalization을 의미한다 이외에 파라미터 값은 기본값으로 설정된 분포에서 샘플링해 랜덤으로 설정 마치 nn.Linear가 위의 분포에서
설정된 결과처럼 말이다.**


## 학습 결과 

<image src = "assets/image/inits.PNG">


