---
title: Batch Normalization
categories: Pytorch
author_profile: true
---



## Batch Normalization

신경망에는 과적합과 Gradient Vanishing 외에도 Internal Covariance shift라는 현상이 발생한다. Internal Covariance shift란  각 Layer마다 Input 분포가 달라짐에 따라 학습 속도가 느려지는 현상
**Batch Normalization**은 이를 방지하기 위한 기법이다. 말 그대로 Layer의 input분포를 정규화해 학습 속도를 빠르게 하겠다는것 **Batch Normalization** x는 input의 분포를 의미 Beta(B)와
Gamma가 없다고 가정하면 정규화하는 수식과 일치하는 것을 알 수 있다. Beta와 Gamma는 각각 분포를 shift시키고 Scaling시키는 값으로 Back Propagation을 통해 학습 시킨다.


**Batch Normalization**을 사용하면 학습 속도를 향상시켜주고 Gradient Vanishing 문제도 완화 해준다.


<image src ="/assets/images/bn.PNG">
**출처**
https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html

ReLU 함수를 통해 데이터의 0의 미만 값이 0으로 된 것을 알 수 있다. 이후 다음 weight와 선형 결합을 통해 분포가 우측으로 이동되는 것을 볼 수 있고 
Batch Normalization을 통해 정규 분포와 비슷한 형태로 정규화되는 것을 볼 수 있다. 이 분포를 다시 ReLU 함수를 통해 0 미만 값이 0으로 되는 것을 볼 수 있다. 만약, Batch Normalization을 하지
않고 바로 ReLU함수에 들어갔다면 input 값이 0 미만이면 0 input 값이 0 이상이면 자기자신 그대로 출력하는 함수 일것이다.**즉, Batch Normalization을 사용하지 않는다면 Hedden Layer를 쌓으면서 
비선형 활성 함수를 사용하는 의미가 없어질 가능성이 있다. Batch Normalization의 분포를 정규화해 비선형 활성함수의 의미를 살리는 개념이라고 볼 수 있다.**




## Batch Normalization이 적용되었을 때 Layer 분포
<image src = "https://guillaumebrg.files.wordpress.com/2016/02/bn.png?w=768">
**image from**
https://guillaumebrg.wordpress.com/2016/02/28/dogs-vs-cats-adding-batch-normalization-5-1-error-rate/