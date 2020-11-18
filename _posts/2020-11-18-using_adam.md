---
title: Pytorch로 MNIST를 이용해 MLP(muti Layer Perceptron)설계할 때 Dropout + ReLU +Batch+ He Uniform Intialization Adam 적용해보기 
categories: Pytorch
author_profile: true
---



**지금까지 만들어 놨던 모델에서 약간만 수정하면 된다**


```python
# 이 코드를 밑에 코드로 바꿔주면 된다.
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum = 0.5)
# 이코드로 교체 
optimizer = torch.optim.Adam(model.parameters(), lr= 0.01)
```


## 학습 결과

<img src = "/assets/images/admss.PNG">

