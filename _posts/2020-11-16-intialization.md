---
title: Initialization기법
categories: Pytorch
author_profile: true
---




## Initialization 

**Initialize는 초기화하다라는 뜻이다. 신경망은 처음에 Weight를 랜덤하게 초기화 하고 Loss가 최소화 되는 부분을 찾아간다. 이전에는 초기 분포로 Uniform Distribution이나 Normal Distribution을 
사용했다. Weight를 랜덤하게 초기화하면 신경망의 초기 Loss가 달라진다. 즉, 신경망을 초기화할 때 마다 신경망의 Loss상에서의 위치가 달라질 수 있다. 따라서 최적의 신경망 Loss를 찾아줘야한다.**


**즉 신경망을 어떻게 초기화하느냐에 따라 학습 속도가 달라질 수 있다는것이다. 그렇기 때문에 신경망의 초기화 기법에 대해 다양한 연구가 이뤄지고 있다. 대표적인 초기화 기법을 소개한다.**


**기법**
  - LeCun Initialization 
    - LeCun이라는 Convolutional Neural 네트워크의 창시자의 이름에서 따온 기법으로 LeCon Normal Initialization과 Lecon Uniform Initialization이 있다 각각 초기 분포가 다음과 같은 분포를 따르도록 weight를 초기화 하는것이다
    - W ~ N(0, Var(W)), Var(w) = root(1/n_in)
    - 여기서 n_in은 이전 Layer의 노드 수

  - He Initialization
    - Xavier Initialization은 ReLU 함수를 사용할 때 비효율적이라는 것을 보이는데 이를 보완한 초기화 기법이 He Initialization이다.
    - W ~ U(-root(1 / n_in), +root(1 / n_in))

