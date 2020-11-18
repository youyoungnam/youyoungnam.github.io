---
title: Optimizer
categories: Pytorch
author_profile: true
---




## Optimizer

**이전에 Batch 단위로 Back Propagation하는 과정을 Stochastic Gradient Descent(SGD)라 하고 이러한 과정을 Optimization이라고 한다. SGD 외에도 SGD의 단점을 보완하기 위한 다양한 
Optimizer가 있다. 대표적인 Optimizer에 대해 간단하게 살펴보겠다.**

**Momentum**
 - Momentum은 미분을 통한 Gradient 방향으로 가되, 일종의 관성을 추가하는 개념이다.
 - 일반적인 SGD는 다음과 같이 조금씩 최적의 해(Global Optinum)를 찾아간다. 전체 데이터에 대해 Back Propagation을 하는 것이 아니라 Batch 단위로 Back Propagation하기 때문에 일직선으로 찾아가지 않는다.


 <img src="https://engmrk.com/wp-content/uploads/2018/04/Fig2.png">
 이미지 출처 https://engmrk.com/mini-batch-gd/


 **Momentum**을 사용한다면 밑에 그림 같이 최적의 장소로 더 빠르게 수렴하는 것을 볼 수 있다. 걸어가는 보폭을 크게 하는 개념이라 이해하면 된다. 또한 최적 해가 아닌 지역해(Local Mininum)를 지나칠 수도 있다는 장점이 있다


 <img src ="https://media.springernature.com/original/springer-static/image/chp%3A10.1007%2F978-1-4842-4470-8_33/MediaObjects/463852_1_En_33_Fig1_HTML.jpg">
 이미지 출처 https://eloquentarduino.github.io/2020/04/stochastic-gradient-descent-on-your-microcontroller/




**Adaptive Moment Estimation(Adam)**
  - Adam은 딥러닝 모델을 디자인할 때 기본적으로 가장 많이 사용하는 Optimizer로 RMSProp와 Momentum 방식의 특징을 결합한 방법이다.
  - 2020년 기준으로 많은 딥러닝 모델에서 기본적으로 Adam을 많이 사용하고 있다.


