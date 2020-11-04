---
title: Imputation을 이용한 결측치 다루기(Handle Missing Data)
categories: Feature Engineering
author_profile: true
---


### Imputation
**일단, 데이터에 결측치가 있다면 모델을 학습할 때 방해가 될 수 도 있고, 잘못된 방향으로 결과가 나올 수 있다. 그렇다면 
결측치를 처리 해야하는데 어떤 방식으로 처리를 해야할까? 결측치가 있는 데이터를 지우기? 아니면 결측치 채우기? 여기서 결측치를 다루는 방법중 하나가 Imputation 방법이다. 
Imputation은 결측치를 통계값(mean, Median, Mode)으로 처리하는 방법이다.**


**나는 일단 Mean, Median, Mode 각각 결측치를 채워보고 모델에 학습후 어떤 방법이 더 좋은 성능이 나왔는지 테스트를 해볼것이다. 어떤 방법에는 Mean 방법이 좋을 수 도 있고
Median 방법이 좋을 수 도 있고 Mode 방법이 더 좋을 수 도있다. 그래서 실험을 해볼 것이다.** 



### 실험에 사용 할 데이터를 가져오자 
**실험에서 사용 할 데이터는 타이타닉 데이터셋이다.**