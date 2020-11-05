---
title: Imputation을 이용한 결측치 다루기(Handle Missing Data)
categories: Feature_Engineering
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



```python
train = pd.read_csv("/kaggle/input/titanic/train.csv")
test = pd.read_csv("/kaggle/input/titanic/test.csv")
train
```

<img src="/assets/images/ca5.PNG">

**전처리를 해야하니 정답칼럼을 지워주자!!**

```python
all_data2 = all_data.drop(["Survived"], axis=1)
all_data2
>>
     PassengerId  Pclass                                               Name  \
0              1       3                            Braund, Mr. Owen Harris   
1              2       1  Cumings, Mrs. John Bradley (Florence Briggs Th...   
2              3       3                             Heikkinen, Miss. Laina   
3              4       1       Futrelle, Mrs. Jacques Heath (Lily May Peel)   
4              5       3                           Allen, Mr. William Henry   
..           ...     ...                                                ...   
413         1305       3                                 Spector, Mr. Woolf   
414         1306       1                       Oliva y Ocana, Dona. Fermina   
415         1307       3                       Saether, Mr. Simon Sivertsen   
416         1308       3                                Ware, Mr. Frederick   
417         1309       3                           Peter, Master. Michael J   

        Sex   Age  SibSp  Parch              Ticket      Fare Cabin Embarked  
0      male  22.0      1      0           A/5 21171    7.2500   NaN        S  
1    female  38.0      1      0            PC 17599   71.2833   C85        C  
2    female  26.0      0      0    STON/O2. 3101282    7.9250   NaN        S  
3    female  35.0      1      0              113803   53.1000  C123        S  
4      male  35.0      0      0              373450    8.0500   NaN        S  
..      ...   ...    ...    ...                 ...       ...   ...      ...  
413    male   NaN      0      0           A.5. 3236    8.0500   NaN        S  
414  female  39.0      0      0            PC 17758  108.9000  C105        C  
415    male  38.5      0      0  SOTON/O.Q. 3101262    7.2500   NaN        S  
416    male   NaN      0      0              359309    8.0500   NaN        S  
417    male   NaN      1      1                2668   22.3583   NaN        C  

```


### 결측치 확인

```python
all_data2.isnull().sum()
>>
PassengerId       0
Pclass            0
Name              0
Sex               0
Age             263
SibSp             0
Parch             0
Ticket            0
Fare              1
Cabin          1014
Embarked          2
dtype: int64

# 결측치를 가지고 있는 컬럼 가져오기 
all_data2.columns[all_data2.isnull().sum() >= 1]

```


### 결측치 시각화 missingo 라이브러리를 활용해서 결측치 비율을 알 수 있다.


```python
# 결측치 시각화 
import missingno as msno

msno.bar(all_data2, figsize=(12,6), fontsize=12, color='steelblue')

```
<img src="data:image/png;base64,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">


###수치형 데이터만 가져오겠다



```python
# 수치형 데이터만
all_data2 = all_data2[all_data2.columns[all_data2.dtypes !=object]]

# 결측치를 평균으로 채워줬을 때 
all_data2["Age"] = all_data2["Age"].fillna(all_data2["Age"].mean())
all_data2["Fare"] = all_data2["Fare"].fillna(all_data2["Fare"].mean())

# 결측치를 median으로 채워줬을 때
all_data2["Age"] = all_data2["Age"].fillna(all_data2["Age"].median())
all_data2["Fare"] = all_data2["Fare"].fillna(all_data2["Fare"].median())
```


### 데이터 분리

```python
from sklearn.model_selection import train_test_split

train2 = all_data2[:len(train)]
test2 = all_data2[len(train):]
x_train, x_valid, y_train, y_valid = train_test_split(train2, train["Survived"], test_size= 0.2, random_state=42, stratify=train["Survived"])

```

### Baseline을 Randomforest

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(x_train, y_train)

result = rf.predict(x_valid)
```

### roc_auc socre 로 모델 평가 해보기 

```python
from sklearn.metrics import roc_auc_score


# 평균으로 결측치를 채웠을 때 모델 성능 
roc_auc_score(y_valid, pred)
>>
0.6408432147562583

# median의 결과는 위에서 mean을 median만 바꿔서 테스트 해보면 된다. 
```

### 결과 

<img src="/assets/images/cas.PNG">



**장점**
  - 통계의 대표값들로 결측치를 채우면 사용하는 방법이 빠르다 

**단점**
  - 다른 feature의 상관관계를 고려하지 않는다.
  - 정확도가 떨어진다.
  - 최빈값(Mode)의 경우 데이터의 편향이 생긴다.

**더 좋은 방법이 없을까? 하다가 찾아본 방법이 KNN방법이 있다 K-nn방법은 어떤 한점과 그 점과 가까운  k개의 데이터를 선택하고 그 k개의 평균으로 결측치를 채우는 방법이다
하지만, 이방법도 단점이 있다 계산량이 많고, K-nn은 outliner에 민감하다. 고차원의 데이터의 매우 부정확할 수 있다.**


## K-nn을 활용한 Imputation은 다음 글에서 정리