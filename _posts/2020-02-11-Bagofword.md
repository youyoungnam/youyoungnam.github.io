---
title: Bag of Word(BoW)
categories: NLP
author_profile: true
---

# Bag of Word(Bow)  
Bag of Word란: 단어의 순서는 전혀 고려하지 않고, 단어들의 출현 빈도만 집중하는 텍스트 데이터의 수치화 방법이다.

그렇다면, BoW는 어떻게 만들어야할까?

**순서**
- 1 우선 각 단어에 고유한 인덱스를 부여한다.
- 2 각 인덱스의 위치에 단어 토큰의 등장횟수를 기록한 벡터를 만든다.

**예를들어**

```python
from konlpy.tag import Okt
import re
okt = Okt()
token=re.sub("(\.)","","정부가 발표하는 물가상승률과 소비자가 느끼는 물가상승률은 다르다.")  
# 정규 표현식을 통해 온점을 제거하는 정제 작업 
token=okt.morphs(token)  

word2index ={}
bow = []

for voca in token:
    if voca not in wordindex.key():
        word2index[voca] = len(word2index)

        bow.insert(len(word2index)-1, 1)
    else:
        index = word2index.get(voca)
        bow[index] +=1
print(word2index)
print(bow)
```
>('정부': 0, '가': 1, '발표': 2, '하는': 3, '물가상승률': 4, '과': 5, '소비자': 6, '느끼는': 7, '은': 8, '다르다': 9)  
>[1, 2, 1, 1, 2, 1, 1, 1, 1, 1] 


**다시말하자면 BoW는 단어의 빈도수를 기록하는것이다.**

## 다른 방식으로 BoW만들기

**CounterVectorizer 클래스로 BoW 만들기**

```python
from sklearn.feature_extraction.text import CountVectorizer
corpus = ['you know I want your love. because I love you.']
vector = CounterVectorizer()
print(vector.fit_transform(corpus).toarray())
# 코퍼스에서 단어의 빈도 수를 기록한것을 보는방법

print(vector.vocabulary_)
# 각 단어의 인덱스가 어떻게 부여되었는지 보여주는 방법.
```

> [ [1 1 2 1 2 1] ] # 단어의 빈도 수를 기록
{'you': 4, 'know': 1, 'want': 3, 'your': 5, 'love': 2, 'because': 0} # 단어가 어떻게 부여 되었는지 결과

