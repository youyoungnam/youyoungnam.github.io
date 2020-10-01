---
title: Sentiment Analysis on Movie Reviews(NLP)
categories: kaggle
author_profile: true
---

### 영화리뷰에 대한 감정분석(Classify the sentiment of sentences from the Rotten Tomatoes datasets)

**영화리뷰 즉, 댓글들을 보고 어떤 영화에 대한 리뷰글을 보았을때 이 리뷰가 긍정인지 부정인지 아니면 보통인지 를 구분하는 분류 대회이다.**


**데이터 읽어오기**

```python
train = pd.read_table("/kaggle/input/sentiment-analysis-on-movie-reviews/train.tsv.zip")
test = pd.read_table("/kaggle/input/sentiment-analysis-on-movie-reviews/test.tsv.zip")
```
**데이터 보기**

```python
display(train, test)

# Pharse에 영화 리뷰가 담겨있음 
train["Pharse"]
>>
0         A series of escapades demonstrating the adage ...
1         A series of escapades demonstrating the adage ...
2                                                  A series
3                                                         A
4                                                    series
                                ...                        
156055                                            Hearst 's
156056                            forced avuncular chortles
156057                                   avuncular chortles
156058                                            avuncular
156059                                             chortles
Name: Phrase, Length: 156060, dtype: object

# 첫번째 리뷰 보기
train["Pharse"][0]
>>
A series of escapades demonstrating the adage that what is good for the goose is also good for the gander ,
some of which occasionally amuses but none of which amounts to much of a story .
```
**대충 글을 읽어보았을때 good이라는 단어가 있어서 좋은 리뷰인줄 알았는데 정답값을 보면 부정으로 평가하고있다. 이런 리뷰들은 어떻게 처리해야할까?
또다른 예를들자면 이 영화는 정말 자막도 완벽하고 화질도 좋았어 근데 나는 재미없었어 라는 리뷰가 있다면 이 글은 긍정일까 부정일까 내 생각은 앞은 긍정인데 뒤에 문장때문에 부정이라고 평가 할것이다.**



**train, test데이터 결합하기**

```python
all_data = pd.concat([train, test])
all_data


# keras을 활용한 Tokenizer
from keras.preprocessing import Tokenizer

tk = Tokenizer()
tk.fit_on_texts(all_data["Phrase"])


# 가장 많이 등장한 tokenizer(토큰)이 정렬 

print(tk.word_index)

>>
{'the': 1,
 'a': 2,
 'of': 3,
 'and': 4,
 'to': 5,
 "'s": 6,
 'in': 7,
 'is': 8,
 'that': 9,
 'it': 10,
 'as': 11,
 'with': 12,
 'for': 13,
 'its': 14,
 'film': 15,
 'an': 16,
 'movie': 17,
 'this': 18,
 'but': 19,
 'be': 20,
 'on': 21,
 'you': 22,
 'by': 23,
 "n't": 24,
 'more': 25,
 'his': 26,
 'not': 27,
 'one': 28,
 'than': 29,
 'about': 30,
 'at': 31,
 'from': 32,
 'or': 33,
 'all': 34,
 'like': 35,
 'are': 36,
 'have': 37,
 'has': 38,
 'so': 39,
 "'": 40,
 'out': 41,
 'story': 42,
 'who': 43,
 'rrb': 44,
 'up': 45,
 'too': 46,
 'good': 47,
 'most': 48,
 'into': 49,
 'lrb': 50,
 'time': 51,
 'much': 52,
 'what': 53,
 'if': 54,
 'characters': 55,
 'no': 56,
 'comedy': 57,
 'their': 58,
 'just': 59,
 'i': 60,
 'some': 61,
 'can': 62,
 'even': 63,
 'life': 64,
 'your': 65,
 'little': 66,
 'does': 67,
 "''": 68,
 'way': 69,
 'well': 70,
 'will': 71,
 'make': 72,
 'been': 73,
 'funny': 74,
 'only': 75,
 'very': 76,
 'he': 77,
 'do': 78,
 'director': 79,
 'any': 80,
 'enough': 81,
 'us': 82,
 'her': 83,
 'bad': 84,
 'there': 85,
 'new': 86,
 'love': 87,
 'movies': 88,
 'which': 89,
 'they': 90,
 'was': 91,
 'work': 92,
 'two': 93,
 'own': 94,
 'them': 95,
 'best': 96,
 'when': 97,
 'action': 98,
 'old': 99,
 'something': 100,
 'made': 101,
 'we': 102,
 'through': 103,
 'other': 104,
 'people': 105,
 'never': 106,
 'off': 107,
 'makes': 108,
 'would': 109,
 'world': 110,
 'see': 111,
 'many': 112,
 'character': 113,
 'films': 114,
 'self': 115,
 'could': 116,
 'how': 117,
 'long': 118,
 'over': 119,
 'big': 120,
 'may': 121,
 'plot': 122,
 'get': 123,
 'first': 124,
 'look': 125,
 'audience': 126,
 'being': 127,
 'those': 128,
 'every': 129,
 'performances': 130,
 'drama': 131,
 'better': 132,
 'great': 133,
 'real': 134,
 'really': 135,
 'humor': 136,
 'fun': 137,
 'sense': 138,
 "'re": 139,
 'screen': 140,
 'should': 141,
 'man': 142,
 'another': 143,
 'year': 144,
 'few': 145,
 'feel': 146,
 'still': 147,
 'without': 148,
 'both': 149,
 'minutes': 150,
 'ever': 151,
 'hollywood': 152,
 'cast': 153,
 'american': 154,
 'such': 155,
 'down': 156,
 'while': 157,
 'kind': 158,
 'human': 159,
 'less': 160,
 'nothing': 161,
 'heart': 162,
 'between': 163,
 'far': 164,
 'might': 165,
 'full': 166,
 'interesting': 167,
 'hard': 168,
 'performance': 169,
 'family': 170,
 'script': 171,
 'seen': 172,
 'our': 173,
 'because': 174,
 'rather': 175,
 'thriller': 176,
 'often': 177,
 'picture': 178,
 'right': 179,
 'my': 180,
 'same': 181,
 'acting': 182,
 'had': 183,
 'these': 184,
 'were': 185,
 'original': 186,
 'also': 187,
 'go': 188,
 'tale': 189,
 'moments': 190,
 'things': 191,
 'quite': 192,
 'back': 193,
 'emotional': 194,
 'almost': 195,
 'take': 196,
 'itself': 197,
 'end': 198,
 'young': 199,
 'before': 200,
 'thing': 201,
 'here': 202,
 'after': 203,
 'going': 204,
 'music': 205,
 'times': 206,
 'watching': 207,
 'entertaining': 208,
 'scenes': 209,
 'dialogue': 210,
 "'ve": 211,
 'star': 212,
 'romantic': 213,
 'cinema': 214,
 'ca': 215,
 'actors': 216,
 'find': 217,
 'watch': 218,
 'material': 219,
 'where': 220,
 'come': 221,
 'lot': 222,
 'war': 223,
 'kids': 224,
 'documentary': 225,
 'half': 226,
 'style': 227,
 'video': 228,
 'high': 229,
 'years': 230,
 'seems': 231,
 'me': 232,
 'yet': 233,
 "'ll": 234,
 'last': 235,
 'though': 236,
 'feels': 237,
 'point': 238,
 'special': 239,
 'show': 240,
 'know': 241,
 'give': 242,
 'history': 243,
 'compelling': 244,
 'takes': 245,
 'subject': 246,
 'him': 247,
 'worth': 248,
 'anything': 249,
 'comes': 250,
 'art': 251,
 'entertainment': 252,
 'making': 253,
 'gets': 254,
 'did': 255,
 'least': 256,
 'women': 257,
 'say': 258,
 'three': 259,
 'keep': 260,
 'around': 261,
 'cinematic': 262,
 'true': 263,
 'seem': 264,
 'pretty': 265,
 'ultimately': 266,
 'part': 267,
 'tv': 268,
 'care': 269,
 'think': 270,
 'want': 271,
 'sweet': 272,
 'dark': 273,
 'together': 274,
 'away': 275,
 'theater': 276,
 'laughs': 277,
 'day': 278,
 'works': 279
 ...}

 # the가 몇번등장했는지 보기
 print(tk.word_count["a"])

```

**문자 정수 인덱싱 하기**
```python
# 단어 하나하나가 숫자로 인덱싱 되어있다. 
# 특수문자같은거는 지워지는 옵션이 들어있다. , . 이모티콘
all_texts = tk.text_to_sequences(all_data["Phrase"])
```
**이미지에서 모델에 학습시킬려고 이미지들의크기를 맞춰줬다 머신러닝도 칼럼의 개수를 맞춰줬다 요소의 개수 여기서도 텍스트도 맞춰줘야한다. 
텍스트를 보면 길이가 다 다르다. 즉, 데이터별로 들어가있는 단어의 개수가 다르면 모델에 들어가는 데이터의 사이즈(차원), shape, 모양 를 맞춰줘야한다.
그래서 각가의 데이터의 길이를 맞춰줘야한다.**

```python
# 문장의 길이가 짧으면 단어가 많이 들어가있지 않으면은 어떤 특정한 숫자를 넣어줘서 padding을 늘려주겠다 너무 단어가 많이 들어가있으면 잘라주겠다는 느낌
from keras.preprocessing.sequence import pad_sequences
all_pad = pad_sequences(all_texts)
>>
array([[   0,    0,    0, ...,    3,    2,   42],
       [   0,    0,    0, ...,   13,    1, 3940],
       [   0,    0,    0, ...,    0,    2,  315],
       ...,
       [   0,    0,    0, ...,    2,  118, 4456],
       [   0,    0,    0, ...,    2,  118, 4456],
       [   0,    0,    0, ...,    0,  343, 1623]], dtype=int32)

all_pad.shape
>>(222352, 52)

```

**all_pad를 보면 보지못했던 0이 추가된걸 볼 수 있다. 아까 tokenizer을 했을때 숫자의 시작은 1번이였다 이유가 있다 근데 all_pad에 0이 등장 했다는것은 뭔가 새로운 번호가 등장 했구나 그래서 0번이라는 번호가 알아서 채워준다 
나중에는 0번을 뒤에다가 채워줄 수 있다. shape를 보면 각각의 데이터는 52개의 단어가 들어가있다. 이게 기본값이 뭐냐면 가장 길었던 최대 단어의 개수로 맞춰진다 그래서 단어의 개수가 많지 않은 애들은 0으로 채워진거다.
즉, 가장길었던 문장으로 똑같이 맞춰버리기 때문에 데이터가 날아가지않는다 만약에 size=20으로 해버리면 20개보다 많이 들어간 데이터들은 잘라지게되니 그러면 정보가 날아간다 
그러면 그냥 문장이 가장 길었던거 기준으로 하면 되는거 아니냐 이게 더 안좋을 수 있다. 왜냐면 기본적으로 데이터에 들어가있는 단어의 분포를 보면 기본적으로는 단어의 개수가 많지 않다 그러니까
대부분의 데이터셋에 최대 길이가 예를들어서 100개다라고 하면은 대부분 단어의 개수가 20~30개를 넘지 않는다.  그러기때문에 기본적으로 길지 않는데  괜히 길어지기 때문에 어떤 문제가 생긴다? 예를들어서 사실 데이터가 1만개가 있는데 그중에서 약 100개만 값이 튀어가지고 단어들이 막 많이 들어가있는거다 50개씩
괜히 이 50 몇개때문에 전체의 단어길이가 문장의길이가 단어가 100개씩 들어가게 되면 왜냐면은 원래 그런애들이 있어서 그러면은 다른 데이터는 그렇게 길지않은데 3개 들어가있는데 괜히 50몇개 들어가게 되면 과대적합이 될수 있다. 
특히 원래 길었던 애들을 결국 다 학습을 하게 되니 괜히 과대적합이 될 수 있다. 왜냐면 단어의 개수가 많지도 않은데 하나하나 학습하는게 문제가 될 수. 있다. 그래서 오히려 나중에는 우리 모델에 속도를 개선을 하고 결국에는 과대적합 막고자 보통은 잘라준다. 그래서 지금이야 52 라서 괜찮은데 나중에는 문장 하나의 단어가 500개 1000개씩 들어가있는데 그럴때는 200개로 잘라주는거다
우리 생각에는 괜히 300개 혹은 800개 단어가 날라간거 같아서 안좋을 수 있다고 생각 할 수 있는데 그렇지 않다. 오히려 정보를 날리는게 좋을 수 있다.**

### 텍스트 정보는 문맥이 중요하다
  - 단어들에 순서 앞에 있는지 뒤에있는지 중요하다. 



### train, test 데이터 분리

```python
pad_train = all_pad[:len(train)]
pad_test = all_pad[len(train):]
```

### Modeling()
```python
from keras import Sequential
from keras.layers import *

model = Sequential()
model.add(Embedding(len(tk.word_index)+1), 1, input_length=52)
model.add(Flatten())
model.add(Dense(5, activation="softmax"))
model.compile(optimizer= "adam", loss= "sparse_categorical_crossentropy", metrics = ["acc"])
model.fit(pad_train, train["Sentiment"], epochs=5)
result = model.predict(pad_test)
```

