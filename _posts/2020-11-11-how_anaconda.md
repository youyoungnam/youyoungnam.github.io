---
title: Ubuntu에 Anaconda 설치하기 
categories: Linux/Ubuntu
author_profile: true
---



**이번에는 우분투에 Anaconda 설치하는 방법을 정리할것이다. 아무래도 우분투를 처음 사용해서 그런지 다양한 에러를 만나서 설치하는데 오래걸렸다..
이제 우분투에 Anaconda를 설치하는 방법을 알아보자!!**


**첫번째로 아나콘다 설치를 위해 Anaconda 다운로드 사이트로 들어가보자! https://www.anaconda.com/products/individual <<-- 주소**

<img src="/assets/images/anains.PNG">


**위 사진에서 맨 오른쪽 Linux를 다운로드하자 다운로드가 끝났다면 확인해야 할 작업이 있다. 다운받은 아나콘다 파일에 hash코드와 아나콘다 hash 페이지에 있는 hash랑 같은지 확인 해야한다.**

**Hash코드 확인방법**
  - sha256sum 다운받은 리눅스버전 아나콘다파일 
  - https://docs.anaconda.com/anaconda/install/hashes/all/ <-- 아나콘다 Hash파일 확인 

<img src="/aseets/images/hash1.PNG">

**Hash가 다운로드한 파일이랑 같다면 이제 설치를하자**

```python
bash Anaconda3-2020.07-Linux-x86_64.sh 
```

<img src="/assets/images/bash2.PNG">

**위 사진처럼 코드를 입력하고 Enter을 눌러주면 설치가 될것이다. 중간에 Yes or no가 나오는데 Yes를 해주면 된다.**



**설치가 끝났다면 새로운 PATH 환경 변수를 로드 해야한다.**


```python
source ~/.bashrc
conda info

```



**도움을 받은 블로그 https://bddung.tistory.com/259 <<--**





