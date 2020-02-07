---
title : Github블로그 에러(error)에 당황
categories:
  - Git    
author_profile: true

---
# 처음만난 github 에러(error)
나는 데이터사이언스를 공부를 시작하고 나도 다른사람들처럼 개발블로그를 만들고 싶었다.

 그러다가 우연히 유튜브에 [안수빈(고려대)](https://subinium.github.io/)님이[T-academy]("https://www.youtube.com/watch?v=eCv_bh-Ax-Q&t=2831s")에서 강의하신 동영상을 봤다. 내가 가장 하고 싶었던 블로그를 어떻게 시작해야 하는지 잘 설명해주셨다.

#### 에러내용

>  error: failed to push some refs to ~~~~~~
hint: Updates were rejected because the tip of your current branch is behind
hint: its remote counterpart. Integrate the remote changes (e.g.
hint: 'git pull ...') before pushing again.
hint: See the 'Note about fast-forwards' in 'git push --help' for details      

git push를 하다 만난 에러였다.

너무나 당황스러웠다. 어디서 어떤 문제가 생겼는지 몰랐고 한 2시간? 정도 해결을 못하고 다음날 와서 구글에 에러를 복사한뒤 검색해서 겨우 해결했다.

나도 이제 깃허브 블로그를 만들었으니 다음번에는 까먹지 않게 여기다 기록을하자.


## 첫번째 구글에서 찾은 방법은
1. git remote add origin repository
2. git pull 
3. git pull origin master
4. git push origin master



## 두번째 방법은
1. git init
2. git add .
3. git commit -m "아무이름"
4. git remote add origin 자기자신 깃헙주소/repository
5. 연결이 잘됬는지 확인을 위해
5. git remote -v
7. git push --force --set-upstream origin master



두번째 방법으로 시도해보니 해결이 되었다.

