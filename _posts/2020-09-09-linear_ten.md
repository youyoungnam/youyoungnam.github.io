---
title: Linear Regression tensorflow 만들어보기
categories: deeplearning_study
author_profile: true
---


```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class Model(object):
    def __init__(self):
        # W를 임의로 설정 
        self.W = tf.Variable(5.0)
        # b를 임의로 설정
        self.b = tf.Variable(0.0)

# MSE 구현 
def loss_function(y, predicted):
    return tf.reduce_mean(tf.square(predicted - y))


# 모델 선언

model = Model()


# 

true_w = 3.0
true_b = 2.0
example= 1000 # 노이즈 생성

inputs = tf.random.normal(shape= [example])
noise = tf.random.normal(shape = [example])

outputs = true_w * inputs + true_b + noise

# 시각화 

plt.scatter(inputs, outputs, c="b")
plt.scatter(inputs, model(inputs), c="r")



# 위에서 만든 선형 모델을 가지고 loss측정

loss = loss_function(outputs, model(inputs))
print(loss)



# 선형회귀 모델 훈련 

def train(model, inputs, outputs, learning_rate):
    with tf.GradientTape() as t:
       current_loss = loss_function(outputs, model(inputs))
    W, b = t.gradient(current_loss, [model.W, model.b])

    model.W.assign_sub(learning_rate*W)
    model.b.assign_sub(learning_rate*b)


Ws, bs = [], [] # plot 
epochs = 10


for epoch in range(epochs):
    # numpy()를해주는 이유는 숫자값만 가지고 오기위해서
    Ws.append(model.W.numpy())
    bs.append(model.b.numpy())

    current_loss = loss_function()

     train(model, inputs, outputs, learning_rate=0.1)
     print('epoch: %2d, W=%1.2f, b=%1.2f, loss=%2.5f'%(epoch, Ws[-1], bs[-1], current_loss))



```