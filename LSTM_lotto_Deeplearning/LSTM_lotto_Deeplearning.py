from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import os
import tensorflow as tf
from tensorflow.python.saved_model.load import metrics
import chardet
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

dir = os.getcwd()
print(dir)

with open("./lotto_till_1114.csv", 'rb') as f:
    rawdata = f.read()
    result = chardet.detect(rawdata)
    encoding = result['encoding']
    
data = np.genfromtxt("./lotto_till_1114.csv", delimiter=",", dtype='str', encoding='utf-8')

#이거 csv파일이라서 무조건 string으로 저장된다고 chat gpt가 그러는데...
#인터넷에 아니라고 하기도하는듯 우선 1번째 string에 \ufeff가 있어서 정수형 변환이 안됨. 그래서 해당 코드 넣음
def safe_convert_to_int(s):
    # BOM 제거
    s = s.replace('\ufeff', '')
    try:
        return np.int64(s)
    except ValueError:
        return None

vectorized_convert = np.vectorize(safe_convert_to_int)
integer_data = vectorized_convert(data)

data = integer_data

row_count = len(data)

def ToOnehotencoding(numbers):
    ohbin = np.zeros(45) 
    for i in range(6): 
        ohbin[int(numbers[i]) - 1] = 1 
        
    return ohbin

def ToNumber(ohbin):
    numbers = []
    for i in range(len(ohbin)):
        if ohbin[i] == 1.0:
            numbers.append(i+1)
        
    return numbers

numbers = data[:, 1:7]
ohbins = list(map(ToOnehotencoding, numbers))

x_samples = ohbins[0:row_count - 1]
y_samples = ohbins[1:row_count]

train_idx = (0, 800)
val_idx = (801, 900)
test_idx = (901, len(x_samples))

model = keras.Sequential([
    keras.layers.LSTM(128, batch_input_shape=(1, 1, 45), return_sequences=False, stateful=True),
    keras.layers.Dense(45, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

train_loss = []
train_acc = []
val_loss = []
val_acc = []

for epoch in range(100):
    model.reset_states() 

    batch_train_loss = []
    batch_train_acc = []
    
    for i in range(train_idx[0], train_idx[1]):
        
        xs = x_samples[i].reshape(1, 1, 45)
        ys = y_samples[i].reshape(1, 45)
        
        loss, acc = model.train_on_batch(xs, ys) 

        batch_train_loss.append(loss)
        batch_train_acc.append(acc)

    train_loss.append(np.mean(batch_train_loss))
    train_acc.append(np.mean(batch_train_acc))

    batch_val_loss = []
    batch_val_acc = []

    for i in range(val_idx[0], val_idx[1]):

        xs = x_samples[i].reshape(1, 1, 45)
        ys = y_samples[i].reshape(1, 45)
        
        loss, acc = model.test_on_batch(xs, ys) 
        
        batch_val_loss.append(loss)
        batch_val_acc.append(acc)

    val_loss.append(np.mean(batch_val_loss))
    val_acc.append(np.mean(batch_val_acc))
    
    print('epoch {0:4d} train acc {1:0.3f} loss {2:0.3f} val acc {3:0.3f} loss {4:0.3f}'.format(epoch, np.mean(batch_train_acc), np.mean(batch_train_loss), np.mean(batch_val_acc), np.mean(batch_val_loss)))


mean_price = [np.mean(data[87:, 8]),
           np.mean(data[87:, 9]),
           np.mean(data[87:, 10]),
           np.mean(data[87:, 11]),
           np.mean(data[87:, 12])]

def calc_reward(true_numbers, true_bonus, pred_numbers):

    count = 0

    for ps in pred_numbers:
        if ps in true_numbers:
            count += 1

    if count == 6:
        return 0, mean_price[0]
    elif count == 5 and true_bonus in pred_numbers:
        return 1, mean_price[1]
    elif count == 5:
        return 2, mean_price[2]
    elif count == 4:
        return 3, mean_price[3]
    elif count == 3:
        return 4, mean_price[4]

    return 5, 0

def gen_numbers_from_probability(nums_prob):

    ball_box = []

    #번호가 45개니까 1번부터 쭉 시작해서 뽑기
    for n in range(45):
        ball_count = int(nums_prob[n] * 100 + 1)
        ball = np.full((ball_count), n+1) 
        ball_box += list(ball)

    selected_balls = []

    while True:
        
        if len(selected_balls) == 6:
            break
        
        ball_index = np.random.randint(len(ball_box), size=1)[0]
        ball = ball_box[ball_index]

        if ball not in selected_balls:
            selected_balls.append(ball)

    return selected_balls

train_total_reward = []
train_total_grade = np.zeros(6, dtype=int)

val_total_reward = []
val_total_grade = np.zeros(6, dtype=int)

test_total_reward = []
test_total_grade = np.zeros(6, dtype=int)

model.reset_states()

print('[No. ] 1st 2nd 3rd 4th 5th 6th Rewards')

for i in range(len(x_samples)):
    xs = x_samples[i].reshape(1, 1, 45)
    # 모델의 출력값을 predict 함.
    ys_pred = model.predict_on_batch(xs) 
    
    sum_reward = 0
    # numpy에 들어있는 6개까지 변수 뺌
    sum_grade = np.zeros(6, dtype=int) 

    # 일주일에 만원치 살거니까 10번 반복
    for n in range(10): 
        numbers = gen_numbers_from_probability(ys_pred[0])
        
        grade, reward = calc_reward(data[i+1,1:7], data[i+1,7], numbers) 
        
        sum_reward += reward
        sum_grade[grade] += 1

        if i >= train_idx[0] and i < train_idx[1]:
            train_total_grade[grade] += 1
        elif i >= val_idx[0] and i < val_idx[1]:
            val_total_grade[grade] += 1
        elif i >= test_idx[0] and i < test_idx[1]:
            val_total_grade[grade] += 1
    
    if i >= train_idx[0] and i < train_idx[1]:
        train_total_reward.append(sum_reward)
    elif i >= val_idx[0] and i < val_idx[1]:
        val_total_reward.append(sum_reward)
    elif i >= test_idx[0] and i < test_idx[1]:
        test_total_reward.append(sum_reward)
                        
    print('[{0:4d}] {1:3d} {2:3d} {3:3d} {4:3d} {5:3d} {6:3d} {7:15,d}'.format(i+1, sum_grade[0], sum_grade[1], sum_grade[2], sum_grade[3], sum_grade[4], sum_grade[5], int(sum_reward)))

# 최대 100번 반복
for epoch in range(100):

    model.reset_states() 

    for i in range(len(x_samples)):
        
        xs = x_samples[i].reshape(1, 1, 45)
        ys = y_samples[i].reshape(1, 45)
        
        loss, acc = model.train_on_batch(xs, ys) 

        batch_train_loss.append(loss)
        batch_train_acc.append(acc)

    train_loss.append(np.mean(batch_train_loss))
    train_acc.append(np.mean(batch_train_acc))

    print('epoch {0:4d} train acc {1:0.3f} loss {2:0.3f}'.format(epoch, np.mean(batch_train_acc), np.mean(batch_train_loss)))  
    
    
    
# 최근 회차까지 predict 시킨거로 다음주꺼 예측

print('receive numbers')

xs = x_samples[-1].reshape(1, 1, 45)

ys_pred = model.predict_on_batch(xs)

list_numbers = []

for n in range(10):
    numbers = gen_numbers_from_probability(ys_pred[0])
    numbers.sort()
    print('{0} : {1}'.format(n, numbers))
    list_numbers.append(numbers)