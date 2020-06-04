---
layout: post
title: 카테고리형(명목형) 변수를 가진 데이터를 딥러닝으로 풀어보자 (with Bank Marketing Data Set)
date:   2020-06-03 10:26:34
author: sangjae kang
categories: deep-learning
tags:	practice
use_math: true
---
## Objective

많은 캐글대회나 머신러닝 예제들을 보면, 카테고리형(명목형) 변수를 가진 데이터는 대체로 Decision Tree 계통의 모형으로 해결합니다. 하지만 딥러닝도 Decision Tree 만큼이나 효과적으로 카테고리형(명목형) 변수를 우수하게 처리할 수 있습니다. [UCI](https://archive.ics.uci.edu/ml/datasets.php)에서 카테고리형 변수에 대한 데이터셋인 은행 정기예금 가입 데이터를 통해 살펴보도록 하겠습니다.

![](https://imgur.com/RTV6hAo.png)

### 패키지 가져오기

Tensorflow 2 버전으로 작성되어 있습니다. 


<div class="input_area" markdown="1">

```python
import numpy as np
import pandas as pd
import tensorflow as tf

np.set_printoptions(precision=3)
```

</div>

## 데이터 톺아보기

### 데이터 가져오기

[ucl - bank marketing data set](https://archive.ics.uci.edu/ml/datasets/Bank%2BMarketing)에서 제공됩니다. 동일한 데이터를 구글 드라이브에 올려두어서, 아래와 같이 간단히 다운받을 수 있습니다.


<div class="input_area" markdown="1">

```python
from tensorflow.keras.utils import get_file

fpath = get_file("bank-full.csv",
                 "https://docs.google.com/uc?id=16Z2Jyg9BPB8kLeuGDRNLpXZdF77W32_p")
df = pd.read_csv(fpath, sep=';')
```

</div>

### 데이터 파악하기

고객의 여러 정보들을 통해, 해당 고객이 장기 예금에 가입할 것인지($y$)를 예측하는 문제입니다. 총 45211건의 데이터가 존재하고, 입력 변수는 총 16개로 명목형 변수와 수치형 변수가 섞여 있습니다.

* 명목형 변수 : job, marital, education, default, housing, loan, contact, month, poutcome
* 수치형 변수 : age, balance, day, duration, campaign, pdays, previous


<div class="input_area" markdown="1">

```python
print("데이터의 크기 : {df.shape}")
df.head()
```

</div>

Output : 

{:.output_stream}

```
데이터의 크기 : {df.shape}

```




<div markdown="0">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>balance</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>day</th>
      <th>month</th>
      <th>duration</th>
      <th>campaign</th>
      <th>pdays</th>
      <th>previous</th>
      <th>poutcome</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>58</td>
      <td>management</td>
      <td>married</td>
      <td>tertiary</td>
      <td>no</td>
      <td>2143</td>
      <td>yes</td>
      <td>no</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>261</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <th>1</th>
      <td>44</td>
      <td>technician</td>
      <td>single</td>
      <td>secondary</td>
      <td>no</td>
      <td>29</td>
      <td>yes</td>
      <td>no</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>151</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <th>2</th>
      <td>33</td>
      <td>entrepreneur</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>2</td>
      <td>yes</td>
      <td>yes</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>76</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <th>3</th>
      <td>47</td>
      <td>blue-collar</td>
      <td>married</td>
      <td>unknown</td>
      <td>no</td>
      <td>1506</td>
      <td>yes</td>
      <td>no</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>92</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <th>4</th>
      <td>33</td>
      <td>unknown</td>
      <td>single</td>
      <td>unknown</td>
      <td>no</td>
      <td>1</td>
      <td>no</td>
      <td>no</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>198</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
  </tbody>
</table>
</div>
</div>



#### 명목형 변수

총 9개의 명목형 변수가 존재하고, 각 변수 별로 아래와 같은 값들이 존재합니다.


<div class="input_area" markdown="1">

```python
cat_col_names = ["marital",'job','contact','education','month',"poutcome","housing","loan",'default']

for col_name in cat_col_names:
    cat_values = np.unique(df[col_name])
    print(f"{col_name}: {cat_values}\n")
```

</div>

Output : 

{:.output_stream}

```
marital: ['divorced' 'married' 'single']

job: ['admin.' 'blue-collar' 'entrepreneur' 'housemaid' 'management' 'retired'
 'self-employed' 'services' 'student' 'technician' 'unemployed' 'unknown']

contact: ['cellular' 'telephone' 'unknown']

education: ['primary' 'secondary' 'tertiary' 'unknown']

month: ['apr' 'aug' 'dec' 'feb' 'jan' 'jul' 'jun' 'mar' 'may' 'nov' 'oct' 'sep']

poutcome: ['failure' 'other' 'success' 'unknown']

housing: ['no' 'yes']

loan: ['no' 'yes']

default: ['no' 'yes']


```

#### 수치형 변수

총 7개의 수치형 변수가 존재하고, 값의 범위는 아래와 같습니다. 


<div class="input_area" markdown="1">

```python
num_col_names = ['age', 'balance', 'day', 'duration',
                 'campaign','pdays', 'previous']

for col_name in num_col_names:
    print(f"{col_name}: ({df[col_name].min()},{df[col_name].max()})")
```

</div>

Output : 

{:.output_stream}

```
age: (18,95)
balance: (-8019,102127)
day: (1,31)
duration: (0,4918)
campaign: (1,63)
pdays: (-1,871)
previous: (0,275)

```

### 데이터 변환하기

딥러닝 모형에 넣기 전에 우선 데이터들을 전처리해주어야 합니다. 명목형 변수는 인덱스로 변환하는 Label Encoder를 적용하고, 수치형 변수는 정규분포로 변환하는 StandardScaler를 적용합니다.

#### 명목형 변수 변환하기

각 변수 별로 각각 LabelEncoder을 선언하여 적용해 줍니다. 라벨인코더는 각 명목형 변수의 값을 대응하는 인덱스(숫자)에 맵핑시켜주는 전처리 클래스입니다.


<div class="input_area" markdown="1">

```python
from sklearn.preprocessing import LabelEncoder

category_xs = []
category_encoders = []
for col_name in cat_col_names:
    encoder = LabelEncoder()
    encoded_xs = encoder.fit_transform(df[col_name])

    category_xs.append(encoded_xs)
    category_encoders.append(encoder)

category_xs = np.stack(category_xs, axis=1) 
category_xs
```

</div>

Output : 




{:.output_data_text}

```
array([[1, 4, 2, ..., 1, 0, 0],
       [2, 9, 2, ..., 1, 0, 0],
       [1, 2, 2, ..., 1, 1, 0],
       ...,
       [1, 5, 0, ..., 0, 0, 0],
       [1, 1, 1, ..., 0, 0, 0],
       [1, 2, 0, ..., 0, 0, 0]])
```



#### 수치형 변수 변환하기


<div class="input_area" markdown="1">

```python
from sklearn.preprocessing import StandardScaler

numeric_encoder = StandardScaler()
numeric_xs = numeric_encoder.fit_transform(df[num_col_names])
numeric_xs
```

</div>

Output : 




{:.output_data_text}

```
array([[ 1.607,  0.256, -1.298, ..., -0.569, -0.411, -0.252],
       [ 0.289, -0.438, -1.298, ..., -0.569, -0.411, -0.252],
       [-0.747, -0.447, -1.298, ..., -0.569, -0.411, -0.252],
       ...,
       [ 2.925,  1.43 ,  0.143, ...,  0.722,  1.436,  1.05 ],
       [ 1.513, -0.228,  0.143, ...,  0.399, -0.411, -0.252],
       [-0.371,  0.528,  0.143, ..., -0.247,  1.476,  4.524]])
```



#### 타깃 변수($y$) 변환하기

가입할지에 대한 Binary Classification 문제입니다. 가입할 경우 True, 가입하지 않은 경우 False로 학습합니다.


<div class="input_area" markdown="1">

```python
ys = df.y.map({'yes':True,'no':False}).values
ys
```

</div>

Output : 




{:.output_data_text}

```
array([False, False, False, ...,  True, False, False])
```



### 학습데이터와 평가데이터 나누기

성능을 평가하기 위해 학습데이터와 평가데이터를 나눕니다. 이때 라벨의 True/False의 비율이 다르기 때문에, startify를 통해 train/test 내 라벨의 비율을 동일하게 맞춰줍니다.


<div class="input_area" markdown="1">

```python
from sklearn.model_selection import train_test_split

splitted = train_test_split(category_xs, numeric_xs, ys, 
                            test_size=0.1,stratify=ys)

train_category_xs, train_numeric_xs, train_ys = splitted[::2]
test_category_xs, test_numeric_xs, test_ys = splitted[1::2]
```

</div>

## 카테고리형 변수를 처리하는 딥러닝 모형 만들기

#### 모델의 입력값 구성하기

모델의 입력값은 명목형 입력값과 수치형 입력값으로 나뉘어져 있습니다. 카테고리형 입력값은 위에서 인덱스, 즉 정수형 값으로 바꾸어 두었고, 수치형 입력값은 실수형 값을 가지고 있습니다.


<div class="input_area" markdown="1">

```python
from tensorflow.keras.layers import Input

category_inputs = Input((9,), dtype=tf.int32)
numeric_inputs = Input((7,), dtype=tf.float32)
```

</div>

#### 명목형 입력값을 임베딩하기

머신러닝에서 임베딩이란, **모형이 다룰 수 있는 숫자 벡터로 바꾸어주는 작업**을 의미합니다. LabelEncoder로 바꾼 인덱스는 사실 연산을 바로 적용하기에는 부적절합니다. 인덱스(0,1,2,3,..,)으로 이루어진 이 값들은 인덱스 간 크기의 대소에 아무런 의미가 없기 때문입니다. 모형이 다루기 위해서는 적절한 숫자 벡터로 바꾸어 주어야 하는데, 이러한 작업을 통칭해 임베딩이라고 부릅니다. 

딥러닝에서는 별도의 Embedding Layer을 제공합니다. 각 인덱스에 대응하는 임베딩 벡터를 반환합니다. 해당 임베딩 벡터는 모형의 학습 과정 중에서 적절한 값으로 바뀌어갑니다.


<div class="input_area" markdown="1">

```python
from tensorflow.keras.layers import Embedding

# 변수 별 임베딩 크기
embed_size = 4

embeds = []
for idx, col_name in enumerate(cat_col_names):
    # 각 변수 별 카테고리의 갯수
    category_size = len(category_encoders[idx].classes_)
    
    # 각 변수 별로 임베딩 레이어 적용 (-> 각 변수 내 카테고리별로 임베딩 값들이 모델에서 학습)
    category_embeded = Embedding(
        category_size, embed_size, name=col_name+'_embed')(category_inputs[:,idx])
    
    embeds.append(category_embeded)
embeds    
```

</div>

Output : 




{:.output_data_text}

```
[<tf.Tensor 'marital_embed/Identity:0' shape=(None, 4) dtype=float32>,
 <tf.Tensor 'job_embed/Identity:0' shape=(None, 4) dtype=float32>,
 <tf.Tensor 'contact_embed/Identity:0' shape=(None, 4) dtype=float32>,
 <tf.Tensor 'education_embed/Identity:0' shape=(None, 4) dtype=float32>,
 <tf.Tensor 'month_embed/Identity:0' shape=(None, 4) dtype=float32>,
 <tf.Tensor 'poutcome_embed/Identity:0' shape=(None, 4) dtype=float32>,
 <tf.Tensor 'housing_embed/Identity:0' shape=(None, 4) dtype=float32>,
 <tf.Tensor 'loan_embed/Identity:0' shape=(None, 4) dtype=float32>,
 <tf.Tensor 'default_embed/Identity:0' shape=(None, 4) dtype=float32>]
```



#### 명목형 변수와 수치형 변수 합치기

위와 같이 임베딩된 명목형 변수는 수치형 변수와 같이 연산에 적절한 벡터로 바뀌어져 있습니다. 수치형 변수와 이제 합치도록 하겠습니다.


<div class="input_area" markdown="1">

```python
from tensorflow.keras.layers import Concatenate

inputs_list = embeds + [numeric_inputs]
concats = Concatenate(name='embed_concat')(inputs_list)
concats
```

</div>

Output : 




{:.output_data_text}

```
<tf.Tensor 'embed_concat/Identity:0' shape=(None, 43) dtype=float32>
```



#### 딥러닝 모형 구성하기

3층 신경망으로 구성하도록 하겠습니다. Overfitting을 방지하기 위해, L2 정규화와 Dropout Layer을 추가하였습니다.


<div class="input_area" markdown="1">

```python
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model

hidden = Dense(50,activation='relu', name='hidden1',
               kernel_regularizer=l2(1e-5))(concats)
hidden = Dropout(0.3, name='dropout1')(hidden)
hidden = Dense(50,activation='relu', name='hidden2',
               kernel_regularizer=l2(1e-5))(hidden)
hidden = Dropout(0.3, name='dropout2')(hidden)
output = Dense(1, activation='sigmoid', name='output',
               kernel_regularizer=l2(1e-5))(hidden)

model = Model([category_inputs, numeric_inputs], output)
```

</div>

#### 모델 학습시키기

모형의 손실함수와 옵티마이저를 설정한 후, 데이터를 통해 모델을 학습시켜 보도록 하겠습니다.


<div class="input_area" markdown="1">

```python
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy

model.compile(optimizer=Adagrad(1e-2),
              loss=BinaryCrossentropy(),
              metrics=[BinaryAccuracy()])

train_xs = [train_category_xs,train_numeric_xs]
model.fit(x=train_xs, y=train_ys,
          batch_size=64, epochs=100, 
          validation_split=0.1, verbose=0);
```

</div>

#### 모델 평가하기

테스트 데이터를 통해 모형의 정확도를 산출해보도록 하겠습니다.



<div class="input_area" markdown="1">

```python
loss, acc = model.evaluate(x=[test_category_xs, test_numeric_xs], y=test_ys, verbose=0)
print(f"딥러닝 모형의 정확도 : {acc:.3%}")
```

</div>

Output : 

{:.output_stream}

```
딥러닝 모형의 정확도 : 90.513%

```

### c.f) Random Forest 모형과 비교해보기

대표적인 Decision Tree 모형인 RandomForest로도 데이터를 동일하게 두고 학습시킨 결과입니다. 복잡하지 않은 단순한 형태의 딥러닝 만으로도 충분히 RandomForest와 비슷하게 성능이 나옵니다. 


<div class="input_area" markdown="1">

```python
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier()

# 모형 학습시키기
train_xs = np.concatenate([train_category_xs,train_numeric_xs],axis=1)
rf_clf.fit(train_xs, train_ys)

# 모형 평가하기 
test_xs = np.concatenate([test_category_xs,test_numeric_xs],axis=1)
rf_clf.score(test_xs, test_ys)
```

</div>

Output : 




{:.output_data_text}

```
0.9053516143299425
```


