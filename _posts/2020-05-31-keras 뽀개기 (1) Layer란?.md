---
layout: post
title: keras 뽀개기 (1) Layer란?
date:   2020-05-31 22:43:23
author: sangjae kang
categories: deep-learning
tags:	tensorflow
use_math: true
---
딥러닝에서 모델은 **층(Layer)** 으로 구성합니다. 입력층, 은닉층, 출력층을 순서에 맞게 연결하여 하나의 모형을 구성합니다. keras도 똑같이 층(Layer)을 기준으로 모델을 작성합니다. keras의 `Layer`을 하나씩 뜯어보며 어떻게 동작하는지를 뼈 속까지 파악해보도록 하겠습니다.




## keras의 Layer란


딥러닝에서의 **층**은 기능의 단위입니다. **층** 별로 각기 다른 기능들을 수행하죠. 우리가 이미지에서 특징을 추출할 때는 보통 합성곱층(Conv)을 쓰고, 시계열에서 특징을 추출할 때는 RNN을 이용합니다. 이러한 층들은 가중치(weight)와 연산으로 이루어져 있습니다. Keras에서는 이러한 가중치와 연산을 하나의 **Layer**로 묶어서 관리합니다.

### Layer 생성하기

Keras는 Tensorflow의 생태계 안으로 들어왔습니다. 여기에서는 tensorflow 내 keras를 이용합니다. 많은 부분 비슷하지만, Tensorflow와의 호환성이 좀 더 잘 되기 때문에 개인적으로 tf.keras를 선호합니다.

<div class="prompt input_prompt">
In&nbsp;[1]:
</div>

<div class="input_area" markdown="1">

```python
from tensorflow.keras.layers import Dense

hidden_layer = Dense(3, activation='relu', name='hidden')
output_layer = Dense(1, activation='sigmoid', name='output')

print(hidden_layer)
print(output_layer)
```

</div>

{:.output_stream}

```
<tensorflow.python.keras.layers.core.Dense object at 0x109dd3908>
<tensorflow.python.keras.layers.core.Dense object at 0x109dd37b8>

```

위와 같이 생성된 레이어는 하나의 **함수**처럼 동작합니다. 레이어는 `np.ndarray` 혹은 `tf.Tensor`를 받아 내부 가중치와 함께 연산 후 결과를 반환합니다.

<div class="prompt input_prompt">
In&nbsp;[2]:
</div>

<div class="input_area" markdown="1">

```python
import tensorflow as tf

x = tf.constant([[1,2]],tf.float32)
hidden_layer(x)
```

</div>




{:.output_data_text}

```
<tf.Tensor: shape=(1, 3), dtype=float32, numpy=array([[0.       , 1.5077845, 1.5742133]], dtype=float32)>
```



해당 레이어의 내 가중치는 `.weights`를 통해 가져올 수 있습니다. 이 가중치 행렬은 모델의 학습 과정 중에서 갱신합니다.

<div class="prompt input_prompt">
In&nbsp;[3]:
</div>

<div class="input_area" markdown="1">

```python
W_h, b_h = hidden_layer.weights
print("W : ", W_h)
print("b : ", b_h)
```

</div>

{:.output_stream}

```
W :  <tf.Variable 'hidden/kernel:0' shape=(2, 3) dtype=float32, numpy=
array([[-0.64304525, -0.5670427 ,  0.46216154],
       [-0.25390232,  1.0374136 ,  0.55602586]], dtype=float32)>
b :  <tf.Variable 'hidden/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>

```

`hidden_layer`에서의 연산은 내부에서는 아래처럼 작동합니다.

<div class="prompt input_prompt">
In&nbsp;[4]:
</div>

<div class="input_area" markdown="1">

```python
tf.nn.relu(x @ W_h + b_h)
```

</div>




{:.output_data_text}

```
<tf.Tensor: shape=(1, 3), dtype=float32, numpy=array([[0.       , 1.5077845, 1.5742133]], dtype=float32)>
```



### Layer 내 가중치 생성하기

현재 `hidden_layer`의 가중치는 위처럼 존재합니다. 그러면 `output_layer`의 가중치는 어떻게 되어 있을까요?

<div class="prompt input_prompt">
In&nbsp;[5]:
</div>

<div class="input_area" markdown="1">

```python
output_layer.weights
```

</div>




{:.output_data_text}

```
[]
```



빈 리스트, 즉 가중치가 없다고 나옵니다. 가중치의 크기는 입력의 크기를 알아야 비로소 결정할 수 있는데, 아직 입력의 크기가 어떻게 될지 정해지지 않았기 때문에 가중치가 생성되지 않았습니다.

그래서 입력값의 크기에 따라 가중치를 만드는 메소드로 `.build(input_shape)`가 존재합니다.

<div class="prompt input_prompt">
In&nbsp;[6]:
</div>

<div class="input_area" markdown="1">

```python
output_layer.build((None, 3)) # input의 크기 : (None, 3)

W_o, b_o = output_layer.weights
print("W : ", W_o)
print("b : ", b_o)
```

</div>

{:.output_stream}

```
W :  <tf.Variable 'kernel:0' shape=(3, 1) dtype=float32, numpy=
array([[-0.5790425],
       [-0.6597057],
       [ 0.8126277]], dtype=float32)>
b :  <tf.Variable 'bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>

```

입력값의 크기를 커지면 가중치의 크기도 커집니다. 

<div class="prompt input_prompt">
In&nbsp;[7]:
</div>

<div class="input_area" markdown="1">

```python
output_layer.build((None, 5)) # input의 크기 : (None, 3)

W_o, b_o = output_layer.weights
print("W : ", W_o)
print("b : ", b_o)
```

</div>

{:.output_stream}

```
W :  <tf.Variable 'kernel:0' shape=(5, 1) dtype=float32, numpy=
array([[-0.55155826],
       [-0.6368439 ],
       [-0.23670721],
       [-0.7981725 ],
       [ 0.67237806]], dtype=float32)>
b :  <tf.Variable 'bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>

```

그런데 보통 이런 `build(input_shape)`를 통해 가중치를 정해주기 보다, 첫번째 연산 시의 입력값 형태에 맞춰서 가중치를 초기화하는 방법을 택합니다. 우리가 처음에 `hidden_layer`의 가중치를 초기화시켰을 때처럼, 처음 값을 넣으면 자동으로 그 크기에 맞춰 가중치의 크기가 결정됩니다.

<div class="prompt input_prompt">
In&nbsp;[8]:
</div>

<div class="input_area" markdown="1">

```python
output_layer = Dense(1, activation='sigmoid', name='output')

# call 호출 전
print("Before call : ", output_layer.weights)

# (1,3)짜리 입력 행렬을 넣기
x = tf.constant([[1,3,2]], tf.float32)
output_layer(x)

print("After call : ", output_layer.weights)
```

</div>

{:.output_stream}

```
Before call :  []
After call :  [<tf.Variable 'output/kernel:0' shape=(3, 1) dtype=float32, numpy=
array([[-1.1762437 ],
       [ 1.1605917 ],
       [-0.35425937]], dtype=float32)>, <tf.Variable 'output/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]

```

### Layer을 통해 연산하기

아래와 같이 입력값이 존재한다고 해봅시다.

<div class="prompt input_prompt">
In&nbsp;[9]:
</div>

<div class="input_area" markdown="1">

```python
inputs = tf.constant([[0.2,0.5],
                      [0.3,0.6],
                      [0.1,-.3]], tf.float32)
```

</div>

그리고 현재 우리의 모형은 아래와 같은 2층 신경망이라고 해봅시다. 우리가 구성해야 하는 레이어는 은닉층과 출력층입니다. (입력층은 사실 데이터에 불과하죠)

<img src="https://imgur.com/CBgGgkb.png" width="300">

아래와 같이 레이어를 선언할 수 있습니다.

<div class="prompt input_prompt">
In&nbsp;[10]:
</div>

<div class="input_area" markdown="1">

```python
hidden_layer = Dense(3, activation='relu', name='hidden')
output_layer = Dense(1, activation='sigmoid', name='output')
```

</div>

입력값을 따라 출력값까지 가져가는 순전파 과정은 Keras를 통해 할 수 있습니다.

<div class="prompt input_prompt">
In&nbsp;[11]:
</div>

<div class="input_area" markdown="1">

```python
x = hidden_layer(inputs)
x = output_layer(x)
x #  순전파 결과 
```

</div>




{:.output_data_text}

```
<tf.Tensor: shape=(3, 1), dtype=float32, numpy=
array([[0.5283282 ],
       [0.5290259 ],
       [0.48786113]], dtype=float32)>
```



### layer 내 가중치를 학습시키기

딥러닝에서 가중치를 학습시키기 위해서 쓰는 방법은 주로 **경사하강법**입니다.

$
\mbox{경사하강법} \\
W := W - \alpha \frac{\partial L}{\partial W}
$

그리고 경사하강법을 적용하기 위해 필요한 기울기 정보($\frac{\partial L}{\partial W}$)는 역전파를 통해 구할 수 있습니다. tf2.0 버전부터는 `tf.GradientTape()`를 통해 역전파를 수행합니다.

<div class="prompt input_prompt">
In&nbsp;[12]:
</div>

<div class="input_area" markdown="1">

```python
y_true = tf.constant([[1.],[0.],[1.]],tf.float32) # 정답 Label

with tf.GradientTape() as tape:
    # 순전파 과정
    z = hidden_layer(inputs)
    y_pred = output_layer(z)    
    
    # 손실함수
    loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
```

</div>

현재 가중치는 은닉층의 가중치와 출력층의 가중치로 구성되어 있습니다.

<div class="prompt input_prompt">
In&nbsp;[13]:
</div>

<div class="input_area" markdown="1">

```python
# 모든 가중치 가져오기
weights = hidden_layer.weights + output_layer.weights

weights
```

</div>




{:.output_data_text}

```
[<tf.Variable 'hidden/kernel:0' shape=(2, 3) dtype=float32, numpy=
 array([[ 0.6005299 , -0.4234662 ,  0.25702417],
        [-0.5789593 ,  0.50691354,  0.5335573 ]], dtype=float32)>,
 <tf.Variable 'hidden/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>,
 <tf.Variable 'output/kernel:0' shape=(3, 1) dtype=float32, numpy=
 array([[-0.2077725 ],
        [ 0.7557734 ],
        [-0.04435456]], dtype=float32)>,
 <tf.Variable 'output/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]
```



위와 같이 순전파 과정을 `tf.GradientTape()`으로 감싸줌으로써 텐서플로우 내부에서는 중간 연산과정들이 메모리에 저장됩니다. 해당 정보를 바탕으로 우리는 연산을 진행할 수 있습니다.

<div class="prompt input_prompt">
In&nbsp;[14]:
</div>

<div class="input_area" markdown="1">

```python
# 가중치의 기울기 계산하기 (역전파)
grads = tape.gradient(loss, weights)

# 경사하강법 적용하기
lr = 1e-1
for weight, grad in zip(weights, grads):
    weight.assign(weight - lr*grad)
```

</div>

데이터에 따라 모형 내 Weight들이 갱신되었다는 것을 아래를 통해 확인할 수 있습니다.

<div class="prompt input_prompt">
In&nbsp;[15]:
</div>

<div class="input_area" markdown="1">

```python
# 모든 가중치 가져오기
weights = hidden_layer.weights + output_layer.weights

weights
```

</div>




{:.output_data_text}

```
[<tf.Variable 'hidden/kernel:0' shape=(2, 3) dtype=float32, numpy=
 array([[ 0.59946585, -0.42833138,  0.2573097 ],
        [-0.57576704,  0.500748  ,  0.53391916]], dtype=float32)>,
 <tf.Variable 'hidden/bias:0' shape=(3,) dtype=float32, numpy=array([-0.01064083, -0.00433467,  0.00025439], dtype=float32)>,
 <tf.Variable 'output/kernel:0' shape=(3, 1) dtype=float32, numpy=
 array([[-0.19580172],
        [ 0.754364  ],
        [-0.05036185]], dtype=float32)>,
 <tf.Variable 'output/bias:0' shape=(1,) dtype=float32, numpy=array([0.04547846], dtype=float32)>]
```



### (5) Layer의 Hyper Parameter 가져오기

층의 구조 및 형태는 사람이 설계합니다. 예를 들어 unit 수, activation의 종류, bias의 유무가 바로 사람이 설계해야 하는 요소, Hyper-Parameter입니다. 레이어의 해당 정보를 가져오기 위해서는 `layer.get_config()`를 이용하면 됩니다.

<div class="prompt input_prompt">
In&nbsp;[16]:
</div>

<div class="input_area" markdown="1">

```python
hidden_layer.get_config()
```

</div>




{:.output_data_text}

```
{'name': 'hidden',
 'trainable': True,
 'dtype': 'float32',
 'units': 3,
 'activation': 'relu',
 'use_bias': True,
 'kernel_initializer': {'class_name': 'GlorotUniform',
  'config': {'seed': None}},
 'bias_initializer': {'class_name': 'Zeros', 'config': {}},
 'kernel_regularizer': None,
 'bias_regularizer': None,
 'activity_regularizer': None,
 'kernel_constraint': None,
 'bias_constraint': None}
```



<div class="prompt input_prompt">
In&nbsp;[17]:
</div>

<div class="input_area" markdown="1">

```python
output_layer.get_config()
```

</div>




{:.output_data_text}

```
{'name': 'output',
 'trainable': True,
 'dtype': 'float32',
 'units': 1,
 'activation': 'sigmoid',
 'use_bias': True,
 'kernel_initializer': {'class_name': 'GlorotUniform',
  'config': {'seed': None}},
 'bias_initializer': {'class_name': 'Zeros', 'config': {}},
 'kernel_regularizer': None,
 'bias_regularizer': None,
 'activity_regularizer': None,
 'kernel_constraint': None,
 'bias_constraint': None}
```



### (6) Custom Layer, 나만의 연산층 구성하기

딥러닝 레이어를 구성하기 위해서는 

* `.call()` : 어떤 연산을 수행할 것인가
* `.build()` : 어떤 가중치로 구성할 것인가

가 정의되어야 합니다. 그리고 이후에 레이어를 저장하기 위해서는 `get_config()`도 아래와 같이 구성해주어야 합니다. 해당 레이어의 hyper-parameter 정보를 `get_config()`에 담아주어야 json 파일로 변경할 때 올바르게 저장이 됩니다.

<div class="prompt input_prompt">
In&nbsp;[18]:
</div>

<div class="input_area" markdown="1">

```python
from tensorflow.keras.layers import Layer

class MyLayer(Layer):
    def __init__(self, num_units, **kwargs):
        self.num_units = num_units
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        # 가중치를 정의
        self.w = self.add_weight(shape=(input_shape[1],self.num_units),
                                 name='kernel')
        self.b = self.add_weight(shape=(self.num_units,),
                                 initializer='zeros',
                                 name='bias')
        super().build(input_shape)
        
    def call(self, inputs):
        # 연산을 정의
        return tf.nn.relu(inputs @ self.w+ self.b) 
    
    def get_config(self):
        # hyper parameter를 정의
        config = super().get_config()
        config.update({
            'num_units':self.num_units
        })
        return config
```

</div>

<div class="prompt input_prompt">
In&nbsp;[19]:
</div>

<div class="input_area" markdown="1">

```python
my_dense_layer = MyLayer(5)

x = tf.constant([[1.,2.,3.]],tf.float32)

my_dense_layer(x)
```

</div>




{:.output_data_text}

```
<tf.Tensor: shape=(1, 5), dtype=float32, numpy=array([[0.     , 0.     , 2.97744, 0.     , 0.     ]], dtype=float32)>
```



<div class="prompt input_prompt">
In&nbsp;[20]:
</div>

<div class="input_area" markdown="1">

```python
my_dense_layer.get_config()
```

</div>




{:.output_data_text}

```
{'name': 'my_layer', 'trainable': True, 'dtype': 'float32', 'num_units': 5}
```



## 마무리 

케라스는 레이어 단위로 구조화합니다. 텐서플로우 1.x버전에서는 연산과 가중치를 별개로 나누어서 작성했기 때문에 자유도가 매우 높았지만, 복잡한 모형을 만들 때 코드가 매우 복잡해지는 문제를 야기했습니다. 케라스에서는 레이어 단위로 연산과 가중치를 묶어 관리하기 때문에, 상대적으로 자유도는 좀 줄었지만 훨씬 더 간결하게 모형을 작성할 수 있습니다.
