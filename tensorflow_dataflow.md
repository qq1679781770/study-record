### tensorflow  flow
#### 例子1：简单模型训练mnist数据集

利用简单模型简单数据研究训练过程和checkpoint保存的内容

下面是具体的代码
```
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)

model.save("model.h5")
```

其中`x_train,y_train`是训练集,维度为`(60000, 28, 28)`，`x_test,y_test`是训练集
模型`model`由4层网络层组成，分别为`Flatten、Dense、Dropout、Dense`
在模型训练之前还需要在编译步骤`（compile）`添加一些配置：`loss`定义损失函数，将函数最小化以使模型拟合，`optimizer`是模型根据数据和损失函数进行更新的方式，`metrics`用于监控训练和测试步骤
`model.fit`更加训练集训练模型，`epoch`指定训练的轮数
`model.save`保存模型

##### 训练过程
###### 首先简单介绍模型是怎么由输入计算最终输出的过程

![](/home/jsxnh/图片/flow.png)
假设`batch size=32`，一条数据的大小为`（28,28）`那么输入的数据shape为`（32,28,28）`，第一层网络层为Flatten，一条数据从一个二维数组(包含着28x28个像素)转换成为一个包含着28 * 28 = 784个像素的一维数组，数据经过第一层后形状变成`（32,784）`。第二层网络层为全连接层(Dense)，激活函数为relu，因此对数据执行`relu(dot(x,w)+b)`的操作，其中w和b为Dense层的参数,数据形状变为`(32,512)`。第三层为dropout层，指在训练期间随机“丢弃”（即设置为 0）该层的多个输出特征，即对该层的输出的值中随机设置为0，此层是为了不让模型过拟合，因此形状不变。最后一层同样是Dense层，激活函数为`softmax`，经过此层后最终输出为`（32,10）`的数据。

代码中由`model.fit(x_train, y_train, epochs=5)`一句执行训练。`model.fit`调用`training_arrays.fit_loop`，`training_arrays.fit_loop`是对`tensorflow/python/keras/engine/training_arrays.py`中`model_iteration`函数的封装，因此`model_iteration`函数是python层中包含训练过程的函数，`model_iteration`定义如下:
```
def model_iteration(model,
                    inputs,
                    targets=None,
                    sample_weights=None,
                    batch_size=None,
                    epochs=1,
                    verbose=1,
                    callbacks=None,
                    val_inputs=None,
                    val_targets=None,
                    val_sample_weights=None,
                    shuffle=True,
                    initial_epoch=0,
                    steps_per_epoch=None,
                    validation_steps=None,
                    validation_freq=1,
                    mode=ModeKeys.TRAIN,
                    validation_in_fit=False,
                    prepared_feed_values_from_dataset=False,
                    steps_name='steps',
                    **kwargs):
```
对每个`epoch`，有`steps_per_epoch`步`iteration`，每个`iteration`会执行一次训练，`batch_outs = f(actual_inputs)`就是每个iteration中对输入数据进行训练的代码，
这个函数最终会调用c的函数，在c后端执行图的执行。`void TF_SessionRunCallable(TF_Session* session, int64_t handle,
                           PyObject* feed_values, PyObjectVector* out_values,
                           TF_Buffer* run_metadata, TF_Status* status)`这个函数就是连接了前端python的f函数，然后再经过调用去执行图的计算，返回给前端python结果。


##### CheckPoint
checkpoint是在模型训练的过程中保存的，tensorflow有两种保存的格式，一种是HDF5文件，另一种是tensorflow自己创建的格式。checkpoint的文件中保存的对象有模型结构，参数和优化器的状态，我们也可以只保存模型参数。接下来简单介绍一下保存的具体内容是怎么样的。
对于模型元数据，有两个重要的部分，分别是模型配置和训练的配置，具体如下：
model_config
```
{'class_name': 'Sequential',
 'config':
	{'name': 'sequential',
    'layers':
    	[{'class_name': 'Flatten', 'config': {'name': 'flatten', 'trainable': True, 'batch_input_shape': (None, 28, 28), 'dtype': 'float32', 'data_format': 'channels_last'}},
        {'class_name': 'Dense', 'config': {'name': 'dense', 'trainable': True, 'dtype': 'float32', 'units': 512, 'activation': 'relu', 'use_bias': True, 'kernel_initializer':{'class_name': 'GlorotUniform', 'config': {'seed': None, 'dtype': 'float32'}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {'dtype': 'float32'}}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None}},
        {'class_name': 'Dropout', 'config': {'name': 'dropout', 'trainable': True, 'dtype': 'float32', 'rate': 0.2, 'noise_shape': None, 'seed': None}},
        {'class_name': 'Dense', 'config': {'name': 'dense_1', 'trainable': True, 'dtype': 'float32', 'units': 10, 'activation': 'softmax', 'use_bias': True, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None, 'dtype': 'float32'}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {'dtype': 'float32'}}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None}}]
    }
}
```
train_config
```
{'loss': 'sparse_categorical_crossentropy', 'metrics': ['accuracy'], 'weighted_metrics': None, 'sample_weight_mode': None, 'loss_weights': None, 'optimizer_config': {'class_name': 'Adam', 'config': {'name': 'Adam', 'learning_rate': 0.001, 'decay': 0.0, 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-07, 'amsgrad': False}}}
```

对于模型参数，因为第一层`Flatten`和第三层`DropOut`只是对输入数据进行处理，没有计算，因此不包含参数。具体的参数如下：
```
dense/kernel:0
(784, 512)
dense/bias:0
(512,)
dense_1/kernel:0
(512, 10)
dense_1/bias:0
(10,)
```
带有dense是第二层参数，带有dense_1是第四层参数，列出的是参数名和参数形状，说明具体的参数值都是多维数组。如下所示
![](/home/jsxnh/图片/weight.png)

如果保存在HDF5中，会对图结构，参数等创建不同的group，然后把模型结构，参数这些数据保存到不同的group中。
如果保存为tensorflow的格式文件，checkpoint会有三个文件，一个是模型相关的文件，另外两个分别是参数索引文件和参数值文件。`ps`:对于posix文件系统，写实现用到了`<stdio.h>`中的`fwrite`函数

###### 然后是更新参数的过程

从图中可以发现，在输入到最后一层网络层输出后，利用优化算法进行更新网络层参数的操作。从输入数据开始，网络每一层的输出的数据都会用来更新相关网络层中的参数。例如图中的Relu layer中的参数更新与这一层的输出有关。
![](/home/jsxnh/图片/train.gif)



#### 例子2：较大数据集的读入和训练

利用较大数据集研究读入数据和训练数据的时间过程和时间分布

##### 实验环境
处理器：Intel® Core™ i7-8700 CPU @ 3.20GHz × 12
内存：31.2G
操作系统：ubuntu 16.04 LTS 64位
数据集：85x550MBtfrecord文件（每个文件由1000万条数据组成,因此一共有`850000000`条数据）

##### 数据流
我们可以简单的把模型的训练分为准备数据和训练数据，下图是准备数据的一般过程，上面的那张图是训练数据的过程。

![](/home/jsxnh/图片/dataflow.png)

从图上可以看到，准备数据包括从文件中读取数据(read),解析数据(map),打乱数据顺序(shuffle)，提取数据训练(batch)这些过程。图中有Read Buffer和Shuffle Buffer，这两个Buffer的大小可以提前设置。Read Buffer的大小是每次读入原始数据的大小，Shuffle Buffer的大小是经过map之后的数据乱序的空间的大小。如果Read Buffer的容量小于Shuffle Buffer的容量，那么需要多次读取填满Read Buffer然后填充Shuffle Buffer。在读取文件时，posix文件系统的读实现是用到`<unistd.h>`的`pread`函数。

为了研究模型训练时长的影响因素和IO对模型训练的影响。我们分别设置了不同的Read Buffer，Shuffle Buffer和Batch size作了实验。
对于数据集，有`iteration×batch_size=n`,其中n代表有多少条数据，iteration代表每一步训练，batch_size为每一步训练的数据量，因此对数据集一次训练有iteration步。

##### Batch_size
在这个实验中，设置read uffer的大小为默认(262144),shuffle buffer的大小为0(对数据没有进行操作)。我们使用了Tensorflow的python client。在下面的伪代码中，简化了整个代码的执行。
```
准备工作
for i in len(epoches):
   for j in len(iterations)：
        ...
        out = f(input)//python
        ...
收尾工作

f(input)：
准备
调用c
c语言中图的执行://如果数据集太大，那么图的执行包括数据的准备和训练数据
  if 内存中有数据:
    训练数据
  else:
    准备数据
    训练数据
返回

```
在实验中，记录了三个指标：程序运行的总时间，每个iteration中图的执行时间，如果有准备数据的过程，那么在图的执行过程中再记录下读取原始数据的时间(即在上图中Fill Up Buffer的时间)。下面的表是实验结果。

| batch_size | total_time(程序运行总时间) | total_exec_time(每次图执行时间的和) | avg_exec_time(图执行的平均时间) | total_read_time(读取数据的时间和) | avg_read_time(读取平均时间) | read_times:train_times(读的次数和训练次数的比) |
|-------|--------|--------|--------|--------|--------|--------|
|32|57192741381(微秒),15.887(h)|37345263018(微秒),10.373(h)|1406(微秒)|458273310(微妙),458(s)|2567(微秒)|1:149|
|500|13952060894(微秒),3.876(h)|12357054386(微秒),3.432(h)|7268(微秒)|114737080(微秒),114(s)|642(微秒)|1:10|
|5000|12582498790(微秒),3.495(h)|12396713462(微秒),3.443(h)|72921(微秒)|154803350(微秒),154(s)|867(微秒)|1:1|

- 由于batch_size变化，读次数和训练次数比在变化
- 当batch_size=5000,与batch_size=500时，比为10:1，因此图平均时间比约为10:1。
- 当batch_size=32时，总运行时间约是其他两个的4倍，(原因未知，猜测batch_size与图平均执行时间并未有线性关系，图平均执行操作中有固定的时间部分)

##### Shuffle Buffer
在程序中，shuffle buffer的大小可以设置，这个大小是数据记录的条数。同时使batch_size为500，read buffer的大小为默认。在这次实验中，分别设置shuffle size为0,500,50000。下面是实验结果。

| shuffle_size | total_time(程序运行总时间) | total_exec_time(每次图执行时间的和) | avg_exec_time(图执行的平均时间) | total_read_time(读取数据的时间和) | avg_read_time(读取平均时间) | read_times:train_times(读的次数和训练次数的比) |
|-------|--------|--------|--------|--------|--------|--------|
|0|14648376539(微秒),4.069(h)|13144630268(微秒),3.651(h)|7732(微秒)|113464921(微妙),113(s)|636(微秒)|1:10|
|500|26305746404(微秒),7.307(h)|24865115767(微秒),6.907(h)|14626(微秒)|120628173(微秒),120(s)|676(微秒)|1:10|
|50000|28220387981(微秒),7.839(h)|26674553064(微秒),7.409(h)|15690(微秒)|118786872(微秒),118(s)|665(微秒)|1:10|


- 当shuffle_size等于0的时候，说明没有shuffle这个过程
- 当shuffle_size等于500，这时shuffle的大小等于batch的大小。图执行的平均时间14626和没有shuffle的图执行的时间7732比，扩大了一倍。可以反应出有了shuffle后，shuffle的占比还是相当多的。
- read_times:train_times可以理解为batch的大小比read buffer的大小，例如这里1:10表示一次填满read buffer 可以用于10个batch的训练
- 当shuffle_size等于50000，这时候shuffle buffer的大小约是read buffer的10倍。那么程序开始时，首先要填满10次read buffer以此填满shuffle buffer，在程序运行过程中，shuffle buffer中用掉read buffer的大小，就会发生fill up read buffer事件。此时的图平均时间比当shuffle_size=500时稍大，可能是因为此时的shuffle_size比较大的原因。
- 因为read buffer相同,平均读取时间差不多

##### Read Buffer
为了测试读文件的次数的影响，这里使用了另一个数据集：1000000*550B文件(一个文件有10条记录，因此一共有10000000条记录)。
设置batch_size=500,shuffle_size=50000,分别设置read buffer=275B，550B，default(262144B),记录和以上相同的指标如下表。


| read buffer | total_time(程序运行总时间) | total_exec_time(每次图执行时间的和) | avg_exec_time(图执行的平均时间) | total_read_time(读取数据的时间和) | avg_read_time(读取平均时间) | read_times:train_times(读的次数和训练次数的比) |
|-------|--------|--------|--------|--------|--------|--------|
|275B|574902897(微秒),574(s)|542755237(微秒),542(s)|27137(微秒)|138540511(微妙),138(s)|46(微秒)|150:1|
|550B|572959504(微秒),572(s)|540906547(微秒),540(s)|27045(微秒)|137948400(微秒),137(s)|69(微秒)|100:1|
|262144B|570431145(微秒),570(s)|538802653(微秒),538(s)|26940(微秒)|138270799(微秒),138(s)|69(微秒)|100:1|

- 当read buffer=275B，这个大小为文件大小的一半。batch_size=500,那么训练的次数为20000次。对于每个文件读，先读取目录项120微秒左右，然后读取文件一半1微秒左右，然后读取文件另一半1微秒左右,所以读的次数一共有3000000次，read_times:train_times=150:1。
- 在相同batch_size,shuffle_size的条件下，这个实验的图平均执行时间27000比上面的实验中的15690多了许多，可能是因为需要读取多个小文件导致的。
- 由于缓存的存在，对其中的实验再做一次时，发现运行时间比先前少100秒左右，发现是由于没有了读取目录项120微秒这个读操作，加上存在数据缓存。





















































































