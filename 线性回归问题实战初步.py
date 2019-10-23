#在jupyter中，使用matplotlib显示图像需要设置为inline模式，否则不会显示图像
%matplotlib inline

import matplotlib.pyplot as plt#载入matplotlib
import numpy as np
import tensorflow as tf


#
#设置随机数种子
np.random.seed(5)

#直接采用np生成等差数列的方法，生成100个点，每个点的取值在-1~1之间
x_data = np.linspace(-1,1,100)

#y = 2x +1 +噪声，其中，噪声的维度与x_data一致

y_data  = 2 * x_data +1.0 +np.random.randn(*x_data.shape)*0.4



#
#画出随机生成数据的散点图
plt.scatter(x_data,y_data)

#画出我们想要学习到的线性函数 y = 2x+1
plt.plot(x_data,2*x_data+1.0,color = 'red',linewidth = 3)




#
#定义训练数据的占位符，x是特征值，y是标签值
x = tf.placeholder("float",name = 'x')
y = tf.placeholder("float",name ='y')

#定义模型函数
def model(x,w,b):
    return tf.multiply(x,w)+b



#
定义模型结构
TensorFlow变量的声明函数是tf.Variable
tf.Varoable的作用格式保存和更新参数
变量的初始值可以使随机数、常数，或是通过其他变量的初始值计算得到

# 构建线性函数的斜率，变量w
w = tf.Variable(1.0,name ="w0")

#构建线性函数的截距，变量b
b = tf.Variable(0.0,name ="b0")

#pred是预测值，前向计算
pred = model(x,w,b)



#
设置训练参数

#迭代次数（训练轮数）
train_epochs = 10

#学习率
learning_rate = 0.05




#
定义损失函数
损失函数用于描述预测值与真实值之间的误差，从而指导模型收敛方向
常见损失函数：均方差和交叉熵

#采用均方差作为损失函数
loss_function = tf.reduce_mean(tf.square(y-pred))



#
定义优化器
定义优化器Optimizer，初始化一个GradientDescentOptimizer
设置学习率和优化目标：最小化损失¶

# 梯度下降优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)

#声明会话
sess = tf.Session()




#
变量初始化
在真正执行计算之前，需要将所有变量初始化
通过tf.global_variables_initializer 函数可实现对所有变量的初始化


init = tf.global_variables_initializer()

sess.run(init)




#
迭代训练
模型训练阶段，设置迭代轮次，每次通过讲样本逐个输入模型，进行梯度下降优化操作，每轮迭代后，回执出模型曲线¶



#开始训练，轮次为 epoch，采用SGD随机梯度下降优化方法
for epoch in range(train_epochs):
    for xs,ys in zip(x_data,y_data):
        _,loss = sess.run([optimizer,loss_function],feed_dict = {x:xs,y :ys})
    
    b0temp = b.eval(session = sess)
    w0temp = w.eval(session = sess)
    plt.plot (x_data,w0temp*x_data+b0temp)#画图
    
   


#
    结果查看
当训练完成后，打印查看参数


print("w: ",sess.run(w))#w的值应该在2附近
print("b: ",sess.run(b))#b的值应该在1附近

#每次数据运行都可能会有所不同




#
结果可视化
plt.scatter(x_data,y_data,label  = "Original data")
plt.plot(x_data,x_data*sess.run(w)+sess.run(b),\
        label = "Fitted line",color = 'r',linewidth = 3)
plt.legend(loc = 2)#通过参数loc制定图例位置




#
利用模型 进行预测
x_test = 3.21
​
predict = sess.run(pred,feed_dict={x:x_test})
print("预测值：%f"%predict)
​
target = 2* x_test +1.0
print("目标值： %f"%target)
