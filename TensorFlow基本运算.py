import tensorflow as tf

#创建一个常值运算，将作为一个节点加入到默认计算图中
hello = tf.constant("hello world")

#创建一个TF对话
sess = tf.Session()

#运行并获得结果
print(sess.run(hello))


'''
计算图   每一个计算都是通过计算图的形式表示计算的变成系统
        每一个计算都有事计算图上的一个节点
        节点之间的变描述计算之间的关系
        
        
        
        计算图是一个有限图：
        一组节点，每个节点都代表一个曹祖，是一种运算
        一组邮箱变，每条变代表节点之间的关系（数据传递和控制依赖）
        
        Tensorflow有两种边：
        实线边：代表数据依赖关系，一个节点的运算输出成为另一个节点的输入，两个节点之间有tensor流动（值传递）
        虚线边：不携带值，表示两个节点之间的控制相关性，源节点必须在目的几点执行之前完成执行
        
        
       
        
    计算图中的接待就是操作（加法乘法、！！构建一些变量的初始值也是一个操作！！）
    
    如果操作A的输入是曹祖B执行的结果，那么这个操作A就依赖操作B
        '''


#计算图实例
node1 = tf.constant(3.0,tf.float32,name = "node1")
node2 = tf.constant(4.0,tf.float32,name = "node2")
node3 = tf.add(node1,node2)

print(node3)
'''
输出结果不是一个具体的数字，而是一个张量的结构

因为

创建计算图指数建立静态计算模型
执行对话才能提供数据病获得结果

'''

#建立对话并显示运行结果
sess = tf.Session()

print("运行sess.run(node1)的结果：",sess.run(node1))#运行sess.run(node1)的结果

#更新变量并返回计算结果 
print("运行sess.tun(node3)的结果：",sess.run(node3))#运行sess.run(node3)

#关闭Session
sess.close()




#张量的属性

#Tensor("Add(节点名称）:0（来自节点的第几个输出）",shape=()（维度信息）,dtype=float32)

import tensorflow as tf

tf.reset_default_graph()#清楚defaule graph和不断增加的节点

#定义变量a
a = tf.Variable(1,name="a")
#定义操作b为a+1
b = tf.add(a,1,name="b")
#定义操作c为b+4
c = tf.multiply(b,4,name="c")
#定义d为c-d
d = tf.subtract(c,b,name = "d")


'''
会话  拥有并管理TensorFlow程序运行是所有资源
当所有计算完成之后需要关闭会话帮助系统回收资源'''


#会话模式1

#定义计算图
tens1 = tf.constant([1,2,3])

#创建一个会话
sess = tf.Session()

'''

#使用跟这个创建好的会话来得到关心的运算的结果，
#来得到张量result的取值
print(sess.run(tens1))

#关闭会话使得本次运行中使用管道的资源可以被释放
sess.close()#关闭会话并释放资源
'''
try:
    print(sess.run(tens1))
except:
    print("Exception!")
finally:
    #确保能关闭会话使得。。。
    sess.close()#无论如何程序最后都会执行关闭操作

    
    
    #会话模式2
node1 = tf.constant(3.0,tf.float32,name = "node1")
node2 = tf.constant(4.0,tf.float32,name = "node2")
result = tf.add(node1,node2)

#创建会话，并通过python中的上下文管理器来管理这个会话
with tf.Session() as sess:
    #使用跟着创建好的会话来计算关心的结果
    print(sess.run(result))
    
#不需要在调用Session.close()函数来关闭会话
#当上下文退出是会话关闭和资源释放也自动完成了



制定默认的会话

TensorFlow不会自动给生成默认的会话，需要手动指定

当默认的会话被指定之后可以通过 tf.Tensor.eval函数来计算一个张量的取值

node1 = tf.constant(3.0,tf.float32,name = "node1")
node2 = tf.constant(4.0,tf.float32,name = "node2")
result = tf.add(node1,node2)

sess = tf.Session()
#
with sess.as_default():
    print(result.eval())
    
#
print(sess.run(result))
print(result.eval(session=sess))

交互式环境下，Jupyter下，通事故设置默认会话来获取张量的取值更加方便

tf.InteractiveSession使用这个函数会自动将生成的会花注册为默认会话

node1 = tf.constant(3.0,tf.float32,name = "node1")
node2 = tf.constant(4.0,tf.float32,name = "node2")
result = tf.add(node1,node2)

sess = tf.InteractiveSession()

print(result.eval())
sess.close()
