#常量constant   在运行过程中不会改变的单元，在TensorFlow中无需进行初始化操作

import tensorflow as tf

a = tf.constant(1.0,name = "a")
b = tf.constant(2.5,name= "b")
c = tf.add(a,b,name = "c")

sess = tf.Session()
c_value = sess.run(c)
print(c_value)
sess.close()



#变量  Variable
#在运行过程中会该改变的单元，在TensorFlow中须进行初始化操作

'''
个别变量初始化：
            init_op = name_variable.initializer()
所有变量初始化：
            init_op = tf.global_variable_initializer()
            '''
node1 = tf.Variable(3.0,tf.float32,name = "node1")
node2 = tf.Variable(4.0,tf.float32,name = "node2")
result = tf.add(node1,node2,name = "add")

sess = tf.Session()

#变量初始化
init = tf.global_variables_initializer()
#init只是一个操作，也需要先运行才能实现（即所有操作只有运行后才能实现）
sess.run(init)

print(sess.run(result))


#变量赋值

#特殊情况需要人工更新的可用变量赋值语句
#变量更新语句：
#        update_op = tf.assign(variable_to_be_updates,new_value)

#通过变量赋值输出1、2、 3 、、、10
import tensorflow as tf

value = tf.Variable(0,name='value')
one = tf.constant(1)
new_value = tf.add(value,one)
update_value = tf.assign(value,new_value)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(10):
        sess.run(update_value)
        print(sess.run(value))

'''
通过TensorFlow的变量赋值计算：1+2+3+...10？
'''



'''TensorFlow中的Variable变量类型  定义时需要初始化   但有些变量定义时  并不知道其数值  运行时 由外部输入
如训练数据   此时需要用到占位符

tf.placeholder 占位符
'''


#   tf.placeholder(dtype,shape = None,name = None)

x = tf.placeholder(tf.float32,[2,3],name = 'tx')
#此代码生成一个2*3的二维数组，矩阵中每个元素的类型都是tf.float32  




#构建一个包含placeholder操作的计算图，在sessiomn中调用run方法时，
#placeholder占用的变量比持续用feed_dict餐胡传递进去

import tensorflow as tf

a = tf.placeholder(tf.float32,name = 'a')
b = tf.placeholder(tf.float32,name ='b')
c = tf.multiply(a,b,name ='c')

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    #通过fees_dict的从参数传值，按字典格式
    result = sess.run(c,feed_dict = {a:8.0,b:3.5})
    
    print(result)



#多个操作可以通过一次Feed完成执行


import tensorflow as tf

a = tf.placeholder(tf.float32,name = 'a')
b = tf.placeholder(tf.float32,name ='b')
c = tf.multiply(a,b,name ='c')
d = tf.subtract(a,b,name = 'd')

init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    
    result = sess.run([c,d],feed_dict = {a:[8.0,2.0,3.5],b:[1.5,2.0,4.]})
    
    print(result)
    #取结果的第一个
    print(result[0])





#一次返回多个值分别赋给多个变量

import tensorflow as tf

a = tf.placeholder(tf.float32,name = 'a')
b = tf.placeholder(tf.float32,name ='b')
c = tf.multiply(a,b,name ='c')
d = tf.subtract(a,b,name = 'd')

init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    
    rc,rd = sess.run([c,d],feed_dict = {a:[8.0,2.0,3.5],b:[1.5,2.0,4.]})
    
    
    print("value of c = ",rc,"value of d = ",rd)
