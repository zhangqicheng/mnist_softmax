import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.examples.tutorials.mnist import input_data
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#控制训练过程中的参数
learning_rate=0.01
training_epochs=20
batch_size=256
display_step=1
examples_to_show=10

#网络模型参数
n_hidden_units=256  #隐藏层神经元数量
n_input_units=784   #输入层神经元数量
n_output_units=n_input_units    #输入与输出必须相同

#根据输入输出节点数量初始化指定好的权重
def WeightsVariable(n_in,n_out,name_str):
    return tf.Variable(tf.random_normal([n_in,n_out]),dtype=tf.float32,name=name_str)

#根据输入输出节点数量初始化指定好的偏置
def BiasesVariable(n_out,name_str):
    return tf.Variable(tf.random_normal([n_out]),dtype=tf.float32,name=name_str)

#构建编码器
def Encoder(x_origin,activate_func=tf.nn.sigmoid):
    with tf.name_scope('Layer'):
        weights=WeightsVariable(n_input_units,n_hidden_units,'weights')
        biases=BiasesVariable(n_hidden_units,'biases')
        x_code=activate_func(tf.add(tf.matmul(x_origin,weights),biases))
    return x_code

#构建解码器
def Decoder(x_code,activate_func=tf.nn.sigmoid):
    # 解码器第一隐藏层
    with tf.name_scope('Layer'):
        weights=WeightsVariable(n_hidden_units,n_output_units,'weights')
        biases=BiasesVariable(n_output_units,'biases')
        x_decode=activate_func(tf.add(tf.matmul(x_code,weights),biases))
    return x_decode

#调用函数构造构造计算图
with tf.Graph().as_default():
    #计算图输入
    with tf.name_scope('X_Origin'):
        X_Origin=tf.placeholder(tf.float32,[None,n_input_units])
    #构建编码器模型
    with tf.name_scope('Encoder'):
        X_code=Encoder(X_Origin,activate_func=tf.nn.sigmoid)
    #构建解码器模型
    with tf.name_scope('Decoder'):
        X_decode=Decoder(X_code,activate_func=tf.nn.sigmoid)

    #定义损失节点
    with tf.name_scope('Loss'):
        Loss=tf.reduce_mean(tf.pow(X_Origin-X_decode,2))

    #定义优化器，训练节点
    with tf.name_scope('Train'):
        Optimizer=tf.train.RMSPropOptimizer(learning_rate)
        Train=Optimizer.minimize(Loss)

    #为所有变量添加初始化节点
    Init=tf.global_variables_initializer()

    print('把计算图写入时间文件中，并在tensorboard中查看')
    writer = tf.summary.FileWriter(logdir='logs', graph=tf.get_default_graph())
    writer.flush()

    #导入mnist data
    mnist=input_data.read_data_sets('../mnist_data/',one_hot=True)

    #产生会话，启动计算图
    with tf.Session() as sess:

        sess.run(Init)
        total_batch=int(mnist.train.num_examples/batch_size)#批次
        for epoch in range(training_epochs):#训练20轮
            for i in range(total_batch):
                batch_xs,batch_ys=mnist.train.next_batch(batch_size)

                _,loss=sess.run([Train,Loss],feed_dict={X_Origin:batch_xs})

            if epoch%display_step==0:
                print('Epoch','%04d'%(epoch+1),'loss=','{:.9f}',format(loss))
        writer.close()
        print('模型训练完毕')

        #模型用在测试集，输出重建后的样本
        reconstructions=sess.run(X_decode,feed_dict={X_Origin:mnist.test.images[:examples_to_show]})
        #比较原始数据和重建后的图像
        f,a=plt.subplots(2,10,figsize=(10,2))
        for i in range(examples_to_show):
            a[0][i].imshow(np.reshape(mnist.test.images[i],(28,28)))
            a[1][i].imshow(np.reshape(reconstructions[i],(28,28)))
        f.show()
        plt.draw()
        plt.waitforbuttonpress()

