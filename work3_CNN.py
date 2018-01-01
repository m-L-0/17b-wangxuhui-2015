
# coding: utf-8

# In[1]:



import tensorflow as tf
import numpy as np
#import glob
from PIL import Image
from tqdm import tqdm
import pandas as pd
import time
import random
import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')

import os,sys,string
import logging
import multiprocessing
import json
import cv2
from sklearn.model_selection import train_test_split

import keras
import keras.backend as K
from keras.datasets import mnist
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *
from keras import backend as K

# from keras.utils.visualize_util import plot
#from visual_callbacks import AccLossPlotter
#plotter = AccLossPlotter(graphs=['acc', 'loss'], save_graph=True, save_graph_path=sys.path[0])


# # 1.统计数据

# In[2]:

def _int64_feature(value):
    if not isinstance(value,list):
        value=[value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _byte_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# In[3]:

# data=pd.read_csv('../data/captcha/labels/labels.csv',header=None)
# data[0]='../'+data[0]
# image_path_data =data[0]
# image_labels_data = data[1]
# data.head()


# # In[4]:

# image_labels_data.max()


# # 统计数据集总数、统计不同位数验证码的数量与所占比例。

# # In[5]:

# print('总条数为：',len(image_labels_data),'条')
# num_dict=dict()
# print('\n验证码的数量:\n',data[1].map(lambda x:len(str(x))).value_counts())

# print('\n验证码数量所占比例：\n',data[1].map(lambda x:len(str(x))).value_counts()/len(image_labels_data))


# # 2.将数据集制作成TFRecord文件。

# 将数据集按照8:1:1比例划分为训练集、验证集与测试集；
# 
# 每个TFRecord文件不得超过5000个样本。

# In[6]:

def getMaxHW(captcha_list_img):
    height_max=1
    width_max=1
    for i in tqdm(captcha_list_img):
        im=np.asarray(Image.open(i))
        a=im.shape
        if height_max<a[0]:
            height_max=a[0]
        if width_max<a[1]:
            width_max=a[1]
    print('height_max:',height_max)
    print('width_max:',width_max)
    return height_max,width_max


# height,width=getMaxHW(image_path_data)


# In[7]:

# #识别字符集
# char_ocr='0123456789' #string.digits
# #char_ocr=string.ascii_uppercase
# #定义识别字符串的最大长度
# seq_len=4
# #识别结果集合个数 0-9
# label_count=len(char_ocr)+1
# batch_sum=2000


# In[8]:

def encode_to_tfrecords(image_path_data,image_labels_data,name,width=57,height=43):#从图片路径读取图片编码成tfrecord
    writer=tf.python_io.TFRecordWriter(name)
    numclass=len(image_path_data)
    num=0
    #print(img_names)###############################################
    for img_id in tqdm(range(numclass)):
        img_path=image_path_data[img_id]
        #print(img_path)############################################
        im=np.asarray(Image.open(img_path).convert('P'))#转换成灰度
        a=im.shape
        add_height_n=(height-a[0])//2
        add_height_s=(height-a[0])-add_height_n
        add_width_w=(width-a[1])//2
        add_width_e=(width-a[1])-add_width_w
        #print(add_height_n,add_height_s,add_width_w,add_width_e)
        img=np.pad(im,((add_height_n,add_height_s),(add_width_w,add_width_e)) , 'constant', constant_values=(0,))#255
        #print(img.shape)
        #img=img.resize((height,width))
        #print(img.shape)
        img_raw=img.tobytes()
        img_label=image_labels_data[img_id]
#         img_label=str(img_label)
#         labels_len=4#len(img_label)
#         labels=[0]*labels_count*labels_len#我用的是softmax，要和预测值的维度一致
        
#         if len(img_label)<4:
#             cur_seq_len = len(img_label)
#             for i in range(4 - cur_seq_len):
#                 img_label=img_label+'*' #

#         for i,j in enumerate(img_label):
#             if j=='*':
#                 j=10
#             labels[i*labels_count+int(j)]=1
            
        example=tf.train.Example(features=tf.train.Features(feature={#填充example
            'image_raw':_byte_feature(img_raw),
            'label':_int64_feature(img_label)}))#labels
        writer.write(example.SerializeToString())#把example加入到writer里，最后写到磁盘。
        num=num+1
        #print('num:',num)
    writer.close()
        


# In[9]:

# name=[]
# for n in range(1,len(image_labels_data)//batch_sum+1):#20个
#     name.append(r'../data/'+str(n)+'.tfrecords')
#     print(name[n-1])
#     i=(n-1)*batch_sum
#     j=i+batch_sum
#     print('i:',i,'j:',j)
#     #print(data.ix[i:j,0])
    
#     encode_to_tfrecords(data.ix[i:j,0].values,data.ix[i:j,1].values,name[n-1])


# In[10]:

def decode_from_tfrecord(filequeuelist,width=57,height=43):
    #fileNameQue = tf.train.string_input_producer([filequeuelist])
    reader=tf.TFRecordReader()#文件读取
    _,example=reader.read(filequeuelist)#fileNameQue
    features=tf.parse_single_example(example,features={'image_raw':#解码
            tf.FixedLenFeature([],tf.string),
            'label':tf.FixedLenFeature([],tf.int64)})#labels_count*4,1
    image=tf.decode_raw(features['image_raw'],tf.uint8)
    image.set_shape(height*width)#rows*cols
    image=tf.cast(image,tf.float32)*(1./255)#-0.5
    label=tf.cast(features['label'],tf.int32)
    return image,label

def generate_filenamequeue(filequeuelist):
    filename_queue=tf.train.string_input_producer(filequeuelist)#,num_epochs=5
    return filename_queue

def get_batch(filename,batch_size):
    filename_queue=generate_filenamequeue(filename)
    with tf.name_scope('get_batch'):
        [image,label]=decode_from_tfrecord(filename_queue,width=width,height=height)
        images,labels=tf.train.shuffle_batch([image,label],batch_size=batch_size,num_threads=len(filename),
                 capacity=100+3*batch_size,min_after_dequeue=batch_size)
        #images,labels=tf.train.shuffle_batch_join
        return images,labels


# In[11]:

# test_name=random.choices(name,k=2)
# [name.remove(i) for i in test_name]

# var_name=random.choices(name,k=2)
# [name.remove(i) for i in var_name]


# # 3.设计模型。
# 

#     只能使用卷积神经网络；
#     只能设计一个模型来完成不定长度验证码的识别（端到端模型）；
#     输出模型参数数量，参数数量不得多于300万。
#     使用一个或多个name_scope（或variable_scope）包裹模型或模型中的相关操作部分。
#     将模型的结果特点与设计思想写入文档。

# ### 定义网络结构

# In[14]:

def gen_data(name,batch_size=128):
    batch_size=batch_sum
    print(len(name))
    init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
    with tf.Session() as sess:
        for t,name_dir in enumerate(name):
            print(t,name_dir)
            if t==0:
                images,labels=get_batch([name_dir],batch_size)##train=name
                sess.run(init_op)
                coord=tf.train.Coordinator()
                threads=tf.train.start_queue_runners(sess=sess,coord=coord)
                image,label=sess.run([images,labels])
            #     print('image:',image.shape)#[batch_size,2451]
            #     print('label:',label.shape)##[batch_size,40,1]
                X=image.reshape(batch_size,height,width,1)#(5624, 150, 50, 1)
                #image=image.transpose(0,2,1,3)
                #X = np.zeros((batch_size, width, height, 3), dtype=np.uint8)
                #y = [np.zeros((batch_size, labels_count), dtype=np.uint8) for i in range(4)]#初始化训练集Y
                Y = np.zeros((batch_size, seq_len), dtype=np.uint8)#seq_len
                Y=Y+11
                for i in range(batch_size):
                    for j, ch in enumerate(str(label[i])):
                        #print('ch:',ch,type(ch))
                        Y[i,j]=ch
            else:
                images,labels=get_batch([name_dir],batch_sum)##train=name 
                sess.run(init_op)
                #coord=tf.train.Coordinator()
                threads=tf.train.start_queue_runners(sess=sess,coord=coord)
                image,label=sess.run([images,labels])
            #     print('image:',image.shape)#[batch_size,2451]
            #     print('label:',label.shape)##[batch_size,40,1]
                image=image.reshape(batch_size,height,width,1)#
                #image=image.transpose(0,2,1,3)
                #X = np.zeros((batch_size, width, height, 3), dtype=np.uint8)
                #y = [np.zeros((batch_size, labels_count), dtype=np.uint8) for i in range(4)]#初始化训练集Y
                y = np.zeros((batch_size, seq_len), dtype=np.uint8)#seq_len
                y=y+11
                for i in range(batch_size):
                    for j, ch in enumerate(str(label[i])):
                        #print('ch:',ch,type(ch))
                        y[i,j]=ch                
                X=np.row_stack((X, image))
                Y=np.row_stack((Y, y))
        coord.request_stop()  
        time.sleep(1)
    coord.join(threads)
    return X,Y


# In[15]:

# X,Y=gen_data(name)
# var_X,var_Y=gen_data(var_name)


# In[16]:

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accs = []

    def on_epoch_end(self, epoch, logs=None):
        model.save_weights('model_1018.w')
        base_model.save_weights('base_model_1018.w')
        

        acc = test(base_model,var_X,var_Y)*100#evaluate(model)*100
        self.accs.append(acc)
        tf.summary.scalar('acc', acc)
        tf.summary.histogram('acc',acc)
        print( '>>acc: %f%%'%acc)

    def on_batch_end(self, batch, logs={}):
        print('logs_loss:',logs.get('loss'))
        if float(logs.get('loss'))<np.inf:
            tf.summary.scalar('loss', logs.get('loss'))
            tf.summary.histogram('acc',logs.get('loss'))


# In[17]:

# the actual loss calc occurs here despite it not being
# an internal Keras loss function

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    # y_pred = y_pred[:, 2:, :] 测试感觉没影响
    y_pred = y_pred[:, :, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def test(base_model,var_X,var_Y):
    file_list = []
    #test_X,test_Y=gen_data([test])#gen_image_data(dir=r'./test')#gen_image_data(r'data\test', file_list)
    y_pred = base_model.predict(var_X)
    shape = y_pred[:, :, :].shape  # 2:
    out = K.get_value(K.ctc_decode(y_pred[:, :, :], input_length=np.ones(shape[0]) * shape[1])[0][0])[:,
          :seq_len]  # 2:
    #print('out:',out)
    error_count=0
    batch_num=len(var_X)
    for i in range(batch_num):
        #print(file_list[i])
        #str_src = ''.join(char_ocr[i] for i in Y)
        str_src = ''.join([str(x) for x in var_Y[i] if x!=11])
        #print('out[i]:',out[i])
        str_out = ''.join([str(x) for x in out[i] if x!=-1 ])
        #print('str_src, str_out:',str_src, str_out)
        if str_src!=str_out:
            error_count+=1
    print('#########error_count#',error_count,'###acc:',(batch_num-error_count)/batch_num)
        # img = cv2.imread(file_list[i])
        # cv2.imshow('image', img)
        # cv2.waitKey()
    return (batch_num-error_count)/batch_num


# In[18]:
def create_CNN():
    input_tensor = Input((height, width, 1))
    x = input_tensor
    for i in range(3):
        x = Convolution2D(32*2**i, (3, 3), activation='relu', padding='same')(x)
        x = Convolution2D(32*2**i, (3, 3), activation='relu',padding='same')(x)
      

        x = MaxPooling2D(pool_size=(2, 2))(x)

    conv_shape = x.get_shape()
    # print(conv_shape)
    x = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2] * conv_shape[3])))(x)

    x = Dense(32, activation='relu')(x)

    gru_1 = GRU(32, return_sequences=True, kernel_initializer='he_normal', name='gru1')(x)
    gru_1b = GRU(32, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(x)
    gru1_merged = add([gru_1, gru_1b])  ###################

    gru_2 = GRU(32, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
    gru_2b = GRU(32, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(
        gru1_merged)
    x = concatenate([gru_2, gru_2b])  ######################
    x = Dropout(0.25)(x)
    x = Dense(label_count, kernel_initializer='he_normal', activation='softmax')(x)
    base_model = Model(inputs=input_tensor, outputs=x)

    labels = Input(name='the_labels', shape=[seq_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([x, labels, input_length, label_length])

    model = Model(inputs=[input_tensor, labels, input_length, label_length], outputs=[loss_out])
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adadelta')
    model.summary()
    return base_model,model,conv_shape
# y_pred 是模型的输出，是按顺序输出的10个字符的概率
# labels 是验证码，是四个数字；
# input_length 表示 y_pred 的长度，我们这里是15；
# label_length 表示 labels 的长度，我们这里是4。


# ### 算法过程描述可视化

# In[19]:

# from keras.utils.vis_utils import plot_model
# from IPython.display import Image as Image1
# import pydot,graphviz
# plot_model(model, to_file="model.png", show_shapes=True)
# Image1('model.png',width=650, height=550)

# #############################################
# Image1('model.png',width=650, height=550)
# ############################################


# # 计算图可视化

# In[20]:



#写入日志
#tensorboard --logdir=r'./path_to_log/'
# ### 训练模型 并计算当前模型的识别正确率，ACC

# In[21]:

if __name__ == '__main__':

    data=pd.read_csv('../data/captcha/labels/labels.csv',header=None)
    data[0]='../'+data[0]
    image_path_data =data[0]
    image_labels_data = data[1]
    data.head()


    # In[4]:

    image_labels_data.max()


    # 统计数据集总数、统计不同位数验证码的数量与所占比例。

    # In[5]:

    print('总条数为：',len(image_labels_data),'条')
    num_dict=dict()
    print('\n验证码的数量:\n',data[1].map(lambda x:len(str(x))).value_counts())

    print('\n验证码数量所占比例：\n',data[1].map(lambda x:len(str(x))).value_counts()/len(image_labels_data))


    height,width=getMaxHW(image_path_data)
    #识别字符集
    char_ocr='0123456789' #string.digits
    #char_ocr=string.ascii_uppercase
    #定义识别字符串的最大长度
    seq_len=4
    #识别结果集合个数 0-9
    label_count=len(char_ocr)+1
    batch_sum=2000



    name=[]
    for n in range(1,len(image_labels_data)//batch_sum+1):#20个
        name.append(r'../data/'+str(n)+'.tfrecords')
        print(name[n-1])
        i=(n-1)*batch_sum
        j=i+batch_sum
        print('i:',i,'j:',j)
        #print(data.ix[i:j,0])
        #encode_to_tfrecords(data.ix[i:j,0].values,data.ix[i:j,1].values,name[n-1],width=width,height=height)#-------------

    test_name=random.choices(name,k=2)
    [name.remove(i) for i in test_name]

    var_name=random.choices(name,k=2)
    [name.remove(i) for i in var_name]
    X,Y=gen_data(name)
    var_X,var_Y=gen_data(var_name)
    # checkpointer = ModelCheckpoint(filepath="keras_seq2seq_1018.hdf5", verbose=1, save_best_only=True, )

    base_model,model,conv_shape=create_CNN()
    history = LossHistory()
    writer = tf.summary.FileWriter("../path_to_log/")#,tf.

    #X,Y=gen_data(name)#gen_image_data(dir=r'./simple-level')#gen_image_data()
    #maxin=490
    subseq_size = 100
    #batch_size=10
    #var_X,var_Y=gen_data([var])#####
    result=model.fit([X, Y, np.array(np.ones(len(X))*int(conv_shape[1])), np.array(np.ones(len(X))*seq_len)], Y,
                     batch_size=1000,
                     epochs=50,
                     callbacks=[EarlyStopping(patience=22),history], #checkpointer,  plotter, 
                     validation_data=([var_X, var_Y, np.array(np.ones(len(var_X))*int(conv_shape[1])), np.array(np.ones(len(var_X))*seq_len)], var_Y),
                     )#patience=20,准确率75%，到25次的时候有77%，更高的迭代次数没有跑过
    

    # # 测试模型

    # In[22]:

    test(base_model,var_X,var_Y)


    # # 计算模型总体准确率

    # In[23]:

    test_X,test_Y=gen_data(test_name)
    test(base_model,test_X,test_Y)


    # In[24]:

    model.save('../data/model.h5')
    base_model.load_weights('../data/base_model_1018.w')
    model.load_weights('../data/model_1018.w')


    writer.close()
    K.clear_session()

