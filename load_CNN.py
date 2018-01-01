import tensorflow as tf
import numpy as np
#import glob
from PIL import Image
import time

import os,sys,string
import logging
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
height,width=43,57
#识别字符集
char_ocr='0123456789' #string.digits
#char_ocr=string.ascii_uppercase
#定义识别字符串的最大长度
seq_len=4
#识别结果集合个数 0-9
label_count=len(char_ocr)+1
batch_sum=2000
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    # y_pred = y_pred[:, 2:, :] 测试感觉没影响
    y_pred = y_pred[:, :, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

# def load_CNN(model_file='../data/model.h5'):
# 	input_tensor = Input((height, width, 1))
# 	x = input_tensor
# 	for i in range(3):
# 	    x = Convolution2D(32*2**i, (3, 3), activation='relu', padding='same')(x)
# 	    x = Convolution2D(32*2**i, (3, 3), activation='relu',padding='same')(x)
	  

# 	    x = MaxPooling2D(pool_size=(2, 2))(x)

# 	conv_shape = x.get_shape()
# 	# print(conv_shape)
# 	x = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2] * conv_shape[3])))(x)

# 	x = Dense(32, activation='relu')(x)

# 	gru_1 = GRU(32, return_sequences=True, kernel_initializer='he_normal', name='gru1')(x)
# 	gru_1b = GRU(32, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(x)
# 	gru1_merged = add([gru_1, gru_1b])  ###################

# 	gru_2 = GRU(32, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
# 	gru_2b = GRU(32, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(
# 	    gru1_merged)
# 	x = concatenate([gru_2, gru_2b])  ######################
# 	x = Dropout(0.25)(x)
# 	x = Dense(label_count, kernel_initializer='he_normal', activation='softmax')(x)
# 	base_model = Model(inputs=input_tensor, outputs=x)

# 	labels = Input(name='the_labels', shape=[seq_len], dtype='float32')
# 	input_length = Input(name='input_length', shape=[1], dtype='int64')
# 	label_length = Input(name='label_length', shape=[1], dtype='int64')
# 	loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([x, labels, input_length, label_length])

# 	model = Model(inputs=[input_tensor, labels, input_length, label_length], outputs=[loss_out])
# 	model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adadelta')
# 	model.summary()

# 	model.load_weights(model_file)######model.save('cnn.h5')
# 	return base_model



# def get_picture(base_model,file_name):
#     im=np.asarray(Image.open(r"../data/captcha/images/"+file_name.split('/')[-1]).convert('P'))#转换成灰度
#     a=im.shape
#     add_height_n=(height-a[0])//2
#     add_height_s=(height-a[0])-add_height_n
#     add_width_w=(width-a[1])//2
#     add_width_e=(width-a[1])-add_width_w
#     #print(add_height_n,add_height_s,add_width_w,add_width_e)
#     img=np.pad(im,((add_height_n,add_height_s),(add_width_w,add_width_e)) , 'constant', constant_values=(0,))#255
#     var_X=img.reshape(1,height,width,1)#(batch_size,height,width,1)
#     file_list = []
#     #test_X,test_Y=gen_data([test])#gen_image_data(dir=r'./test')#gen_image_data(r'data\test', file_list)
#     y_pred = base_model.predict(var_X)
#     shape = y_pred[:, :, :].shape 
#     out = K.get_value(K.ctc_decode(y_pred[:, :, :], input_length=np.ones(shape[0]) * shape[1])[0][0])[:,:seq_len]  # 2:
#     batch_num=len(var_X)
#     K.clear_session()
#     for i in range(batch_num):
#         str_out = ''.join([str(x) for x in out[i] if x!=-1 ])
#     return str_out

def load_CNN2(file_name,model_file='../data/model.h5'):
	pre=r''#r'../data/captcha/images/'
	im=np.asarray(Image.open(pre+file_name).convert('P'))#转换成灰度.split('/')[-1]
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
	#model.summary()
	
	model.load_weights(model_file)######model.save('cnn.h5')
	a=im.shape
	add_height_n=(height-a[0])//2
	add_height_s=(height-a[0])-add_height_n
	add_width_w=(width-a[1])//2
	add_width_e=(width-a[1])-add_width_w
	img=np.pad(im,((add_height_n,add_height_s),(add_width_w,add_width_e)) , 'constant', constant_values=(0,))#255
	var_X=img.reshape(1,height,width,1)#(batch_size,height,width,1)
	file_list = []
	y_pred = base_model.predict(var_X)
	#test_X,test_Y=gen_data([test])#gen_image_data(dir=r'./test')#gen_image_data(r'data\test', file_list)
	shape = y_pred[:, :, :].shape 
	out = K.get_value(K.ctc_decode(y_pred[:, :, :], input_length=np.ones(shape[0]) * shape[1])[0][0])[:,:seq_len]  # 2:
	batch_num=len(var_X)
	K.clear_session()
	for i in range(batch_num):
		str_out = ''.join([str(x) for x in out[i] if x!=-1 ])
	return str_out



# if __name__ == '__main__':
# # 	base_model = load_CNN(model_file='model.h5')
# # 	file="<FileStorage: '0.jpg' ('image/jpeg')>"
# # 	img=get_picture(base_model,file.split(r": '")[1].split("' (")[0])
# # 	print('img:................',img)#img
# # 	print('load_model')
# 	file_name='0.jpg'
# 	print(load_CNN2(file_name,model_file='../data/model.h5'))