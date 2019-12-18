import glob
import os
import cv2
os.environ['TRIDENT_BACKEND'] = 'pytorch'
import trident as T
from trident import *
from trident.layers.pytorch_activations import __all__

import mpl_toolkits.mplot3d
import torch
import torch.optim as optim
_session=get_session()
_backend=_session.backend










dataset=T.load_examples_data('pokemon')
dataset.image_transform_funcs=[resize((128,128)),
                               normalize(127.5,127.5)]

#
# net=Sequential(
#     Dense(64,use_bias=False,activation='relu'),
#     Dense(10,use_bias=False,activation='softmax'))
# net.shape_infer([28*28])
# summary(net,(28*28))
#
# plan=TrainingPlan()\
#     .add_training_item(
#     TrainingItem(model=net,optimizer='RAdam',lr=2e-3,betas=(0.9, 0.999),weight_decay=1e-3).with_loss(CrossEntropyLoss).with_metrics(accuracy).with_regularizers('l2'))\
#     .with_data_loader(dataset)\
#     .within_minibatch_size(16).only_steps(num_steps=2,keep_weights_history=True,keep_gradient_history=True )
#
#
# aa=__all__
#



encoder=Sequential(
    Conv2d_Block((5,5),32,strides=2,auto_pad=True,activation='leaky_relu',normalization='batch',use_bias=False, add_noise=True,noise_intensity=0.02),#(64,128,128)
    Conv2d_Block((3,3),64,strides=2,auto_pad=True,activation='leaky_relu',normalization='batch',use_bias=False),#(64,64,64)
    Conv2d_Block((3,3),64,strides=2,auto_pad=True,activation='leaky_relu',normalization='batch',use_bias=False),#(128,32,32)
    Conv2d_Block((3,3),128,strides=2,auto_pad=True,activation='leaky_relu',normalization='batch',use_bias=False,dropout_rate=0.5),#(128,16,16)
    Conv2d_Block((3,3),128,strides=2,auto_pad=True,activation='leaky_relu',normalization='batch',use_bias=False),#(256,8,8)
    Conv2d_Block((3,3),256,strides=2,auto_pad=True,activation='leaky_relu',normalization='batch',use_bias=False),#(256,4,4)
    Conv2d_Block((3,3),256,strides=1,auto_pad=True,activation='leaky_relu',normalization='batch',use_bias=False),#(512,4,4)
    Reshape((-1,1,1)), #(512*4*4)
    Conv2d((1,1),128,strides=1,auto_pad=True,activation='tanh',use_bias=False)
)


decoder=Sequential(
    Conv2d((1,1),128*4*4,strides=1,auto_pad=True,activation='tanh',use_bias=False), #(1024,1,1 )
    Reshape((128,4,4)), #(64,4,4))
    Conv2d_Block((3,3),128,strides=1,auto_pad=True,activation='leaky_relu',normalization='batch',use_bias=False) ,#((64,4,4))
    TransConv2d_Block((3,3),64,strides=2,auto_pad=True,activation='leaky_relu',normalization='batch',use_bias=False),#(64,8,8)
    TransConv2d_Block((3,3),64,strides=2,auto_pad=True,activation='leaky_relu',normalization='batch',use_bias=False),#(64,16,16)
    TransConv2d_Block((3,3),64,strides=2,auto_pad=True,activation='leaky_relu',normalization='batch',use_bias=False),#(64,32,32)
    TransConv2d_Block((3,3),64,strides=2,auto_pad=True,activation='leaky_relu',normalization='batch',use_bias=False),#(64,64,64)
    TransConv2d_Block((3,3),64,strides=2,auto_pad=True,activation='leaky_relu',normalization='batch',use_bias=False),#(64,128,128)
    Conv2d((1,1),3,strides=1,auto_pad=True,activation='tanh',use_bias=False)
)

autoencoder=Sequential(
    encoder,
    decoder
)
autoencoder.input_shape=[3,128,128]

#summary(autoencoder,(3,128,128))


#
# dict={'betas':(0.9, 0.999),'weight_decay':1e-5}
# ta=tRAdam(autoencoder.parameters(),0.01,**dict)
#
# ta.base_lr=1e-5
# print(ta.base_lr)
# print(ta)

import inspect




plan=TrainingPlan()\
    .add_training_item(
    TrainingItem(model=autoencoder,optimizer=Ranger,lr=2e-3,betas=(0.9, 0.999),weight_decay=1e-5).with_loss(MSELoss).with_metrics(rmse).with_regularizers('l2'))\
    .with_data_loader(dataset)\
    .repeat_epochs(250)\
    .within_minibatch_size(16)\
    .with_learning_rate_schedule(reduce_lr_on_plateau,mode='min',factor=0.5,patience=5,threshold=1e-4,warmup=5)\
    .print_progress_scheduling(20,unit='batch')\
    .save_model_scheduling(5,'Models',unit='epoch')\
    .display_tile_image_scheduling(1,'epoch','Results/','pokemon_ae_{0}.png',True,True,False,False,True)\
    .start_now()