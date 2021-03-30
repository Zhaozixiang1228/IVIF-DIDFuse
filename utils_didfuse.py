# -*- coding: utf-8 -*-
"""
@author: Zixiang Zhao (zixiangzhao@stu.xjtu.edu.cn)

Pytorch implement for "DIDFuse: Deep Image Decomposition for Infrared and Visible Image Fusion" (IJCAI 2020)

https://www.ijcai.org/Proceedings/2020/135
"""
import numpy as np
import torch
from DIDFuse import AE_Encoder,AE_Decoder
import torch.nn.functional as F

device='cuda'

def output_img(x):
    return x.cpu().detach().numpy()[0,0,:,:]

def l1_addition(y1,y2,window_width=1):
      ActivityMap1 = y1.abs()
      ActivityMap2 = y2.abs()

      kernel = torch.ones(2*window_width+1,2*window_width+1)/(2*window_width+1)**2
      kernel = kernel.to(device).type(torch.float32)[None,None,:,:]
      kernel = kernel.expand(y1.shape[1],y1.shape[1],2*window_width+1,2*window_width+1)
      ActivityMap1 = F.conv2d(ActivityMap1, kernel, padding=window_width)
      ActivityMap2 = F.conv2d(ActivityMap2, kernel, padding=window_width)
      WeightMap1 = ActivityMap1/(ActivityMap1+ActivityMap2)
      WeightMap2 = ActivityMap2/(ActivityMap1+ActivityMap2)
      return WeightMap1*y1+WeightMap2*y2

def Test_fusion(img_test1,img_test2,addition_mode='Sum'):
    AE_Encoder1 = AE_Encoder().to(device)
    AE_Encoder1.load_state_dict(torch.load(
            ".\Models\Encoder_weight_IJCAI.pkl"
            )['weight'])
    
    AE_Decoder1 = AE_Decoder().to(device)
    AE_Decoder1.load_state_dict(torch.load(
            ".\Models\Decoder_weight_IJCAI.pkl"
            )['weight'])
    AE_Encoder1.eval()
    AE_Decoder1.eval()
    
    img_test1 = np.array(img_test1, dtype='float32')/255# 将其转换为一个矩阵
    img_test1 = torch.from_numpy(img_test1.reshape((1, 1, img_test1.shape[0], img_test1.shape[1])))
    
    img_test2 = np.array(img_test2, dtype='float32')/255 # 将其转换为一个矩阵
    img_test2 = torch.from_numpy(img_test2.reshape((1, 1, img_test2.shape[0], img_test2.shape[1])))
    
    img_test1=img_test1.cuda()
    img_test2=img_test2.cuda()
    
    with torch.no_grad():
        F_i1,F_i2,F_ib,F_id=AE_Encoder1(img_test1)
        F_v1,F_v2,F_vb,F_vd=AE_Encoder1(img_test2)
        
    if addition_mode=='Sum':      
        F_b=(F_ib+F_vb)
        F_d=(F_id+F_vd)
        F_1=(F_i1+F_v1)
        F_2=(F_i2+F_v2)
    elif addition_mode=='Average':
        F_b=(F_ib+F_vb)/2         
        F_d=(F_id+F_vd)/2
        F_1=(F_i1+F_v1)/2
        F_2=(F_i2+F_v2)/2
    elif addition_mode=='l1_norm':
        F_b=l1_addition(F_ib,F_vb)
        F_d=l1_addition(F_id,F_vd)
        F_1=l1_addition(F_i1,F_v1)
        F_2=l1_addition(F_i2,F_v2)
    else:
        print('Wrong!')
         
    with torch.no_grad():
        Out = AE_Decoder1(F_1,F_2,F_b,F_d)
     
    return output_img(Out)