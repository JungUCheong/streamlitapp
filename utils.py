# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Author  :   编程学习园地 
@License :   该项目受专利、软著保护，仅供个人学习使用，严禁倒卖，一经发现，编程学习园地团队有必要追究法律责任！！！
'''


import torch
from pathlib import Path
from config import MODEL_LIST,MODEL_DIR

from ultralytics import YOLO
import streamlit as st

@st.cache_resource
def load_model():
    """
    从具体的路径中加载一个模型,并根据当前环境自动加载模型(GPU环境选择pt模型推理,CPU环境选择加载onnx模型进行加速)
    Returns:
        A YOLO object detection model.
    """
    pt_model_path = None
    onnx_model_path = None

    for MODEL in MODEL_LIST:
        if MODEL.endswith('.pt'):
            pt_model_path = Path(MODEL_DIR, MODEL)
            continue
        if MODEL.endswith('.onnx'):
            onnx_model_path = Path(MODEL_DIR, MODEL)
            continue
    if pt_model_path==None and onnx_model_path==None:
        raise Exception('无模型文件')
    if pt_model_path!=None:
        if torch.cuda.is_available():
            model=YOLO(pt_model_path)
            return model
        if onnx_model_path==None:
            model=YOLO(pt_model_path, weights_only=False)
            return model
    if onnx_model_path!=None:
        model=YOLO(onnx_model_path, weights_only=False)
        return model
def infer_image(model,image,conf,iou,is_save=False):
    '''
    图片推理
    '''
    res=model.predict(source=image,conf=conf,iou=iou,save=is_save)
    anno_img=res[0].plot()
    labels = res[0].names
    boxes = res[0].boxes
    labels_num_dict = {}
    rows=[]
    for index,box in enumerate(boxes):
        row=[]
        lable_index = box.cls.cpu().detach().numpy()[0].astype(int)
        label_name = labels[lable_index]
        xyxy=box.xyxy.cpu().tolist()[0]
        x1,y1,x2,y2=int(xyxy[0]),int(xyxy[1]),int(xyxy[2]),int(xyxy[3])
        confidence=box.conf.cpu().detach().numpy()[0]
        conf="{:.2f}".format(confidence)
        row.append(index+1)
        row.append(label_name)
        row.append(conf)
        row.append({'x1':x1,'y1':y1,'x2':x2,'y2':y2})
        rows.append(row)
        for key in labels.keys():
            if int(lable_index) == key:
                if labels[key] in labels_num_dict:
                    labels_num_dict[labels[key]] += 1
                else:
                    labels_num_dict[labels[key]] = 1
    return anno_img,labels_num_dict,rows
def infer_video_frame(model,image,conf,iou,is_save=False):
    '''
    视频推理和本地摄像头推理
    '''
    res = model.predict(source=image, conf=conf, iou=iou)
    anno_img = res[0].plot()
    return anno_img
