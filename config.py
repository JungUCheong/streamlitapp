# -*- encoding: utf-8 -*-
'''
@File    :   config.py
@Author  :   编程学习园地 
@License :   该项目受专利、软著保护，仅供个人学习使用，严禁倒卖，一经发现，编程学习园地团队有必要追究法律责任！！！
'''


import os
import sys
from pathlib import Path
#获取当前文件的绝对路径
file_path = Path(__file__).resolve()

#获取当前文件的上一级目录的路径
root_path = file_path.parent

#如果当前文件的父目录不在搜索路径中则添加进去
if root_path not in sys.path:
    sys.path.append(str(root_path))

#获取当前项目(工作目录)的相对路径
ROOT = root_path.relative_to(Path.cwd())
MODEL_DIR = ROOT / 'weights'
# MODEL_DIR = r'D:\AAA_ECJTU_files\object_detection\html\yolo8html\ultralyticsmain\runs\train\exp31\weights\best.pt'
#注意：如果你想要加载自己训练的模型(yolov3、v5、v6、v7、v8及其微调版本,这些都支持,修改了网络结构的不支持)
# 只需将自己的模型放在weights目录下并删除原本的全部模型文件即可

#侧边栏模型选择列表
MODEL_LIST=[]
weights=os.listdir(MODEL_DIR)
#支持pytroch原生推理与onnx推理
for weight in weights:
    if weight.endswith('.onnx'):
        MODEL_LIST.append(weight)
    elif weight.endswith('.pt'):
        MODEL_LIST.append(weight)
