# -*- encoding: utf-8 -*-
'''
@File    :   app.py
@Author  :   编程学习园地 
@License :   该项目受专利、软著保护，仅供个人学习使用，严禁倒卖，一经发现，编程学习园地团队有必要追究法律责任！！！
'''

import streamlit as st
from PIL import Image
from utils import load_model,infer_image,infer_video_frame
from config import *
import  tempfile
import cv2
import pandas as pd

st.set_page_config(
    page_title="code learn corner",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
    )
hide_streamlit_style = """
            <style>
            MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
login_status = st.session_state.get('login_status', False)
register_status=st.session_state.get('register_status', False)
if not login_status:
    st.markdown("<h1 style='text-align: center;'>欢迎进入西红柿外部品质检测系统</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>Welcome to the Tomato External Quality Detection System</h2>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>登录界面 Login Page</h3>", unsafe_allow_html=True)
    input_username = st.text_input("用户名 Username:")
    input_password = st.text_input("密码 Password:", type="password")
    # input_authentication = st.radio("角色:", ("管理员", "游客"))
    col1,col2=st.columns(2)
    with col1:
        if st.button("登录 Login"):
            if 'username' not in st.session_state:
                st.error('检测到没有账号!请点击注册 Please click Register')
            elif not input_username or not input_username:
                st.error('用户名或者密码不能为空!')
            elif st.session_state['username']!=input_username:
                st.error('用户名错误!')
            elif st.session_state['password']!=input_password:
                st.error('密码错误!')
            else:
                st.session_state.login_status=True
                st.rerun()
    with col2:
        if st.button('没有账号?点击注册 Please click Register'):
            st.session_state.login_status = True
            st.session_state.register_status = True
            st.rerun()
elif register_status:
    st.markdown("<h1 style='text-align: center;'>欢迎进入西红柿外部品质检测系统</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>Welcome to the Tomato External Quality Detection System</h2>",
                unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>注册界面 Register Page</h3>", unsafe_allow_html=True)
    username = st.text_input("输入用户名 Enter Username:")
    password = st.text_input("输入密码 Enter Password:", type="password")
    confirm_password = st.text_input("确认密码 Confirm Password:", type="password")
    authentication = st.radio("角色 Role:", ("管理员 Admin", "游客 Guest"))
    if st.button("注册 Register"):
        if password!=confirm_password:
            st.error('两次输入的密码不一致 The two passwords do not match!')
        elif not username:
            st.error('用户名不能为空 Username cannot be empty!')
        elif not password or not confirm_password:
            st.error('密码不能为空 Password cannot be empty!')
        else:
            st.session_state.login_status = False
            st.session_state.register_status = False
            st.session_state.username=username
            st.session_state.password=password
            st.session_state.authentication=authentication
            st.rerun()
else:
    st.markdown("<h1 style='text-align: center;'>西红柿外部品质检测系统</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>Tomato External Quality Detection System</h2>", unsafe_allow_html=True)
    st.sidebar.header("配置面板 Configuration Panel")
    confidence = float(st.sidebar.slider(
    "调整置信度 Adjust Confidence", 10, 100, 25)) / 100
    iou=float(st.sidebar.slider(
    "调整IOU Adjust IoU", 10, 100, 45)) / 100
    source = ("图片检测 Image Detection", "视频检测 Video Detection",'摄像头检测 Camera Detection')
    select_radio=st.sidebar.radio('检测类型 Detection Type',source)
    source_index=source.index(select_radio)
    model=load_model()
    if source_index == 0:
        uploaded_file = st.sidebar.file_uploader(
            "上传图片 Upload Image", type=['png', 'jpeg', 'jpg'])
        if uploaded_file is not None:
            with st.spinner(text='图片推理执行中...'):
                st.sidebar.image(uploaded_file,caption="原始图片")
                picture = Image.open(uploaded_file)
                anno_img, label_num_dict,rows = infer_image(model=model,image=picture, conf=confidence, iou=iou)
                st.image(anno_img,channels='BGR',caption="图片检测结果")
                df=pd.DataFrame(rows,columns=['序号','类别','置信度','box坐标'])
                df1=pd.DataFrame(list(label_num_dict.items()),columns=['类别','总数'])
                if st.session_state.authentication is not None:
                    if st.session_state['authentication']=='admin':
                        st.data_editor(df,hide_index=True)
                        st.data_editor(df1)
                    elif st.session_state.authentication is not None:
                        st.data_editor(df,hide_index=True,disabled=True)
                        st.data_editor(df1,disabled=True)
    elif source_index==1:
        uploaded_file = st.sidebar.file_uploader("上传视频 Upload Video", type=['mp4'])
        if uploaded_file is not None:
            st.sidebar.video(uploaded_file)
            tfile = tempfile.NamedTemporaryFile(dir=ROOT / 'img_video')
            tfile.write(uploaded_file.read())
            vid_cap = cv2.VideoCapture(tfile.name)
            st_frame = st.empty()
            with st.spinner(text='视频推理执行中...'):
                while (vid_cap.isOpened()):
                    success, image = vid_cap.read()
                    if success:
                        # 调用方法逐帧预测
                        anno_img=infer_video_frame(model=model,image=image,conf=confidence,iou=iou,is_save=False)
                        st_frame.image(anno_img, channels='BGR', caption="视频检测结果")
                    else:
                        vid_cap.release()
                        break

    else:
        with st.spinner(text='摄像头推理执行中...'):
            flag = st.button(
                label="停止检测/stop"
            )
            while not flag:
                vid_cap = cv2.VideoCapture(0)  # 调用本地摄像头
                # 页面创建空容器实时播放画面
                st_frame = st.empty()
                while (vid_cap.isOpened()):
                    success, image = vid_cap.read()
                    if success:
                        # 调用方法逐帧预测
                        anno_img=infer_video_frame(model=model, image=image, conf=confidence, iou=iou,is_save=False)
                        st_frame.image(anno_img, channels='BGR', caption="摄像头检测结果")
                    else:
                        vid_cap.release()
                        break