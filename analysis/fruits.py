import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import pickle
from PIL import Image
from fastai.vision.all import *

# 加载数据和模型
data = pd.read_excel('analysis/fruits.xlsx')
scaler = StandardScaler()
features = data[['Vitamin A', 'Vitamin C', 'Vitamin E', 'Dietary Fiber', 'Calcium', 'Magnesium', 'Iron']]
labels = data['fruits']
features_scaled = scaler.fit_transform(features)

# 加载训练好的 KNN 模型
with open('analysis/knn_model.pkl', 'rb') as f:
    knn = pickle.load(f)

# 加载 CNN 模型
learn = load_learner('analysis/export.pkl')

# 设置页面标题和介绍
st.title("基于营养素需求的果蔬推荐系统")
st.write("请选择您需要补充的营养素：")

# 创建下拉菜单，供用户选择营养素
nutrients = ['Vitamin A', 'Vitamin C', 'Vitamin E', 'Dietary Fiber', 'Calcium', 'Magnesium', 'Iron']
nutrient_needed = st.selectbox("选择营养素", nutrients)

# 获取用户输入的营养素需求量
amount_needed = st.number_input("请输入您期望补充的营养素量 (例如：30)", value=30)

# 定义一个函数来展示图片
def show_image(image_path):
    img = Image.open(image_path)
    st.image(img, caption=image_path, use_column_width=True)

# 当用户点击按钮时，进行推荐
if st.button("获取推荐"):
    # 将用户输入转换为特征向量
    user_input = np.zeros(len(features.columns))
    user_input[features.columns.get_loc(nutrient_needed)] = amount_needed
    user_input_scaled = scaler.transform([user_input])

    # 使用 KNN 模型进行预测
    _, indices = knn.kneighbors(user_input_scaled, n_neighbors=3)
    recommended_fruits = [labels[index] for index in indices[0]]

    # 展示推荐结果
    st.write("根据您的需求，我们为您推荐以下三种水果：")
    for fruit in recommended_fruits:
        st.write(fruit)
        # 假设图片存储在 "images/" 目录下，图片名称与水果名称相同
        image_path = f"images/{fruit}.jpg"
        show_image(image_path)

        # 使用 CNN 模型预测图片标签
        img = PILImage.create(image_path)
        pred, _, _ = learn.predict(img)
        st.write(f"预测的标签: {pred}")
