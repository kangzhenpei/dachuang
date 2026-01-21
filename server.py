# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 14:56:13 2025

@author: Guo
"""

import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import time  # 引入 time 模块来模拟处理时间
import numpy as np
import requests
import uuid

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

from efficientnet_pytorch import EfficientNet

# --- 配置 ---
UPLOAD_FOLDER = 'uploads'  # 设置一个文件夹来存储上传的图片
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}  # 允许的文件扩展名
# 确保上传文件夹存在
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制上传文件大小为 16MB

# 微信小程序配置
APPID = "wxa2b17800f494b2b9"
APPSECRET = "285c56b18e6a08920e34ba4481948031"


def get_access_token():
    """获取微信接口调用凭证"""
    url = f"https://api.weixin.qq.com/cgi-bin/token?grant_type=client_credential&appid={APPID}&secret={APPSECRET}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if 'access_token' in data:
            return data['access_token']
    return None


def check_media_security(media_url, openid):
    """调用微信内容安全检测接口"""
    access_token = get_access_token()
    if not access_token:
        return False, "获取access_token失败"

    check_url = f"https://api.weixin.qq.com/wxa/media_check_async?access_token={access_token}"

    data = {
        "media_url": media_url,
        "media_type": 2,
        "version": 2,
        "scene": 1,
        "openid": openid
    }

    try:
        response = requests.post(check_url, json=data)
        result = response.json()

        if result.get('errcode') == 0:
            return True, result.get('trace_id')
        else:
            return False, f"内容检测失败：{result.get('errmsg')}"
    except Exception as e:
        return False, f"请求失败：{str(e)}"


# --- 辅助函数 ---
def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


class EfficientNetb4(nn.Module):
    def __init__(self):
        super(EfficientNetb4, self).__init__()
        self.EfficientNetb0 = EfficientNet.from_pretrained('efficientnet-b4', num_classes=2)

    def forward(self, x):
        # 前向传播
        x = self.EfficientNetb0(x)
        mu, sigma = x[:, 0], x[:, 1]
        sigma = F.softplus(sigma + 1e-8)

        return mu, sigma


# 裁剪
def CustomCrop(img):
    x1, y1, x2, y2 = 50, 50, 2650, 1350
    # 裁剪图像
    return img.crop((x1, y1, x2, y2))


transform = transforms.Compose([
    CustomCrop,
    transforms.Resize((380, 380)),
    transforms.Grayscale(3),
    transforms.ToTensor(),
])


def model_predict(image_path, gender):
    print(f"prediction for image: {image_path}, gender: {gender}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 模拟处理延迟
    # time.sleep(2)

    if gender == 'male':
        model_state_dict = torch.load('model_male.pth')
    else:
        model_state_dict = torch.load('model_male.pth')

    image = Image.open(image_path)
    input_image = transform(image)
    input_image = input_image.to(device)

    model = EfficientNetb4()
    model.load_state_dict(model_state_dict)
    model.to(device)

    # 设置为评估模式
    model.eval()
    input_image = input_image.unsqueeze(0)  # 添加一个维度

    with torch.no_grad():  # 关闭梯度计算
        input_image = input_image.float()
        age, sigma = model(input_image)

        age = round(age.item(), 2)
        sigma = round(sigma.item(), 2)

        return age, sigma


# --- API 路由 ---
@app.route('/getOpenid', methods=['GET'])
def get_openid():
    code = request.args.get('code')
    if not code:
        return jsonify({'code': 1, 'message': '缺少 code 参数'}), 400

    url = f"https://api.weixin.qq.com/sns/jscode2session?appid={APPID}&secret={APPSECRET}&js_code={code}&grant_type=authorization_code"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if 'openid' in data:
            return jsonify({'code': 0, 'openid': data['openid']})
        else:
            return jsonify({'code': 1, 'message': f"获取 openid 失败: {data.get('errmsg', '未知错误')}"}), 400
    else:
        return jsonify({'code': 1, 'message': '请求微信服务器失败'}), 500


@app.route('/predict', methods=['POST'])
def predict_age():
    """处理图片上传和牙龄预测请求"""
    # 1. 检查请求中是否包含文件部分
    if 'file' not in request.files:
        return jsonify({'code': 1, 'message': '请求中未找到文件部分'}), 400

    file = request.files['file']
    openid = request.form.get('openid')

    # 2. 检查文件名是否存在
    if file.filename == '':
        return jsonify({'code': 1, 'message': '未选择文件'}), 400

    # 3. 检查文件类型和保存文件
    if file and allowed_file(file.filename):
        filename = str(uuid.uuid4()) + os.path.splitext(secure_filename(file.filename))[1]
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(image_path)

            # 构建可访问的URL（需要根据实际部署情况修改）
            media_url = f"https://212393lwyh17.vicp.fun/uploads/{filename}"

            # 调用微信内容安全检测
            if openid:
                check_success, check_result = check_media_security(media_url, openid)
                if not check_success:
                    # 检测失败，删除文件并返回错误
                    if os.path.exists(image_path):
                        os.remove(image_path)
                    return jsonify({'code': 1, 'message': check_result}), 400
            else:
                return jsonify({'code': 1, 'message': '缺少 openid 参数'}), 400

        except Exception as e:
            print(f"Error saving file: {e}")
            return jsonify({'code': 1, 'message': '保存文件失败'}), 500

        # 4. 获取其他表单数据（例如性别）
        gender = request.form.get('gender', 'unknown')  # 从表单获取性别，默认为 unknown

        # 5. 调用模型进行预测（使用模拟函数）
        try:
            age, sigma = model_predict(image_path, gender)
            # 预测成功后可以考虑删除临时文件
            # os.remove(image_path)
            # 返回成功结果 (符合前端期望的格式)
            return jsonify({
                'code': 0,
                'data': {
                    'age': age,
                    'sigma': sigma
                }
            })
        except Exception as e:
            print(f"Prediction error: {e}")
            # 模型预测出错
            # 也可以考虑删除文件
            # if os.path.exists(image_path):
            #    os.remove(image_path)
            return jsonify({'code': 1, 'message': f'预测过程中发生错误: {e}'}), 500
        finally:
            # 无论成功失败，尝试删除临时上传的文件（可选）
            if os.path.exists(image_path):
                try:
                    os.remove(image_path)
                    print(f"Removed temporary file: {image_path}")
                except Exception as remove_err:
                    print(f"Error removing file {image_path}: {remove_err}")

    else:
        # 文件类型不允许
        return jsonify({'code': 1, 'message': '不允许的文件类型'}), 400


# --- 启动应用 ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)  # debug=True 只应在开发时使用