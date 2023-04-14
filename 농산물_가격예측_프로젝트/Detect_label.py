from flask import Blueprint, request, send_file, jsonify
import time
import torch
import cv2
import pandas as pd
import pymysql
import numpy as np

bp = Blueprint('', __name__, url_prefix='/detect_label')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model call
detect_model = torch.hub.load('ultralytics/yolov5', 'custom', path="./models/best.pt")

# inference Settings
detect_model.conf = 0.3
detect_model.iou = 0.45

# use Nvidia cuda
detect_model.to(device)

def detect_inference(dict, image):

    dict["label"] = "detect_fail"
    detect = detect_model(image, size=640)
    res_pd = detect.pandas().xyxy[0]
    
    if res_pd.empty == False:
        dict["label"] = str(res_pd.loc[0, ['name']].item())
        return dict
    return dict


@bp.route("/", methods=['GET','POST'])
def detect():

    # 반환용 dict 만들기
    dict = {"label" : "access_fail"}

    if request.method == "POST":
        f = request.files["agricultural_img"]
        # 사진 저장안하고 바로 사용
        file_bytes = np.fromfile(f, np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        dict = detect_inference(dict, image)
        print(dict)
        return jsonify(dict)

    return jsonify(dict)
