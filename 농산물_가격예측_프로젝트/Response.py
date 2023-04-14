from flask import Blueprint, request, send_file, jsonify
import time
import torch
import cv2
import pandas as pd

bp = Blueprint('response', __name__, url_prefix='/response')

agricultrual_dict_yoon = {
    'chinese-cabbage' : '배추', 'onion' : '양파', 'green-lettuce' : '청상추', 'radish' : '무', 'tomato' : '토마토', 'garlic' : '마늘',
    'greenonion' : '대파', 'cabbage' : '양배추', 'spinach' : '시금치', 'carrot' : '당근', 'paprica' : '파프리카', 'mushroom' : '팽이버섯',
    ' young-squash' : '애호박' , 'grape' : '포도'
}

@bp.route("/table", methods=['GET','POST'])
def table():

    if request.method == "POST":
        params = request.get_json()
        ko_label = agricultrual_dict_yoon[params['label']]
        path = "./formal_data/final_model2/rf_xgb_" + ko_label + ".csv"

    return send_file(path, mimetype='text/csv')