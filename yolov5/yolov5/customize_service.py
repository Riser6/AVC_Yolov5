from PIL import Image
import torch.nn.functional as F
from model_service.pytorch_model_service import PTServingBaseService
import torch.nn as nn
import torch
import json
import numpy as np
from models.experimental import *
from uts.datasets import *
from uts.utils import *
import torch
import torchvision.transforms as transforms

import os

augment=False
conf_thres=0.4
imgsz=1280
iou_thres=0.5
Classes=None
Agnostic_nms=False
device=torch.device('cpu')
class_dict={0:'red_stop',1:'green_go',2:'yellow_back',3:'pedestrian_crossing',4:'speed_limited',5:'speed_unlimited'}

class PTVisionService(PTServingBaseService):

    def __init__(self, model_name, model_path):
        #super(PTVisionService, self).__init__(model_name, model_path)
        # 调用自定义函数加载模型
        self.model=torch.load(model_path, map_location=device)
        imgsz = check_img_size(1280, s=self.model.stride.max())
        self.model.requires_grad_(False)
        self.model.eval()

    def _preprocess(self, data):
        # https两种请求形式
        # 1. form-data文件格式的请求对应：data = {"请求key值":{"文件名":<文件io>}}
        # 2. json格式对应：data = json.loads("接口传入的json体")
        imgs_path = []
        for k, v in data.items():
            for file_name, file_content in v.items():
                imgs_path.append(file_content)
        return imgs_path


    def _inference(self, imgs_path):
        results = []
        dataset = LoadImages(imgs_path, img_size=imgsz)
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            img=img.permute(2,0,1)
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            # Inference
            pred = self.model(img, augment=augment)[0]
            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes=Classes, agnostic=Agnostic_nms)
            #预测y坐标后处理
            pred_copy=[]
            for i, det in enumerate(pred):
                im0 = im0s
                if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                pred_copy.append(det)
            pred=pred_copy
            if len(pred) > 0:
                for single_pred in pred:
                    image_result = {
                        'detection_classes': [],
                        'detection_boxes': [],
                        'detection_scores': []
                        }
                    if isinstance(single_pred, type(None)):
                        results.append(image_result)
                    else:
                        for roi_list in single_pred:
                            image_result['detection_classes'].append(class_dict[int(roi_list[-1])])
                            image_result['detection_boxes'].append([float(roi_list[1]),float(roi_list[0]),float(roi_list[3]),float(roi_list[2])])
                            image_result['detection_scores'].append(float(roi_list[-2]))
                        results.append(image_result)
            else:
                image_result = {
                    'detection_classes': [],
                    'detection_boxes': [],
                    'detection_scores': []
                    }
                result.append(image_result)
        return results

    def _postprocess(self, data):
        if len(data) == 1:
            return data[0]
        else:
            return data
