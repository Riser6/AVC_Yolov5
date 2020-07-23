import os
import random
import shutil

train_percent = 0.7
val_persent = 1-train_percent
imgfilepath = './data/labeled_data_v2'
txtfilepath = './data/labelYOLOs'

img_train_savepath = './dataset/HUAWEI_AVC/images/train'
txt_train_savepath = './dataset/HUAWEI_AVC/labels/train'
img_val_savepath = './dataset/HUAWEI_AVC/images/val'
txt_val_savepath = './dataset/HUAWEI_AVC/labels/val'

if not os.path.exists(img_train_savepath):
        os.makedirs(img_train_savepath)
if not os.path.exists(txt_train_savepath):
        os.makedirs(txt_train_savepath)
if not os.path.exists(img_val_savepath):
        os.makedirs(img_val_savepath)
if not os.path.exists(txt_val_savepath):
        os.makedirs(txt_val_savepath)

total_txt = os.listdir(txtfilepath)
total_img_label = os.listdir(imgfilepath)
num = len(total_txt)
total_range = range(num)

num_train = int(num*train_percent)
train_range= random.sample(total_range,num_train)


for i,txt in enumerate(total_txt,start=0):
    if i in train_range:
        shutil.copy(txtfilepath+"/"+txt,txt_train_savepath)
        shutil.copy(imgfilepath+"/"+txt.split(".")[0]+".jpg",img_train_savepath)
    else:
        shutil.copy(txtfilepath+"/"+txt,txt_val_savepath)
        shutil.copy(imgfilepath+"/"+txt.split(".")[0]+".jpg",img_val_savepath)
print('files copy finish!')