#### 华为云第二届无人车挑战杯海选赛Baseline

---

Baseline基于最新的`Yolov5`项目训练自己的数据集以到达赛题要求的正确区分信号灯、交通标志、人行道的目的。

##### 操作步骤

###### 项目拉取

```
git clone https://github.com/Riser6/AVC_Yolov5.git
```

或者直接手动download

###### 准备工作

1. python环境配置

   若在本地或者个人服务器上，先建立虚拟环境：

   ```
   conda create -yolov5 py37 python=3.7
   conda activate yolov5	#可能需要关闭中断重启
   conda deactivate yolov5	#退出虚拟环境时运行
   ```

   若使用`华为云modelarts`服务器训练，请在`MedelArts`平台Notebook中创建notebook,其中工作环境选择`TF-2.1.0&Pytorch-1.4.0-python3.6`,选择付费规格`GPU`，存储配置使用`云硬盘(EVS)`

   在`Launcher`中选择`Terminal 1`打开中断，输入：

   ```
   python --version	#查看python版本号，貌似预置就是Python 3.7.3，若不是，请按照上面一样建立虚拟环境，python版本达到要求不需要执行虚拟环境配置的命令
   ```

2. 依赖库安装

   项目目录下`requirements.txt`中有依赖库的版本要求，可以直接配置好虚拟环境后，终端键入：

   ```
   pip install -r yolov5/requirements.txt
   ```

   推荐：如果速度慢，可以尝试`sh requiments.sh`安装依赖库，其中脚本前两条指令仅针对使用`华为云modelarts`服务器的用户，指定服务器的`Cuda`版本，本地或者个人服务器用户请查看`Cuda`版本后，在`Pytorch`官网选好对应操作系统，对应`Cuda`版本的安装命令（记得去掉 -c）后执行

3. 数据集准备

   将比赛制定的数据集`labeled_data_backup`上传放置在data目录下（华为云平台操作用户请先看注意事项）

   在项目目录（`./yolov5`）下打开终端，键入`python labelconvect.py`运行已编写好的脚本，用于进行标签转化，将原`xml`类型的标签转化为`txt`类型的标签（Yolo项目要求），运行成功后可以检查data目录下面是否多出了`labelYOLOs`文件夹

   然后在项目目录（`./yolov5`）下打开终端，键入`python train_val_split.py`运行已编写好的脚本，用于进行训练集和验证集的划分，若运行成功可以看见`dataset/HUAWEI_AVC`目录下对应文件夹下面已经有了图片和标签数据。（训练集和验证集按照`7:3`划分，若在正常训练结束后需要将验证集投入训练，直接对调`train`、`val`两个文件夹的名字，重新训练即可）

4. 配置文件修改（已完成）

   models目录下面所有的`yaml`文件第二行：

   ```
   nc: 6
   ```

   data目录下面放置好`HUAWEI_AVC.yaml`文件

5. 预训练权重下载

   前往`https://drive.google.com/drive/folders/1Drs_Aiu7xx6S-ix95f9kNsA6ueKRpN2J`下载预训练模型的权重文件（4个`pt`文件），上传放置在`weights`目录下

   或者直接运行/weights目录下面的`download_weights.sh`

   项目更新后貌似可以运行中直接下载对应模型的预训练权重

###### 训练模型

可以在`train.py`第24~42行调整模型对应的一些超参数

在`train.py`第361~381行命令行选项中更改模型和数据的配置，详见注释（理论上只需要调整我写了中文注释的几个），也可以通过指令键入需要调整的参数

项目目录（`./yolov5`）下

```
python train.py
```

每次训练都会在`run`文件夹下生成一个`exp`目录保存训练日志和训练模型结果，可以在目录下面查看训练日志信息，注意查看`result.jpg`中最后一个折线图（这是这次比赛的指标，若有超过当前成绩，可以考虑提交一波），另外每次训练会保存两个模型结果`best.pt`、`last.pt`(其中best是根据`yolo`项目制定的fitness函数计算，这个指标综合了召回率、精确率、`map`、`map0.5:0.95`,可以直接修改为根据`map0.5:0.95`保存最优模型)。为便于华为云平台发布，需要最后转化一下模型的保存（原项目的模型保存比较特殊，不适合华为云平台发布），记得按注释更改一下每次训练转化读取保存的路径（`train`函数最后几行）。

###### 推理检测

在`inference/images`中上传放置你想要检测的图片，另外在记得更改`if __name__ == '__main__':`，你想要使用的模型文件

执行完后可以在`inference/output`中查看检测结果。

##### 注意

1. 使用华为云平台的用户进行数据传输时，上传超过`100M`数据时（模型文件传输，数据集上传），需要借助OBS桶与notebook进行数据互传，具体操作请查看官方文档介绍`https://support.huaweicloud.com/sdkreference-modelarts/modelarts_04_0127.html`

   传输指令在`ipynb`文件直接执行如果报错的话，请使用命令行python交互环境键入命令

   另外传输依赖库`modelarts`不要直接通过pip安装，请：

   ```
   wget https://cnnorth1-modelarts-sdk.obs.cn-north-1.myhwclouds.com/modelarts-1.1.3-py2.py3-none-any.whl	#直接下载modelarts库的whl文件
   ls #查看当前目录下面是否已经有了该whl文件
    pip install modelarts-1.1.3-py2.py3-none-any.whl #安装modelarts库
   ```

2. 其中项目目录(`./yolov5`)下面的`yolov5`文件夹为用于华为云平台模型导入的文件，项目clone之后可以将这个文件夹剪切到电脑本地，服务器端可以删除，每次模型发布只需要更改`yolov5/model/model.pt`(这个为训练后保存的且经过转化后的模型文件，即`best_convect.pt`或者`last_convect.pt`，理论上上传的文件夹model目录下面只能有一个pt模型文件)，配置文件和推理代码都无需更改，models和`uts`一些`py`文件也尽量不要用原项目对应文件夹下面的文件去替代，因为有些依赖库和部分代码进行了修改（为了适应华为云的发布环境），具体模型导入和发布请查看官方文档介绍。