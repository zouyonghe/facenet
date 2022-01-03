# 使用Tensorflow实现的人脸识别
这是论文 “FaceNet: A Unified Embedding for Face Recognition and Clustering“ 中描述的人脸识别器基于Tensorflow的实现。该项目也使用了牛津大学 Visual Geometry Group 发表的 “Deep Face Recognition“ 论文中一些观点。

## 兼容性
该项目在 Arch rolling 和 Anaconda 中 python3.6.13 下 Tensorflow==1.9 和 Tensorflow-gpu==1.9 进行测试。

## 预训练模型
| 模型名称 | LFW准确率 | 训练数据集 | Architecture |
| -------- | --------- | ---------- | ------------ |
| [20180408-102900](https://drive.google.com/open?id=1R77HmFADxe87GmoLwzfgMu_HY0IhcyBz) | 0.9905        | CASIA-WebFace    | [Inception ResNet v1](https://github.com/davidsandberg/facenet/blob/master/src/models/inception_resnet_v1.py) |
| [20180402-114759](https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-) | 0.9965        | VGGFace2      | [Inception ResNet v1](https://github.com/davidsandberg/facenet/blob/master/src/models/inception_resnet_v1.py) |
| 20211204-143911 |0.96433+-0.01111| CASIA-WebFace | [Inception ResNet v1](https://github.com/davidsandberg/facenet/blob/master/src/models/inception_resnet_v1.py) |

## 启发

该代码深受 [OpenFace](https://github.com/cmusatyalab/openface) 的启发。

## 数据集

The [CASIA-WebFace](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html) dataset has been used for training.
This training set consists of total of 453 453 images over 10 575 identities after face detection. Some performance
improvement has been seen if the dataset has been filtered before training. Some more information about how this was
done will come later. The best performing model has been trained on
the [VGGFace2](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) dataset consisting of ~3.3M faces and ~9000 classes.

## 开始
0. 文件夹结构

～/datasets 存储数据集
～/models 存储预训练模型
～/Project 存储项目文件

1. 克隆存储库

```shell
mkdir ~/Project

cd ~/Project

#git clone https://github.com/davidsandberg/facenet.git 原始项目存储库
git clone https://github.com/zouyonghe/facenet.git      #zouyonghe的存储库
```

2. 准备conda环境

```shell
conda create -n facenet tensorflow==1.9
#conda create -n facenet tensorflow-gpu==1.9 使用支持gpu的tensorflow,生成预训练模型速度更快
conda activate facenet

pip install scipy==1.2.1 sklearn opencv-python Pillow

export PYTHONPATH=~/Project/facenet/src #若不设置环境变量，python会找不到模块

cd ~/Project/facenet
```

3. 对齐数据集

```shell
for N in {1...4}; do \
python src/align/align_dataset_mtcnn.py \
~/datasets/CASIA-WebFace/CASIA-WebFace \ #数据集位置
~/datasets/CASIA-WebFace/CASIA-WebFace_mtcnnalign_160 \ #对齐后图片位置
--image_size 160 \ #图片大小
--margin 32 \
--random_order \ #随机顺序
--gpu_memory_fraction 0.7 \ #显存使用比例
& done
```

4. 生成预训练模型
```shell
#使用三元损失训练，训练时间长，难以回归，但会有更好的准确率
python src/train_tripletloss.py \
--models_base_dir ~/models/facenet  \
--model_def models.inception_resnet_v1 \
--data_dir ~/datasets/lfw/lfw_mtcnnpy_160 \
--image_size 160 \
--optimizer RMSPROP  \
--max_nrof_epochs 150 \
--keep_probability 0.8 \
--random_crop \
--random_flip  \
--weight_decay 5e-5 \
--alpha 0.1 \
--gpu_memory_fraction 0.8 \
--batch_size 3

#使用中心损失训练，回归迅速，后期准确率提升慢
python src/train_softmax.py \
--logs_base_dir ~/logs/facenet \
--models_base_dir ~/models/facenet \
--data_dir ~/datasets/CASIA-WebFace/CASIA-WebFace_160 \
--image_size 160 \
--model_def models.inception_resnet_v1 \
--lfw_dir ~/datasets/lfw/lfw_mtcnnpy_160 \
--optimizer ADAM \
--learning_rate 0.01 \
--max_nrof_epochs 500 \
--keep_probability 0.8 \
--random_crop \
--random_flip \
--weight_decay 5e-4 \
--embedding_size 512 \
--lfw_distance_metric 1 \
--lfw_use_flipped_images \
--lfw_subtract_mean \
--validation_set_split_ratio 0.05 \
--validate_every_n_epochs 5 \
--prelogits_norm_loss_factor 5e-4 \
--gpu_memory_fraction 0.7 \
--batch_size 32

#预训练模型准确率测试
python src/validate_on_lfw.py \
~/datasets/lfw/lfw_mtcnnpy_160 \
~/models/facenet/20180402-114759 \
--distance_metric 1 \
--use_flipped_images \
--subtract_mean \
--use_fixed_image_standardization

#使用TensorBoard展示训练进度
tensorboard --logdir=~/logs/facenet --port 6006
```

5. 训练分类器
```shell
python src/classifier.py  \ 
TRAIN \
~/datasets/lfw/lfw_mtcnnpy_160 \
~/models/facenet/20211204-143911/20211204-143911.pb \
~/models/lfw_classifier.pkl \
--batch_size 100 \
--min_nrof_images_per_class 40 \
--nrof_train_images_per_class 35 \
--use_split_dataset

#测试训练完成的分类器
python src/classifier.py \  
CLASSIFY \
~/datasets/lfw/lfw_mtcnnpy_160 \
~/models/facenet/20211204-143911/20211204-143911.pb \
~/models/lfw_classifier.pkl \
--batch_size 100 \
--min_nrof_images_per_class 40 \
--nrof_train_images_per_class 35 \
--use_split_dataset
```

6. 图片分类
```shell
python contributed/predict.py \
~/Pictures/pic.jpg \
~/models/facenet/20211204-143911 \
~/models/facenet/lfw_classifier.pkl
```

## 性能

The accuracy on LFW for the model [20180402-114759](https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-)
is 0.99650+-0.00252. A description of how to run the test can be found on the
page [Validate on LFW](https://github.com/davidsandberg/facenet/wiki/Validate-on-lfw). Note that the input images to the
model need to be standardized using fixed image standardization (use the option `--use_fixed_image_standardization` when
running e.g. `validate_on_lfw.py`).
