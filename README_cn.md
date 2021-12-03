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

pip install scipy sklearn opencv-python Pillow

export PYTHONPATH=~/Project/facenet/src

cd ~/Project/facenet
```

3. 开始对齐，时间较长

```shell
for N in {1..4}; do \
python src/align/align_dataset_mtcnn.py \
~/datasets/lfw/lfw \
~/datasets/lfw/lfw_mtcnnpy_160 \
--image_size 160 \
--margin 32 \
--random_order \
--gpu_memory_fraction 0.25 \
& done
```

4. 生成预训练模型
```shell
python src/train_tripletloss.py \
--models_base_dir ~/models/facenet  \
--model_def models.inception_resnet_v1 \
--data_dir ~/datasets/lfw/lfw_mtcnnpy_160 \
--image_size 160 \
--optimizer RMSPROP  \
--max_nrof_epochs 20 \
--keep_probability 0.8 \
--random_crop \
--random_flip  \
--weight_decay 5e-5 \
--alpha 0.1 \
--gpu_memory_fraction 0.6 \
--batch_size 3  
```

5. 运行准确性测试

```shell
   python src/validate_on_lfw.py \
   ~/datasets/lfw/lfw_mtcnnpy_160 \
   ~/models/facenet/20180402-114759 \
   --distance_metric 1 \
   --use_flipped_images \
   --subtract_mean \
   --use_fixed_image_standardization
```
