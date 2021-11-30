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

conda activate facenet

pip install scipy sklearn opencv-python Pillow

export PYTHONPATH=~/Project/facenet/src
```

3. lfw数据集放在～/datasets/lfw中(最终结果是~/datasets/lfw/lfw/人名/图片名）

4. 预训练模型放在～/models中

   1）下载预训练模型

   https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-

   2）解压放在~/models/facenet中

5. 修改(如果是zouyonghe的存储库中的facenet，不用修改）

6. 开始对齐，时间较长

```shell
cd ~/Project/facenet

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

7. 训练完毕，在～/datasets/lfw/lfw_mtcnnpy_160中查看结果

8. 运行测试

```shell
   python src/validate_on_lfw.py \
   ~/datasets/lfw/lfw_mtcnnpy_160 \
   ~/models/facenet/20180402-114759 \
   --distance_metric 1 \
   --use_flipped_images \
   --subtract_mean \
   --use_fixed_image_standardization
```

