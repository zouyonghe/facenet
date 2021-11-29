1. 克隆存储库
```shell
mkdir ~/Project

cd ~/Project

git clone https://github.com/davidsandberg/facenet.git
```

2. 准备conda环境
```shell
conda create -n facenet tensorflow==1.9

conda activate facenet

pip install scipy sklearn opencv-python Pillow

export PYTHONPATH=~/Project/facenet/src
```
3. lfw数据集放在～/datasets中

4. 预训练模型放在～/models中

   1）下载预训练模型

   https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-

   2）解压放在~/models/facenet中

5. 修改(如果是zouyonghe的存储库中的分支，不用修改）

   1)align下detect_face.py的85行

![img](/home/buding/Pictures/pic1.png)
```PYTHON
img = cv2.imread(image_path)
```

   2)在align_datasetmtcnn中添加两个import
```python
import cv2
from PIL import Image
```
![image-20211129174710234](/home/buding/.config/Typora/typora-user-images/image-20211129174710234.png)

   3）修改align_dataset_mtcnn的126行注释掉，添加127行

```python
scaled = np.array(Image.fromarray(cropped).resize((args.image_size, args.image_size)))
```

![image-20211129175153228](/home/buding/.config/Typora/typora-user-images/image-20211129175153228.png)

4)修改134行，注释掉，增加

```python
cv2.imwrite(output_filename_n, scaled)
```

![image-20211129175255406](/home/buding/.config/Typora/typora-user-images/image-20211129175255406.png)


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

![image-20211129175522453](/home/buding/.config/Typora/typora-user-images/image-20211129175522453.png)

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
8. 测试结果![image-20211129184700899](/home/buding/.config/Typora/typora-user-images/image-20211129184700899.png)
