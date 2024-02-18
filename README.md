# OWSOL
Code of our paper Open-World Weakly-Supervised Object Localization.

![](method.png)



## Dependencies

- Python 3
- PyTorch 1.7.1
- OpenCV-Python
- Numpy
- Scipy
- MatplotLib
- faiss-gpu
- munch

## Dataset

### ImageNet-1K

"train" and "val" splits of original [ImageNet](http://www.image-net.org/)  are treated as our `train` and `test`, [ImageNetV2](https://github.com/modestyachts/ImageNetV2) is treated as our `val`.

Make sure your `dataset/ILSVRC`  folder is structured as follows:
```
├── ILSVRC/
|   ├── train/
|	|	|── n01440764
|	|	|── n01443537
|	|	|── ...
|   ├── val
|	|	|── n01440764
|	|	|── n01443537
|	|	|── ...
|   ├── val2
|	|	|── 0
|	|	|── 1
|   |	└── ...
```

## Metadata

You can download the annotations of datasets from [Download metadata](https://pan.baidu.com/s/1QNsIb6UMn63J2XzHGSwONw?pwd=dqdf ), the password is `dqdf`.

Make sure your `metadata/ImageNet`  folder is structured as follows:

```
├── ImageNet/
|   ├── train/
|	|	|── class_labels.txt
|	|	|── image_ids.txt
|	|	|── image_ids_labeled.txt
|   ├── test
|	|	|── class_labels.txt
|	|	|── image_ids.txt
|	|	|── image_sizes.txt
|	|	|── localization.txt.txt
|	|	|── partitions.txt
|   ├── val
|	|	|── class_labels.txt
|	|	|── image_ids.txt
|	|	|── image_sizes.txt
|	|	|── localization.txt.txt
|	|	|── partitions.txt
```

for `test` and `val`, the partitions of **Known**, **Nov-S** and **Nov-D** is described in `partitions.txt`. In detail, **0**, **1** and **2** are correspond to **Known**, **Nov-S** and **Nov-D**, respectively.

## Training

perform contrastive representation co-learning on ImageNet-1K dataset

```
bash bash_script/train_imagenet.sh
```

## G-CAM

perform g-cam on ImageNet-1K dataset

```
bash bash_script/gcam_imagenet.sh
```

