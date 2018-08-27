# SSD_Realization_TensorFlow
[『TensorFlow』SSD项目复现](https://www.cnblogs.com/hellcat/p/9540591.html)<br>
## 使用
###### 生成TFR压缩数据
使用jupyter运行`tfr_generate.ipynb`脚本，使用TFR格式压缩图片对于提升训练速度大有裨益
###### 训练模型
调用脚本即可
```python
python train_ssd_network.py
```

## 日志
#### 18.8.27
截止目前，本改版网络已经可以正常运行稳定收敛，之前的问题及解决思路如下：
###### 1.解决了之前版本Loss值过大且不收敛的问题
这个问题实际上是因为我个人的疏忽，将未预处理的图片送入ssd.bboxes_encode函数中，修正后如下，
```python
image, glabels, gbboxes = \
    tfr_data_process.tfr_read(dataset)

image, glabels, gbboxes = \
    preprocess_img_tf.preprocess_image(image, glabels, gbboxes, out_shape=(300, 300))

gclasses, glocalisations, gscores = \
    ssd.bboxes_encode(glabels, gbboxes, ssd_anchors)
```
这个疏忽导致Loss值维持在200～400之间且不收敛，修改后经过300左右steps损失函数会稳定到60左右，和原SSD网络一致(示意如下)。
![](https://images2018.cnblogs.com/blog/1161096/201808/1161096-20180827102137740-1449500022.png)
###### 2.解决了训练速度过慢的问题
原SSD模型训练速度（CPU：E5-2690，GPU：1080Ti）大概50 samples/sec（实际上略高与此），我的训练速度仅仅22-24 samples/sec，经对比查验，应该是节点分配硬件设备的配置优化问题，涉及队列（主要是数据输入）、优化器设定的节点分派给CPU后（其他节点会默认优先分配给GPU），速度明显提升，大概到达44-46 samples/sec。<br>
另外，tfr数据解析过程放在GPU下，生成队列过程放在CPU下有不明显加速，理想情况下能提升0-1 samples/sec。<br>
综上，现阶段的程序比原程序训练阶段还是要慢5 samples/sec左右，原因还在排查中。<br>
