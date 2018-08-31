# Author : hellcat
# Time   : 18-8-30

"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
 
import numpy as np
np.set_printoptions(threshold=np.inf)
 
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
"""

import tensorflow as tf

from eval.eval_img_process_tf import preprocess_for_eval, Resize
from ssd_vgg300_tf import SSDNet

tf.logging.set_verbosity(tf.logging.INFO)
slim = tf.contrib.slim

gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)


img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
image_pre, labels_pre, bboxes_pre, bbox_img = preprocess_for_eval(
    img_input,
    None,
    None,
    (300, 300),
    'NHWC',
    resize=Resize.WARP_RESIZE
)
image_4d = tf.expand_dims(image_pre, 0)

ssd = SSDNet()
predictions, localisations, _, _ =\
    ssd.net(image_4d, is_training=False)

isess.run(tf.global_variables_initializer())

ckpt = tf.train.get_checkpoint_state('./logs/model/')
# saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
saver = tf.train.Saver()
saver.restore(isess, ckpt.model_checkpoint_path)


# 在网络模型结构中，提取搜索网格的位置
ssd_anchors = ssd.anchors


import cv2
from eval import np_methods
import matplotlib.pyplot as plt
import matplotlib.cm as mpcm
l_VOC_CLASS = [
                'aeroplane',   'bicycle', 'bird',  'boat',      'bottle',
                'bus',         'car',     'cat',   'chair',     'cow',
                'diningTable', 'dog',     'horse', 'motorbike', 'person',
                'pottedPlant', 'sheep',   'sofa',  'train',     'TV'
]


def colors_subselect(colors, num_classes=21):
    dt = len(colors) // num_classes
    sub_colors = []
    for i in range(num_classes):
        color = colors[i*dt]
        if isinstance(color[0], float):
            sub_colors.append([int(c * 255) for c in color])
        else:
            sub_colors.append([c for c in color])
    return sub_colors


def bboxes_draw_on_img(img, classes, scores, bboxes, colors, thickness=2):
    shape = img.shape
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i]
        color = colors[classes[i]]
        # Draw bounding box...
        p1 = (int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))
        p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
        cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)
        # Draw text...
        s = '%s/%.3f' % ( l_VOC_CLASS[int(classes[i])-1], scores[i])
        p1 = (p1[0]-5, p1[1])
        #cv2.putText(img, s, p1[::-1], cv2.FONT_HERSHEY_DUPLEX, 1.5, color, 3)

colors_plasma = colors_subselect(mpcm.plasma.colors, num_classes=21)


# Main image processing routine.
def process_image(img, select_threshold=0.3, nms_threshold=.8, net_shape=(300, 300)):
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})

    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            rpredictions, rlocalisations, ssd_anchors,
            select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)

    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    bboxes_draw_on_img(img, rclasses, rscores, rbboxes, colors_plasma, thickness=8)
    return img

img = cv2.imread("./eval/timg.jpeg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(process_image(img))
plt.show()

