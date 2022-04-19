import time

start = time.time()
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import pandas as pd

cv2.namedWindow("frame", 0)
cv2.resizeWindow("frame", 640, 480)

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

os.chdir('D:\\tf_train\\models\\research\\object_detection')

# load model

PATH_TO_FROZEN_GRAPH = 'D:/tf_train/workspaces/fire_detection/trained_frozen_models/firedetection_model/frozen_inference_graph.pb'

# load labels
PATH_TO_LABELS = 'D:/tf_train/workspaces/fire_detection/annotations/label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

detection_graph = tf.Graph()


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def video_capture(image_tensor, detection_boxes, detection_scores, detection_classes, num_detections, sess, video_path):
    if video_path == "nopath":
        # 0是代表摄像头编号，只有一个的话默认为0
        cap = cv2.VideoCapture(1)
        print("open")
    else:
        cap = cv2.VideoCapture(video_path)
    i = 1
    while 1:
        ret, frame = cap.read()
        if ret:
            i = i + 1
            if i % 20 == 0:
                loss_show(image_tensor, detection_boxes, detection_scores, detection_classes, num_detections, frame,
                          sess)
            else:
                cv2.imshow("frame", frame)
            # 等待5ms显示图像，若过程中按“Esc”退出
            c = cv2.waitKey(256) & 0xff
            if c == 27:  # ESC 按键 对应键盘值 27
                cap.release()
                break
        else:
            print("ref == false ")
            break


def init_ogject_detection(video_path):
    with detection_graph.as_default():
        with tf.compat.v1.Session(config=tf.ConfigProto(device_count={'gpu': 0})) as sess:
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            video_capture(image_tensor, detection_boxes, detection_scores, detection_classes, num_detections, sess,
                          video_path)


def loss_show(image_tensor, detection_boxes, detection_scores, detection_classes, num_detections, image_np, sess):
    starttime = time.time()
    image_np = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
    image_np = load_image_into_numpy_array(image_np)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    # print("--scores--->", scores)
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=5,
        min_score_thresh=.5)
    # write images
    # 保存识别结果图片
    print("------------use time ====> ", time.time() - starttime)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    cv2.imshow("frame", image_np)


if __name__ == '__main__':
    VIDEO_PATH = "D:/tf_train/workspaces/fire_detection/images/test9.mp4"  # 本地文件传入文件路径   调用camera 传入'nopath'
    init_ogject_detection(VIDEO_PATH)
