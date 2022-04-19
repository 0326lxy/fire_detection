# object_detection_example_1.py演示一个完整的推理(Inference)过程

# -----------------------------------------------------------
# 第一步，导入相关的软件包
# -----------------------------------------------------------
import numpy as np 
import os
import tensorflow as tf 
import matplotlib.pyplot as plt 
from PIL import Image 
from utils import label_map_util
from utils import visualization_utils as vis_util 
from utils import ops as utils_ops

# 检查tensorflow 版本，须≥1.12.0
from pkg_resources import parse_version
if parse_version(tf.__version__) < parse_version('1.12.0'):
    raise ImportError("parse_version:Please upgrade your TensorFlow to V1.12.* or higher")
print("The version of installed TensorFlow is {0:s}".format(tf.__version__))

# -----------------------------------------------------------
# 第二步，导入模型fire_detection_model到内存
# fire_detection_model文件夹应与本程序放在\trained_frozen_models\cats_dogs_model文件夹下
# -----------------------------------------------------------
MODEL_NAME = 'trained_frozen_models/fire-detection_model'
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('annotations', 'label_map.pbtxt')

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# -----------------------------------------------------------
# 第三步，导入标签映射文件(Label map)，这样假如神经网络输出'1',我
# 们就知道对应的是'smoke'
# -----------------------------------------------------------
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# -----------------------------------------------------------
# 第四步，执行推理(Inference)，检测图片中的对象
# -----------------------------------------------------------

# ## 导入图像数据到numpy array 子程序
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size 
    return np.array(image.getdata()).reshape((im_height,im_width,3)).astype(np.uint8)

# ## 从单张图片中检测对象子程序
# ## 图片名称：1.jpg, 2.jpg...20.jpg存放在
# ## \images\test文件夹下
PATH_TO_IMAGES_DIR = 'images/test'
TEST_IMAGE_PATHS = [os.path.join(PATH_TO_IMAGES_DIR, 
                            '{0:d}.jpg'.format(i)) for i in range(33,34)]

# 显示图像的尺寸，单位inches
IMAGE_SIZE = (12, 8)

def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            if 'detection_masks' in tensor_dict:
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[1], image.shape[2])
                detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # 运行推理(Inference)
            output_dict = sess.run(tensor_dict,feed_dict={image_tensor: image})

            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.int64)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict

for image_path in TEST_IMAGE_PATHS:
    image = Image.open(image_path)
    image_np = load_image_into_numpy_array(image)
    # 扩展维度，因为模型要求图像的形状为：[1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # 运行检测程序.
    output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
    # 可视化检测结果.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=8)
    plt.figure(figsize=IMAGE_SIZE)
    plt.imshow(image_np)
plt.show()
