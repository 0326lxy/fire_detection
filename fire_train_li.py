"""
Usage:
python one_command_train.py [--steps <number_of_training_steps>] [--batch_size <batch_size>]

Example usage:
1) python one_command_train.py
2) python one_command_train.py --steps 500
3) python one_command_train.py --batch_size 12
4) python one_command_train.py --steps 500 --batch_size 12
"""


import os
import sys
import glob
import shutil
import re
import argparse
import xml.etree.ElementTree as ET
import urllib.request
import tarfile


def check_directories_and_documents():
    # check one_click_train.py location.
    pattern = re.compile(r".*/workspaces/.+")
    current_work_directory = os.getcwd().replace('\\', '/')
    if re.match(pattern, current_work_directory) is None:
        print("错误：one_command_train.py脚本位置不正确。")
        return

    # check annotations directory
    if not os.path.exists('annotations'):
        os.mkdir('annotations')

    # check images directory
    if not os.path.exists('images'):
        os.mkdir('images')
    if not os.path.exists('images/train'):
        os.mkdir('images/train')
    if not os.path.exists('images/eval'):
        os.mkdir('images/eval')
    if not os.path.exists('images/test'):
        os.mkdir('images/test')
    if len(glob.glob('images/train/*.xml')) == 0 or len(glob.glob('images/eval/*.xml')) == 0:
        print("错误：images/train或images/eval中缺少*.xml标注文件。")
        return

    # check pre_trained_model directory
    if not os.path.exists('pre_trained_model'):
        os.mkdir('pre_trained_model')
    if len(glob.glob('pre_trained_model/ssd_inception_v2_coco*/model.ckpt.*')) == 0:
        print("Downloading ssd_inception_v2_coco model...")
        url = 'http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz'
        save_as = 'pre_trained_model/ssd_inception_v2_coco_2018_01_28.tar.gz'
        urllib.request.urlretrieve(url, save_as)
        print("extracting the ssd_inception_v2_coco_2018_01_28.tar.gz file")
        with tarfile.open(save_as) as tar:
            tar.extractall(path='pre_trained_model')
        os.remove('pre_trained_model/ssd_inception_v2_coco_2018_01_28.tar.gz')
    if os.path.exists('pre_trained_model/ssd_inception_v2_coco_2018_01_28'):
        os.rename('pre_trained_model/ssd_inception_v2_coco_2018_01_28', 'pre_trained_model/ssd_inception_v2_coco')


    # check training directory
    if not os.path.exists('training'):
        os.mkdir('training')
    if not os.path.exists('training/ssd_inception_v2_coco.config'):
        src_config_file = "../../models/research/object_detection/samples/configs/ssd_inception_v2_coco.config"
        dst_config_file = "training/ssd_inception_v2_coco.config"
        if not os.path.exists(src_config_file):
            print("错误：TensorFlow/models/research/object_detection/samples/configs/ssd_inception_v2_coco.config文件不存在。")
            return
        shutil.copy(src_config_file, dst_config_file)

    # copy train.py
    if not os.path.exists('train.py'):
        src_train_py = "../../models/research/object_detection/legacy/train.py"
        if not os.path.exists(src_train_py):
            print("错误：TensorFlow、models/research/object_detection/legacy/train.py文件不存在。")
            return
        shutil.copy(src_train_py, 'train.py')

    # check TensorFlow/scripts directory
    if not os.path.exists('../../scripts'):
        os.mkdir('../../scripts')
    if not os.path.exists('../../scripts/preprocessing'):
        os.mkdir('../../scripts/preprocessing')
    if not os.path.exists('../../scripts/preprocessing/xml_to_csv.py'):
        with open('../../scripts/preprocessing/xml_to_csv.py', 'w') as f:
            f.write('"""\nUsage:\n# Create train data:\npython xml_to_csv.py -i [PATH_TO_IMAGES_FOLDER]/train -o [PATH_TO_ANNOTATIONS_FOLDER]/train_labels.csv\n\n# Create test data:\npython xml_to_csv.py -i [PATH_TO_IMAGES_FOLDER]/test -o [PATH_TO_ANNOTATIONS_FOLDER]/test_labels.csv\n"""\n\nimport os\nimport glob\nimport pandas as pd\nimport argparse\nimport xml.etree.ElementTree as ET\n\n\ndef xml_to_csv(path):\n    """Iterates through all .xml files (generated by labelImg) in a given directory and combines them in a single Pandas datagrame.\n\n    Parameters:\n    ----------\n    path : {str}\n        The path containing the .xml files\n    Returns\n    -------\n    Pandas DataFrame\n        The produced dataframe\n    """\n\n    xml_list = []\n    for xml_file in glob.glob(path + \'/*.xml\'):\n        tree = ET.parse(xml_file)\n        root = tree.getroot()\n        for member in root.findall(\'object\'):\n            value = (root.find(\'filename\').text,\n                    int(root.find(\'size\')[0].text),\n                    int(root.find(\'size\')[1].text),\n                    member[0].text,\n                    int(member[4][0].text),\n                    int(member[4][1].text),\n                    int(member[4][2].text),\n                    int(member[4][3].text)\n                    )\n            xml_list.append(value)\n    column_name = [\'filename\', \'width\', \'height\',\n                \'class\', \'xmin\', \'ymin\', \'xmax\', \'ymax\']\n    xml_df = pd.DataFrame(xml_list, columns=column_name)\n    return xml_df\n\n\ndef main():\n    # Initiate argument parser\n    parser = argparse.ArgumentParser(\n        description="Sample TensorFlow XML-to-CSV converter")\n    parser.add_argument("-i",\n                        "--inputDir",\n                        help="Path to the folder where the input .xml files are stored",\n                        type=str)\n    parser.add_argument("-o",\n                        "--outputFile",\n                        help="Name of output .csv file (including path)", type=str)\n    args = parser.parse_args()\n\n    if(args.inputDir is None):\n        args.inputDir = os.getcwd()\n    if(args.outputFile is None):\n        args.outputFile = args.inputDir + "/labels.csv"\n\n    assert(os.path.isdir(args.inputDir))\n\n    xml_df = xml_to_csv(args.inputDir)\n    xml_df.to_csv(\n        args.outputFile, index=None)\n    print(\'Successfully converted xml to csv.\')\n\n\nif __name__ == \'__main__\':\n    main()')
    if not os.path.exists('../../scripts/preprocessing/generate_tfrecord.py'):
        with open('../../scripts/preprocessing/generate_tfrecord.py', 'w') as f:
            f.write('"""\nUsage:\n\n# Create train data:\npython generate_tfrecord.py --label=<LABEL> --csv_input=<PATH_TO_ANNOTATIONS_FOLDER>/train_labels.csv  --output_path=<PATH_TO_ANNOTATIONS_FOLDER>/train.record\n\n# Create test data:\npython generate_tfrecord.py --label=<LABEL> --csv_input=<PATH_TO_ANNOTATIONS_FOLDER>/test_labels.csv  --output_path=<PATH_TO_ANNOTATIONS_FOLDER>/test.record\n"""\n\nfrom __future__ import division\nfrom __future__ import print_function\nfrom __future__ import absolute_import\n\nimport os\nimport io\nimport pandas as pd\nimport tensorflow as tf\nimport sys\nsys.path.append("../../models/research")\n\nfrom PIL import Image\nfrom object_detection.utils import dataset_util\nfrom collections import namedtuple, OrderedDict\n\nflags = tf.app.flags\nflags.DEFINE_string(\'csv_input\', \'\', \'Path to the CSV input\')\nflags.DEFINE_string(\'output_path\', \'\', \'Path to output TFRecord\')\nflags.DEFINE_string(\'label\', \'\', \'Name of class label\')\n# if your image has more labels input them as\n# flags.DEFINE_string(\'label0\', \'\', \'Name of class[0] label\')\n# flags.DEFINE_string(\'label1\', \'\', \'Name of class[1] label\')\n# and so on.\nflags.DEFINE_string(\'img_path\', \'\', \'Path to images\')\nFLAGS = flags.FLAGS\n\n\n# TO-DO replace this with label map\n# for multiple labels add more else if statements\ndef class_text_to_int(row_label):\n    if row_label == FLAGS.label:  # \'ship\':\n        return 1\n    # comment upper if statement and uncomment these statements for multiple labelling\n    # if row_label == FLAGS.label0:\n    #   return 1\n    # elif row_label == FLAGS.label1:\n    #   return 0\n    else:\n        None\n\n\ndef split(df, group):\n    data = namedtuple(\'data\', [\'filename\', \'object\'])\n    gb = df.groupby(group)\n    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]\n\n\ndef create_tf_example(group, path):\n    with tf.gfile.GFile(os.path.join(path, \'{}\'.format(group.filename)), \'rb\') as fid:\n        encoded_jpg = fid.read()\n    encoded_jpg_io = io.BytesIO(encoded_jpg)\n    image = Image.open(encoded_jpg_io)\n    width, height = image.size\n\n    filename = group.filename.encode(\'utf8\')\n    image_format = b\'jpg\'\n    # check if the image format is matching with your images.\n    xmins = []\n    xmaxs = []\n    ymins = []\n    ymaxs = []\n    classes_text = []\n    classes = []\n\n    for index, row in group.object.iterrows():\n        xmins.append(row[\'xmin\'] / width)\n        xmaxs.append(row[\'xmax\'] / width)\n        ymins.append(row[\'ymin\'] / height)\n        ymaxs.append(row[\'ymax\'] / height)\n        classes_text.append(row[\'class\'].encode(\'utf8\'))\n        classes.append(class_text_to_int(row[\'class\']))\n\n    tf_example = tf.train.Example(features=tf.train.Features(feature={\n        \'image/height\': dataset_util.int64_feature(height),\n        \'image/width\': dataset_util.int64_feature(width),\n        \'image/filename\': dataset_util.bytes_feature(filename),\n        \'image/source_id\': dataset_util.bytes_feature(filename),\n        \'image/encoded\': dataset_util.bytes_feature(encoded_jpg),\n        \'image/format\': dataset_util.bytes_feature(image_format),\n        \'image/object/bbox/xmin\': dataset_util.float_list_feature(xmins),\n        \'image/object/bbox/xmax\': dataset_util.float_list_feature(xmaxs),\n        \'image/object/bbox/ymin\': dataset_util.float_list_feature(ymins),\n        \'image/object/bbox/ymax\': dataset_util.float_list_feature(ymaxs),\n        \'image/object/class/text\': dataset_util.bytes_list_feature(classes_text),\n        \'image/object/class/label\': dataset_util.int64_list_feature(classes),\n    }))\n    return tf_example\n\n\ndef main(_):\n    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)\n    path = os.path.join(os.getcwd(), FLAGS.img_path)\n    examples = pd.read_csv(FLAGS.csv_input)\n    grouped = split(examples, \'filename\')\n    for group in grouped:\n        tf_example = create_tf_example(group, path)\n        writer.write(tf_example.SerializeToString())\n\n    writer.close()\n    output_path = os.path.join(os.getcwd(), FLAGS.output_path)\n    print(\'Successfully created the TFRecords: {}\'.format(output_path))\n\n\nif __name__ == \'__main__\':\n    tf.app.run()')

    print("文件夹及文件检查完成。")
    return 'OK'


def get_labels() -> list :
    # get labels from tht train set.
    labels = []

    for xml_file in glob.glob('images/train/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            label = member[0].text
            if label not in labels:
                labels.append(label)

    for xml_file in glob.glob('images/eval/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            label = member[0].text
            if label not in labels:
                labels.append(label)

    return labels


def create_label_map_pbtxt(labels: list):
    with open('annotations/label_map.pbtxt', 'w') as f:
        i = 1
        for label in labels:
            f.write("item {\n\tid: %d\n\tname: '%s'\n}\n" %(i, label))
            i += 1
    print("label_map.pbtxt创建成功。")
    return


def modify_generate_tfrecord_py(number_of_labels: int):
    with open('../../scripts/preprocessing/generate_tfrecord.py', 'r') as f:
        content = f.readlines()

    # one label
    if number_of_labels == 1:
        pattern1 = re.compile(r"# *flags\.DEFINE_string\('label', '.*', 'Name of class label'\)")
        pattern2 = re.compile(r"flags\.DEFINE_string\('label[0-9]+', '.*', 'Name of class\[[0-9]+] label'\)")
        pattern3 = re.compile(r"    #? *if row_label == FLAGS\.label:")
        pattern4 = re.compile(r"    #? *elif row_label == FLAGS\.label[0-9]+:")

        i = 0
        row_label = 1
        while i < len(content):
            if re.match(pattern1, content[i]) != None:
                content[i] = "flags.DEFINE_string('label', '', 'Name of class label')\n"
            if re.match(pattern2, content[i]) != None:
                content[i] = "# " + content[i]
            if re.match(pattern3, content[i]) != None:
                content[i] = "    if row_label == FLAGS.label:\n"
                i += 1
                content[i] = "        return 1\n"
            if content[i] == "    if row_label == FLAGS.label0:\n":
                content[i] = "    # if row_label == FLAGS.label0:\n"
                i += 1
                content[i] = "    #    return 1\n"
            if re.match(pattern4, content[i]) != None:
                content[i] = "    # elif row_label == FLAGS.label%d:\n" %row_label
                i += 1
                row_label += 1
                content[i] = "    #    return %d\n" %(row_label)
            i += 1

    # two labels
    elif number_of_labels == 2:
        pattern1 = re.compile(r"flags\.DEFINE_string\('label', '.*', 'Name of class label'\)")
        pattern2 = re.compile(r"#? *flags\.DEFINE_string\('label[0-9]+', '.*', 'Name of class\[[0-9]+] label'\)")
        pattern3 = re.compile(r"    if row_label == FLAGS\.label:")
        pattern4 = re.compile(r"    #? *if row_label == FLAGS\.label0:")
        pattern5 = re.compile(r"    #? *elif row_label == FLAGS\.label[0-9]+:")

        i = 0
        label_num = 0
        row_label = 1
        while i < len(content):
            if re.match(pattern1, content[i]) != None:
                content[i] = "# " + content[i]
            if re.match(pattern2, content[i]) != None:
                if label_num < 2:
                    content[i] = "flags.DEFINE_string('label%d', '', 'Name of class[%d] label')\n" %(label_num, label_num)
                else:
                    content[i] = "# flags.DEFINE_string('label%d', '', 'Name of class[%d] label')\n" %(label_num, label_num)
                label_num += 1
            if re.match(pattern3, content[i]) != None:
                content[i] = "    # if row_label == FLAGS.label:\n"
                i += 1
                content[i] = "    #    return 1\n"
            if re.match(pattern4, content[i]) != None:
                content[i] = "    if row_label == FLAGS.label0:\n"
                i += 1
                content[i] = "        return 1\n"
            if re.match(pattern5, content[i]) != None:
                if row_label < 2:
                    content[i] = "    elif row_label == FLAGS.label%d:\n" %row_label
                    i += 1
                    row_label += 1
                    content[i] = "        return %d\n" %row_label
                else:
                    content[i] = "    # elif row_label == FLAGS.label%d:\n" %row_label
                    i += 1
                    row_label += 1
                    content[i] = "    #    return %d\n" %row_label
            i += 1

    # more than two labels
    else:
        pattern1 = re.compile(r"flags\.DEFINE_string\('label', '.*', 'Name of class label'\)")
        pattern2 = re.compile(r"#? *flags\.DEFINE_string\('label[0-9]+', '.*', 'Name of class\[[0-9]+] label'\)")
        pattern3 = re.compile(r"    if row_label == FLAGS\.label:")
        pattern4 = re.compile(r"    #? *if row_label == FLAGS\.label0:")
        pattern5 = re.compile(r"    #? *elif row_label == FLAGS\.label[0-9]+:")

        i = 0
        label_num = 0
        row_label = 1
        while i < len(content):
            if re.match(pattern1, content[i]) != None:
                content[i] = "# " + content[i]
            if re.match(pattern2, content[i]) != None:
                if label_num < number_of_labels:
                    content[i] = "flags.DEFINE_string('label%d', '', 'Name of class[%d] label')\n" %(label_num, label_num)
                else:
                    content[i] = "# flags.DEFINE_string('label%d', '', 'Name of class[%d] label')\n" %(label_num, label_num)
                label_num += 1
            if re.match(pattern3, content[i]) != None:
                content[i] = "    # if row_label == FLAGS.label:\n"
                i += 1
                content[i] = "    #    return 1\n"
            if re.match(pattern4, content[i]) != None:
                content[i] = "    if row_label == FLAGS.label0:\n"
                i += 1
                content[i] = "        return 1\n"
            if re.match(pattern5, content[i]) != None:
                if row_label < number_of_labels:
                    content[i] = "    elif row_label == FLAGS.label%d:\n" %row_label
                    i += 1
                    row_label += 1
                    content[i] = "        return %d\n" %row_label
                else:
                    content[i] = "    # elif row_label == FLAGS.label%d:\n" %row_label
                    i += 1
                    row_label += 1
                    content[i] = "    #    return %d\n" %row_label
            i += 1

        if label_num < number_of_labels:

            last_line = "flags.DEFINE_string('label%d', '', 'Name of class[%d] label')\n" %(label_num-1, label_num-1)
            line_insert_num = content.index(last_line)
            add_content = ""
            while label_num < number_of_labels:
                add_content += "flags.DEFINE_string('label%d', '', 'Name of class[%d] label')\n" %(label_num, label_num)
                label_num += 1
            content[line_insert_num] += add_content

            last_line = "        return %d\n" %(row_label)
            line_insert_num = content.index(last_line)
            add_content = ""
            while row_label < number_of_labels:
                add_content += "    elif row_label == FLAGS.label%d:\n        return %d\n" %(row_label, row_label+1)
                row_label += 1
            content[line_insert_num] += add_content

    # rewrite generate_tfrecord.py
    with open('../../scripts/preprocessing/generate_tfrecord.py', 'w') as f:
        for line in content:
            f.write(line)
    print("generate_tfrecord.py文件修改完成。")
    return


def xml_to_csv_to_tfrecord(labels: list):
    # check pandas 
    try:
        import pandas
    except:
        print("安装pandas模块。")
        with os.popen(r"pip install pandas", "r") as f:
            install_information = f.read()
        print(install_information)
        if "Successfully installed pandas" not in install_information:
            print("错误：pandas安装失败。")
            return

    # xml to csv
    transform_script = "../../scripts/preprocessing/xml_to_csv.py"
    input_dir = "images/train"
    output_dir = "annotations/train_labels.csv"
    command = "python " + transform_script + " -i " + input_dir + " -o " + output_dir
    with os.popen(command, 'r') as f:
        transform_information = f.read()
    if "Successfully converted xml to csv." not in transform_information:
        print("错误：训练数据集.xml转.csv失败。")
        return
    print("训练数据集.xml转.csv完成。")
    input_dir = "images/eval"
    output_dir = "annotations/eval_labels.csv"
    command = "python " + transform_script + " -i " + input_dir + " -o " + output_dir
    with os.popen(command, 'r') as f:
        transform_information = f.read()
    if "Successfully converted xml to csv." not in transform_information:
        print("错误：评估数据集.xml转.csv失败。")
        return
    print("评估数据集.xml转.csv完成。")

    # csv to record
    transform_script = "../../scripts/preprocessing/generate_tfrecord.py"
    label_information = " "
    if len(labels) == 1:
        label_information += "--label=%s" %labels[0]
    else:
        i = 0
        for label in labels:
            label_information += "--label%d=%s " %(i, label)
            i += 1
    input_dir = "annotations/train_labels.csv"
    output_dir = "annotations/train.record"
    images_path = "images/train"
    command = "python " + transform_script + label_information + " --csv_input=" + input_dir + " --output_path=" + output_dir + " --img_path=" + images_path
    with os.popen(command, 'r') as f:
        transform_information = f.read()
    if "Successfully created the TFRecords" not in transform_information:
        print("错误：train_labels.csv转train.record失败。")
        return
    print("train_labels.csv转train.record完成。")
    input_dir = "annotations/eval_labels.csv"
    output_dir = "annotations/eval.record"
    images_path = "images/eval"
    command = "python " + transform_script + label_information + " --csv_input=" + input_dir + " --output_path=" + output_dir + " --img_path=" + images_path
    with os.popen(command, 'r') as f:
        transform_information = f.read()
    if "Successfully created the TFRecords" not in transform_information:
        print("错误：eval_labels.csv转eval.record失败。")
        return
    print("eval_labels.csv转eval.record完成。")

    return


def modify_ssd_inception_v2_coco_config(number_of_labels: int, number_of_steps: int, batch_size: int):
    with open("training/ssd_inception_v2_coco.config", "r") as f:
        content = f.readlines()

    pattern1 = re.compile(r" *num_classes:")
    pattern2 = re.compile(r" *type: '.*'")
    pattern3 = re.compile(r" *batch_size:")
    pattern4 = re.compile(r" *fine_tune_checkpoint:")
    pattern5 = re.compile(r" *num_steps:")
    pattern6 = re.compile(r" *input_path:")
    pattern7 = re.compile(r" *label_map_path:")
    pattern8 = re.compile(r" *num_examples:")

    i = 0
    while i < len(content):
        if re.match(pattern1, content[i]) is not None:
            content[i] = "    num_classes: %d\n"  %number_of_labels
        if re.match(pattern2, content[i]) is not None:
            content[i] = "      type: 'ssd_inception_v2'\n"
        if re.match(pattern3, content[i]) is not None:
            content[i] = "  batch_size: %d\n" %batch_size
        if re.match(pattern4, content[i]) is not None:
            content[i] = "  fine_tune_checkpoint: \"pre_trained_model/ssd_inception_v2_coco/model.ckpt\"\n"
        if re.match(pattern5, content[i]) is not None:
            content[i] = "  num_steps: %d\n" %number_of_steps
        if (re.match(pattern6, content[i]) is not None) and ("train_input_reader:" in content[i-2]):
            content[i] = "    input_path: \"annotations/train.record\"\n"
        if (re.match(pattern6, content[i]) is not None) and ("eval_input_reader:" in content[i-2]):
            content[i] = "    input_path: \"annotations/eval.record\"\n"
        if re.match(pattern7, content[i]) is not None:
            content[i] = "  label_map_path: \"annotations/label_map.pbtxt\"\n"
        if re.match(pattern8, content[i]) is not None:
            content[i] = "  num_examples: %d\n" %len(glob.glob("images/eval/*.xml"))
        i += 1

    with open("training/ssd_inception_v2_coco.config", "w") as f:
        for line in content:
            f.write(line)
    print("ssd_inception_v2_coco.config文件修改完成。")
    return


def main():
    # 1. check directories and documents. 
    if check_directories_and_documents() is not 'OK':
        return

    # 2. get labels list and create the label mapping file.
    labels = get_labels()
    create_label_map_pbtxt(labels)

    # 3. modify generate_tfrecord.py and create TensorFlow records.
    modify_generate_tfrecord_py(len(labels))
    try:
        import tensorflow
    except:
        print("错误：没有tensorflow模块，请激活tensorflow_cpu或tensorflow_gpu虚拟环境。")
        return
    xml_to_csv_to_tfrecord(labels)

    # 4. modify ssd_inception_v2_coco.config
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",
                        help="number of steps",
                        type=int)
    parser.add_argument("--batch_size",
                        help="batch size",
                        type=int)
    args = parser.parse_args()

    if args.steps is None:
        args.steps = 50000
        print("默认训练步数为1000步，自定义训练步数请使用--steps参数，并重新执行脚本。")
    if args.batch_size is None:
        args.batch_size = 4
        print("默认批处理大小为6，自定义请使用--batch_size参数进行设置，值越大所需内存越大，并重新执行脚本。")

    modify_ssd_inception_v2_coco_config(len(labels), args.steps, args.batch_size)

    # 5. start training
    command = "python train.py --logtostderr --train_dir=training2 --pipeline_config_path=training2/ssd_inception_v2_coco.config"
    os.system(command)



if __name__ == '__main__':
    main()