from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify


app = Flask(__name__)

translate = {
    'changfen': '肠粉',
    'luobogao': '萝卜糕',
    'xiajiao': '虾饺',
    'danta': '蛋挞',
    'guilinggao': '龟苓膏',
    'zhengfengzhua': '蒸凤爪',
    'shaomai': '烧麦',
    'yuntun': '云吞',
    'ganchaoniuhe': '干炒牛河',
    'baiqieji': '白切鸡',
}

graph = None    # 计算图
input_op = None
output_op = None
labels = []


def load_model(model_file):
    """载入模型
    """
    graph = tf.Graph()
    graph_def = tf.GraphDef()
    with open(model_file, 'rb') as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)
    return graph


def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label


def read_tensor_from_image(file_name,
                           input_height=224,
                           input_width=224,
                           input_mean=0,
                           input_std=255):
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(file_name, input_name)
    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(
            file_reader, channels=3, name="png_reader")
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(
            tf.image.decode_gif(file_reader, name="gif_reader"))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
    else:
        image_reader = tf.image.decode_jpeg(
            file_reader, channels=3, name="jpeg_reader")
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)
    return result


@app.route('/food-classify/<path:uri>')
def classify(uri):
    """图像分类

    图像分类 api，返回 json 格式的置信度最高的分类，例如：
    {
        "category": "萝卜糕",
        "confidence": 0.7912
    }
    """
    ts = read_tensor_from_image(uri)
    with tf.Session(graph=graph) as sess:
        results = sess.run(output_op.outputs[0], {
            input_op.outputs[0]: ts
        })
    results = np.squeeze(results)
    print(results)
    top1 = results.argsort()[-1]
    return jsonify(category=translate[labels[top1]], confidence=float(results[top1]))


def init():
    global graph
    global input_op
    global output_op
    global labels

    graph = load_model('./output/retrain.pb')
    input_op = graph.get_operation_by_name('import/Placeholder')
    output_op = graph.get_operation_by_name('import/final_result')
    labels = load_labels('./output/output_labels.txt')


if __name__ == '__main__':
    init()
    app.run(debug=False, port=8089)
