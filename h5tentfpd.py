"""
将keras的.h5的模型文件，转换成TensorFlow的pb文件
"""
# ==========================================================

from keras.models import load_model
import tensorflow as tf
import os
from keras import backend


def h5_to_pb(h5_model, output_dir, model_name, out_prefix="output_", log_tensorboard=True):
    """.h5模型文件转换成pb模型文件
    Argument:
        h5_model: str
            .h5模型文件
        output_dir: str
            pb模型文件保存路径
        model_name: str
            pb模型文件名称
        out_prefix: str
            根据训练，需要修改
        log_tensorboard: bool
            是否生成日志文件
    Return:
        pb模型文件
    """
    if os.path.exists(output_dir) == False:
        os.mkdir(output_dir)
    out_nodes = []
    for i in range(len(h5_model.outputs)):
        out_nodes.append(out_prefix + str(i + 1))
        tf.identity(h5_model.output[i], out_prefix + str(i + 1))
    sess = backend.get_session()

    from tensorflow.python.framework import graph_util, graph_io
    # 写入pb模型文件
    init_graph = sess.graph.as_graph_def()
    main_graph = graph_util.convert_variables_to_constants(sess, init_graph, out_nodes)
    graph_io.write_graph(main_graph, output_dir, name=model_name, as_text=False)
    # 输出日志文件
    if log_tensorboard:
        from tensorflow.python.tools import import_pb_to_tensorboard
        import_pb_to_tensorboard.import_to_tensorboard(os.path.join(output_dir, model_name), output_dir)


if __name__ == '__main__':
    #  .h模型文件路径参数
    # input_path = 'G:\\Python\\ipython\\Demo'
    # weight_file = 'tar_model.h5'
    # weight_file_path = os.path.join(input_path, weight_file)
    # output_graph_name = weight_file[:-3] + '.pb'
    #
    #
    # #  加载模型
    # h5_model = load_model(weight_file_path)
    # h5_to_pb(h5_model, output_dir=input_path, model_name=output_graph_name)
    # print('Finished')

    converter = tf.lite.TFLiteConverter.from_keras_model_file('tar_model2.h5')
    tflite_model = converter.convert()
    open("m2.tflite", "wb").write(tflite_model)

