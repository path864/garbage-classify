import cv2 as cv
import numpy as np
import tensorflow as tf
import time

labels = ["harmful", "kitchen", "other", "recycle"]
datas = ['a', 'b', 'd', 'c']

interpreter = tf.lite.Interpreter(model_path='m3.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

vidio = cv.VideoCapture(0)
count = 0
while True:
    # 读取当前的帧
    ret, frame = vidio.read()
    # 对图像进行翻转
    frame = cv.flip(frame, 1)
    # 重置图像大小
    frame = cv.resize(frame, (244, 244))
    # 图像处理
    img = np.expand_dims(frame, axis=0)
    img = img / 255.
    input_data = img.astype(np.float32)

    index = input_details[0]['index']
    interpreter.set_tensor(index, input_data)
    if count % 5 == 0:
        count = 0
        # 开始预测
        interpreter.invoke()
        predict = interpreter.get_tensor(output_details[0]['index'])
        pre_index = np.argmax(predict[0])
        predict_copy = np.copy(predict)
        predict_copy = np.argsort(predict_copy[0])[::-1]
        if predict[0][predict_copy[0]] > 0.95:
            conculude = labels[pre_index]
            print(conculude)
            print(predict[0][predict_copy[0]])

    cv.imshow("src", frame)
    count += 1
    if cv.waitKey(50) == 27:
        break

vidio.release()
cv.destroyAllWindows()
