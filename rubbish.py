# keep calm and carry on
import time
from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk

import numpy as np
import tensorflow as tf
import cv2 as cv
import threading


def run_counter(digit1, digit2, digit3, digit4):
    def counting():
        digit1.config(text=str(count1))
        digit2.config(text=str(count2))
        digit3.config(text=str(count3))
        digit4.config(text=str(count4))
        digit4.after(1000, counting)

    counting()


def startDetect():
    win = Toplevel(master1)
    win.title("垃圾检测")
    win.geometry("600x200+0+200")
    Label(win, text="有害垃圾", font="微软雅黑 14").grid(row=1, column=0, padx=10, pady=5)
    digit1 = Label(win, bg='yellow', fg="blue", font="微软雅黑 14", height=3, width=10)
    digit1.grid(row=2, column=0, padx=10, pady=5)
    Label(win, text="厨余垃圾", font="微软雅黑 14").grid(row=1, column=1, padx=10, pady=5)
    digit2 = Label(win, bg='yellow', fg="blue", font="微软雅黑 14", height=3, width=10)
    digit2.grid(row=2, column=1, padx=10, pady=5)
    Label(win, text="可回收垃圾", font="微软雅黑 14").grid(row=1, column=2, padx=10, pady=5)
    digit3 = Label(win, bg='yellow', fg="blue", font="微软雅黑 14", height=3, width=10)
    digit3.grid(row=2, column=2, padx=10, pady=5)
    Label(win, text="其他垃圾", font="微软雅黑 14").grid(row=1, column=3, padx=10, pady=5)
    digit4 = Label(win, bg='yellow', fg="blue", font="微软雅黑 14", height=3, width=10)
    digit4.grid(row=2, column=3, padx=10, pady=5)
    butt = Button(win, text="满载检测", font="微软雅黑 14", command=fullDetect)
    butt.grid(row=3, column=0, columnspan=4, padx=10, pady=5)
    run_counter(digit1, digit2, digit3, digit4)
    win.mainloop()


def detect():
    labels = ["harmful", "kitchen", "other", "recycle"]
    datas = ['a', 'b', 'd', 'c']
    global count, count1, count2, count3, count4
    interpreter = tf.lite.Interpreter(model_path='converted_model.tflite')
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    startDetect()
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
        if count % 25 == 0:
            count = 0
            # 开始预测
            interpreter.invoke()
            predict = interpreter.get_tensor(output_details[0]['index'])
            pre_index = np.argmax(predict[0])
            predict_copy = np.copy(predict)
            predict_copy = np.argsort(predict_copy[0])[::-1]
            if predict[0][predict_copy[0]] > 0.95:
                conculude = labels[pre_index]
                data = datas[pre_index]
                if data == 'a':
                    count1 += 1
                elif data == 'b':
                    count2 += 1
                elif data == 'c':
                    count3 += 1
                else:
                    count4 += 1
                time.sleep()
                print(conculude)
                print(predict[0][predict_copy[0]])

        cv.imshow("src", frame)
        count += 1
        if cv.waitKey(50) == 27:
            break

    vidio.release()
    cv.destroyAllWindows()


def Exit():
    response = messagebox.askokcancel("exit", "请问您真的要退出么？")
    if response:
        master.destroy()


def play():
    cap = cv.VideoCapture("rubish.mp4")
    while cap.isOpened():
        ret, frame = cap.read()
        cv.imshow("video", frame)
        c = cv.waitKey(25)

        if c == 27:  # ESC键
            break
    cap.release()
    cv.destroyAllWindows()


def fullDetect():
    pass


def ui1():
    global master
    master = Tk()

    master.title("欢迎")

    master.geometry("450x400+800+200")

    canvas = Canvas(master, height=130, width=440)
    image3 = PhotoImage(file="welcome.gif")
    canvas.create_image(0, 0, anchor='nw', image=image3)
    canvas.grid(row=0, column=0, columnspan=2)
    Label(text="亲爱管理员\n"
               "垃圾检测\n"
               "请选择你的操作：", font="微软雅黑 14", justify=LEFT).grid(row=1, column=0, columnspan=2, sticky='w')
    button = Button(master, text="视频循环播放", font="微软雅黑 14", relief="solid", command=play)
    button.grid(sticky='w', row=3, column=0, padx=10, pady=20)
    Button(master, text="垃圾检测", font="微软雅黑 14", relief="solid", command=detect).grid(sticky='e', row=3, column=1,
                                                                                     padx=10,
                                                                                     pady=20)
    Button(master, text="满载检测", font="微软雅黑 14", relief="solid", command=fullDetect).grid(row=4, column=0, columnspan=2)

    master.mainloop()


def ui2():
    global master1
    master1 = Tk()
    master1.title("垃圾检测")
    master1.geometry("250x100+30+20")
    Button(master1, text="查看垃圾检测情况", font="微软雅黑 14", relief="solid", command=startDetect).pack(pady=10)
    master1.mainloop()


t1 = threading.Thread(target=ui1)
t2 = threading.Thread(target=ui2)

t1.start()
t2.start()
