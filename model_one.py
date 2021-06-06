import os
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import Xception


def train_datagen(train_dir, validation_dir):
    # 图像数据增强
    train_datagen = ImageDataGenerator(
        # 对图像进行放缩
        rescale=1. / 255,
        # 表示图像随机旋转的角度范围（0~180）
        rotation_range=40,
        # 图像在水平或垂直方向上平移的范围
        # 相对于总宽度或总高度的比例
        width_shift_range=0.2,
        height_shift_range=0.2,
        # 随机切换的变换角度
        shear_range=0.2,
        # 图像随机缩放的范围
        zoom_range=0.2,
        # 随机将一半图像水平翻转
        horizontal_flip=True,
        # 填充新创建像素的方法
        fill_mode='nearest'
    )
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # 将训练图像转化为张量生成训练数据
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        # 重置图像大小244*244
        target_size=(244, 244),
        # 一次处理的图像个数
        batch_size=10,
        class_mode='categorical'
    )

    # 将验证图像转化为张量生成训练数据
    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(244, 244),
        batch_size=10,
        class_mode='categorical'
    )
    return train_generator, validation_generator


def Net_model(train_generator, validation_generator):
    # 引入Xception模型
    conv_base = Xception(
        weights='imagenet',
        # 不引入最后一层
        include_top=False,
        # 训练图像的大小
        input_shape=(244, 244, 3),
    )
    model_top = models.Sequential()
    model_top.add(layers.Flatten())
    model_top.add(layers.Dense(256, activation='relu'))
    model_top.add(layers.Dropout(0.5))
    model_top.add(layers.Dense(4, activation='softmax'))

    model = models.Sequential()
    model.add(conv_base)
    model.add(model_top)

    model.compile(
        optimizer=optimizers.RMSprop(lr=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit_generator(
        generator=train_generator,
        epochs=50,
        steps_per_epoch=len(train_generator) // 10,
        validation_data=validation_generator,
        validation_steps=len(validation_generator) // 10
    )

    model.save('tar_model2.h5')
    loss, acc = model.evaluate_generator(validation_generator)
    print("loss=%.2f,acc=%.2f" % (loss, acc * 100))


if __name__ == "__main__":
    base_dir = "G:\\Python\\imageData\\garbage_classify"
    train_dir = os.path.join(base_dir, "train_two")
    validation_dir = os.path.join(base_dir, "validation_two")
    train, validation = train_datagen(train_dir, validation_dir)
    Net_model(train, validation)
