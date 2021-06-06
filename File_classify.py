import os, shutil
from glob import glob
import random

base_dir = "G:\\Python\\imageData\\garbage_classify"
# os.mkdir(base_dir)
# 创建train目录
train_dir = os.path.join(base_dir, "train")
# os.mkdir(train_dir)

# 创建validation目录
validation_dir = os.path.join(base_dir, "validation")
# os.mkdir(validation_dir)

# 创建test目录
test_dir = os.path.join(base_dir, "test")
# os.mkdir(test_dir)

# 创建train四类目录
train_other_dir = os.path.join(train_dir, "Other")
# os.mkdir(train_other_dir)
train_kitchen_dir = os.path.join(train_dir, "kitchen")
# os.mkdir(train_kitchen_dir)
train_recycle_dir = os.path.join(train_dir, "recycle")
# os.mkdir(train_recycle_dir)
train_harmful_dir = os.path.join(train_dir, "harmful")
# os.mkdir(train_harmful_dir)

# 创建validation四类目录
validation_other_dir = os.path.join(validation_dir, "Other")
# os.mkdir(validation_other_dir)
validation_kitchen_dir = os.path.join(validation_dir, "kitchen")
# os.mkdir(validation_kitchen_dir)
validation_recycle_dir = os.path.join(validation_dir, "recycle")
# os.mkdir(validation_recycle_dir)
validation_harmful_dir = os.path.join(validation_dir, "harmful")
# os.mkdir(validation_harmful_dir)


# 将所有的图像全部放到img_paths中
def gendata(train_data_dir):
    # 返回所有匹配的.txt文件
    # 获得train_data_dir目录下的所有.txt文件
    label_files = glob(os.path.join(train_data_dir, '*.txt'))
    # 将标签打乱顺序随机获得验证数据
    random.shuffle(label_files)
    img_paths = []
    labels = []
    # 对label_files进行编码生成字典
    for index, file_path in enumerate(label_files):
        # 打开路径下的文件
        with open(file_path, 'r') as f:
            line = f.readline()
        # 获得逗号分割的字符
        line_split = line.strip().split(',')
        # 没有图像名称或标签
        if len(line_split) != 2:
            print("%s contain error lable" % os.path.basename(file_path))
            continue
        img_name = line_split[0]
        label = int(line_split[1])
        img_paths.append(os.path.join(train_data_dir, img_name))
        labels.append(label)

    return img_paths, labels


# 将训练图像复制到指定目录
def moveFile(images_name, path, tarDir):
    src = path
    dst = os.path.join(tarDir, images_name)
    shutil.copyfile(src, dst)


def File_image(image_paths, labels):
    i = 0
    # k = 0
    k = 19736
    for label in labels:
        if label in range(0, 6):
            if k % 5 == 0:
                Dir = validation_other_dir
                validation_name = str(label) + " " + "other{}.jpg".format(k)
                moveFile(validation_name, image_paths[i], Dir)
                k += 1
                i += 1
                continue
            Dir = train_other_dir
            train_name = str(label) + " " + "other{}.jpg".format(k)
            moveFile(train_name, image_paths[i], Dir)

        if label in range(6, 14):
            if k % 5 == 0:
                Dir = validation_kitchen_dir
                validation_name = str(label) + " " + "kitchen{}.jpg".format(k)
                moveFile(validation_name, image_paths[i], Dir)
                k += 1
                i += 1
                continue
            Dir = train_kitchen_dir
            train_name = str(label) + " " + "kitchen{}.jpg".format(k)
            moveFile(train_name, image_paths[i], Dir)

        if label in range(14, 37):
            if k % 5 == 0:
                Dir = validation_recycle_dir
                validation_name = str(label) + " " + "recyle{}.jpg".format(k)
                moveFile(validation_name, image_paths[i], Dir)
                k += 1
                i += 1
                continue
            Dir = train_recycle_dir
            train_name = str(label) + " " + "recyle{}.jpg".format(k)
            moveFile(train_name, image_paths[i], Dir)

        if label in range(37, 40):
            if k % 5 == 0:
                Dir = validation_harmful_dir
                validation_name = str(label) + " " + "harmful{}.jpg".format(k).format(label)
                moveFile(validation_name, image_paths[i], Dir)
                k += 1
                i += 1
                continue
            Dir = train_harmful_dir
            train_name = str(label) + " " + "harmful{}.jpg".format(k)
            moveFile(train_name, image_paths[i], Dir)
        k += 1
        i += 1


if __name__ == '__main__':
    # base_dir = "G:\\Python\\ipython\\garbage_classify_v2\\train_data_v2"
    base_dir = "G:\\BaiduNetdiskDownload\\垃圾分类\\新建文件夹\\garbage_classify_et\\train_data"
    img_path, labels = gendata(base_dir)
    File_image(img_path, labels)
