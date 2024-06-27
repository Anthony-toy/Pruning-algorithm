import cv2
import glob
import numpy as np
import os

# 指定图像所在的文件夹
input_folder = 'D:\FireFit\Flame\Training\Training/1/*.jpg'

# 指定输出结果的文件夹
# output_folder = 'E:/NUAA_light/train/feak/'

# 指定图像分块大小
block_size = 32

# 遍历指定文件夹下的所有图像文件
for filename in glob.glob(input_folder ):
    img = cv2.imread(filename)
    # 载入图像
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    # 求取图像全局平均亮度
    avg_brightness = np.mean(img)

    # 将图像分成小块，并求取每个小块的平均亮度
    block_shape = (block_size, block_size)
    block_means = np.zeros((img.shape[0] // block_size, img.shape[1] // block_size))
    for i in range(block_means.shape[0]):
        for j in range(block_means.shape[1]):
            block = img[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            block_means[i, j] = np.mean(block)

    # 将子块平均亮度矩阵中每个值都减去全局平均亮度，获得子块亮度差值矩阵
    block_diffs = block_means - avg_brightness

    # 通过插值运算扩展子块亮度差值矩阵到与原图像相同大小，获得全图像亮度差值矩阵
    diff_map = cv2.resize(block_diffs, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    # a = np.repeat(diff_map, 3, 2)

    # 将原始图像各像素亮度值各自减去全图像亮度差值矩阵中对应的数值
    enhanced_img = img - diff_map

    # 根据原图像中最低和最高亮度来调节每个子块像素的亮度使之符合整个亮度范围
    enhanced_img = np.clip(enhanced_img, 0, 255)
    # image_np=cv2.cvtColor(enhanced_img ,cv2.COLOR_GRAY2BGR)


    # 输出结果
    # output_filename = output_folder + filename.split('/')[-1]
    cv2.imwrite(os.path.join(
        r'D:\FireFit\Flame\Training\Training/2',
        os.path.basename(filename)), enhanced_img)
    # print(enhanced_img.shape)