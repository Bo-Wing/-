import glob
import random
import cv2
import numpy as np
import os


def randint(xx, yy):
    return random.randint(xx, yy)


def randfloat(xx, yy):
    return random.uniform(xx, yy)


def generate_gaussian_target(size, brightness):
    """
    生成一个高斯分布的小目标（圆形），并且进行高斯模糊处理，形成一个自然的红外小目标。

    size: 目标的大小（即圆形的直径）
    brightness: 目标的亮度
    """
    # 创建一个空白图像（黑色背景）
    target = np.zeros((size, size), dtype=np.uint8)

    # 目标的中心
    center = (size // 2, size // 2)

    # 生成高斯分布图像
    for y in range(size):
        for x in range(size):
            distance = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
            # 根据距离计算高斯值
            target[y, x] = int(brightness * np.exp(-distance ** 2 / (2 * (size / 4) ** 2)))

    # 对目标进行高斯模糊处理，使得边缘更加柔和
    target = cv2.GaussianBlur(target, (5, 5), 0)

    # 转换为三通道（BGR），以便可以与彩色图像叠加
    target_bgr = cv2.cvtColor(target, cv2.COLOR_GRAY2BGR)

    return target_bgr


# 初始参数设定
input_folder = 'data2'  # 输入图像文件夹路径
output_video_path = 'output_video_with_targets.avi'  # 输出视频路径
input_folder = os.path.abspath(input_folder)
print(os.path.join(input_folder, "*.bmp"))

image_files = sorted(glob.glob(os.path.join(input_folder, "*.bmp")),
                     key=lambda x: int(os.path.basename(x).split('.')[0]))  # 输入数据集的图像格式

# 设置视频参数
fps = 30  # 帧率
frame_width, frame_height = 256, 256  # 数据集图像的宽高

fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))


# 设置小目标参数
target_num = 10
targets = []
for idx in range(target_num):
    targets.append({"pos": (randint(0, 256), randint(0, 256)), "v": (randfloat(-1,1), randfloat(-1,1)), "size": 1, "brightness": randint(150,255), "a": (randfloat(-1,1), randfloat(-1,1))})

# 遍历图像文件并处理
for frame_idx, image_file in enumerate(image_files):
    # 加载当前图像帧
    frame = cv2.imread(os.path.join(input_folder, image_file))
    t = frame_idx / fps

    for target in targets:
        # 计算小目标的当前速度
        v_x = target["v"][0] + target["a"][0] * t
        v_y = target["v"][1] + target["a"][1] * t

        # 计算小目标当前的位置
        xpos = int(target["pos"][0] + target["v"][0] * t + 0.5 * target["a"][0] * t * t)
        ypos = int(target["pos"][1] + target["v"][1] * t + 0.5 * target["a"][1] * t * t)

        # 检查小目标是否在图像范围内
        if 0 <= xpos < frame_width and 0 <= ypos < frame_height:
            small_target = generate_gaussian_target(target["size"], target["brightness"])

            # 将小目标添加到帧
            frame[ypos:ypos + target["size"], xpos:xpos + target["size"]] = cv2.addWeighted(
                frame[ypos:ypos + target["size"], xpos:xpos + target["size"]], 1, small_target, 0.5, 0)

    # 处理完一帧后，写入视频文件
    video_writer.write(frame)

# 释放视频写入对象
video_writer.release()
cv2.destroyAllWindows()
