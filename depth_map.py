import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

# frame = file name of .png file
def createMap(rel_path, frame):
    code_dir = os.path.dirname(__file__)

    left_path = os.path.join(code_dir, (rel_path+"image_2/"+frame))
    right_path = os.path.join(code_dir, (rel_path+"image_3/"+frame))

    imgL = cv2.imread(left_path, 0)
    imgR = cv2.imread(right_path, 0)

    stereo = cv2.StereoBM_create(numDisparities=64, blockSize=9)
    disparity = stereo.compute(imgL, imgR)
    return disparity
    

if __name__ == '__main__':
    rel_path = "KITTI/data_scene_flow/testing/"

    disparity = createMap(rel_path, "000183_11.png")
    plt.imshow(disparity, 'gray')
    plt.show()