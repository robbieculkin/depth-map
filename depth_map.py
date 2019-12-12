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


def createSmoothMap(rel_path, frame):
    code_dir = os.path.dirname(__file__)

    left_path = os.path.join(code_dir, (rel_path+"image_2/"+frame))
    right_path = os.path.join(code_dir, (rel_path+"image_3/"+frame))

    imgL = cv2.imread(left_path, 0)
    imgR = cv2.imread(right_path, 0)

    WINDOW_SIZE = 3

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=64,
        blockSize=5,
        P1=8*3*WINDOW_SIZE**2,
        P2=32*3*WINDOW_SIZE**2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY\
    )

    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(8000)
    wls_filter.setSigmaColor(1.2)

    disp_l = np.int16(left_matcher.compute(imgL,imgR))
    disp_r = np.int16(right_matcher.compute(imgR, imgL))
    
    filtered_img = wls_filter.filter(disp_l,imgL,None,disp_r)
    filtered_img = cv2.normalize(src=filtered_img,dst=filtered_img,beta=0,alpha=255,norm_type=cv2.NORM_MINMAX)


    return filtered_img


    

if __name__ == '__main__':
    rel_path = "KITTI/data_scene_flow/testing/"

    disparity = createMap(rel_path, "000015_10.png")
    plt.imshow(disparity, 'gray')
    plt.show()