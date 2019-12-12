# fit_plane.py

import numpy as np
from matplotlib import pyplot as plt
import depth_map
from sklearn.linear_model import RANSACRegressor as RR
from PIL import Image
import cv2


def ransac(v,d):

	ransac_v = RR()
	ransac_v.fit(v,d)
	inlier_mask = ransac_v.inlier_mask_
	outlier_mask = np.logical_not(inlier_mask)

	line_v = np.arange(v.min(), v.max())[:, np.newaxis]
	line_d = ransac_v.predict(line_v)

	return line_v,line_d


def mapGround(path_to_kitti,path_to_output,frame):
	d = depth_map.createSmoothMap(path_to_kitti,frame)
	save_path = path_to_output+frame
	plt.imsave(save_path,d)

	image = cv2.imread(save_path)
	gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	cv2.imwrite(save_path, gray_img)

	OFFSET = 60
	CURVE = 2000
	CUT_OFF = (len(d) // 3)

	# Create road mask
	for i in range(len(d)):
		if i < CUT_OFF:
			d[i] = [-16 for element in d[i]]
		else:
			for j in range(len(d[i])):
				if j != OFFSET and i < int( (float(CURVE)/float(j-OFFSET)) + float(CUT_OFF) ):
					d[i][j] = -16
				elif i < int( (-float(CURVE)/float(j-len(d[i]))) + float(CUT_OFF) ):
					d[i][j] = -16

	u = np.tile(range(d.shape[1]), d.shape[0])
	v = np.repeat(range(d.shape[0]), d.shape[1])
	uv_disp = np.array([u,v,d.flatten()]).T

	uvd_noblanks = uv_disp[uv_disp[:,2] != -16]
	u = uvd_noblanks[:,0].reshape(-1,1)
	v = uvd_noblanks[:,1].reshape(-1,1)
	d = uvd_noblanks[:,2]

	line_v,line_dv = ransac(v,d)
	img = Image.open(save_path)
	color_img = Image.new("RGB", img.size)
	color_img.paste(img)

	for i,disp in enumerate(d):

		if disp <= line_dv[line_v[v[i]-line_v[0]-1]-line_v[0]-1]:
			r,g,b = color_img.getpixel((int(u[i][0]),int(v[i][0])))
			color_img.putpixel((u[i][0],v[i][0]), (r+30,g,b)) # red

	return color_img


if __name__ == '__main__':

	path_to_kitti = "KITTI/data_scene_flow/testing/"
	frame = '000193_10.png'
	img = mapGround(path_to_kitti,"temp/", frame)
	img.save("temp/"+frame)
	img.show()
















#