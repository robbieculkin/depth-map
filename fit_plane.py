# fit_plane.py

import numpy as np
from matplotlib import pyplot as plt
from depth_map import createMap
from sklearn.linear_model import RANSACRegressor as RR
from PIL import Image


def ransac_V(v,d):
	# plt.scatter(x=v,y=d, s=0.01)

	ransac_v = RR()
	ransac_v.fit(v,d)
	inlier_mask = ransac_v.inlier_mask_
	outlier_mask = np.logical_not(inlier_mask)

	line_v = np.arange(v.min(), v.max())[:, np.newaxis]
	line_d = ransac_v.predict(line_v)

	# plt.plot(line_v, line_d, color='gold', linewidth=2,label='RANSAC Regressor')
	# plt.xlabel("V")
	# plt.ylabel("Disparity")
	# plt.show()
	return line_v,line_d



def ransac_U(u,d):
	# plt.scatter(x=u,y=d, s=0.01)

	ransac_u = RR()
	ransac_u.fit(u,d)
	inlier_mask = ransac_u.inlier_mask_
	outlier_mask = np.logical_not(inlier_mask)

	line_u = np.arange(u.min(), u.max())[:, np.newaxis]
	line_d = ransac_u.predict(line_u)

	# plt.plot(line_u, line_d, color='gold', linewidth=2,label='RANSAC Regressor')
	# plt.xlabel("U")
	# plt.ylabel("Disparity")
	# plt.show()

	return line_u,line_d


def get_u_v_disparity(img):
	path_to_kitti = "KITTI/data_scene_flow/testing/"
	disparity = createMap(path_to_kitti,img)
	plt.imsave("temp.png", disparity)
	

	u = np.tile(range(disparity.shape[1]), disparity.shape[0])
	v = np.repeat(range(disparity.shape[0]), disparity.shape[1])
	uv_disp = np.array([u,v,disparity.flatten()]).T

	uv_disp_noblanks = uv_disp[uv_disp[:,2]!=-16]
	u = uv_disp_noblanks[:,0].reshape(-1, 1)
	v = uv_disp_noblanks[:,1].reshape(-1, 1)
	d = uv_disp_noblanks[:,2]

	return [u,v,d]


if __name__ == '__main__':
	path_to_kitti = "KITTI/data_scene_flow/testing/"
	image_path = '000188_11.png'
	disparity = createMap(path_to_kitti,image_path)
	plt.imsave("temp.png", disparity)

	
	arr = get_u_v_disparity(image_path)
	u = arr[0]
	v = arr[1]
	d = arr[2]


	print(len(d))

	N = 3
	
	d = np.repeat(d, N+1)
	d = d.reshape(-1, 1)
	# u = [[u[i][0] for x in range(N+1)] for i in range(len(u))]
	print(d)
	print(type(d))
	print(len(d))


	# i = int(image_path[0:6])

	# direction = -1
	# if i - N < 0:
	# 	direction = 1

	# j = 1
	# while j < N+1:
	# 	i += direction

	# 	zeros = "0" * (6-len(str(i)))
	# 	new_path = zeros + str(i) + "_11.png"

	# 	d.append(get_u_v_disparity(new_path)[2])

	# 	j += 1




	# line_u,line_du = ransac_U(u, d)
	# line_v,line_dv = ransac_V(v,d)
	# img = Image.open("temp.png")

	# for i,disp in enumerate(d):

	# 	# Should below line be an OR or AND???
	# 	if disp <= line_du[line_u[u[i]-line_u[0]-1]-line_u[0]-1] or disp <= line_dv[line_v[v[i]-line_v[0]-1]-line_v[0]-1]:
	# 		img.putpixel((u[i][0],v[i][0]), (255,0,0)) # red

	# 	else:
	# 		img.putpixel((u[i][0],v[i][0]), (0,255,0)) # green


	# img.show()
















#