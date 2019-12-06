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


if __name__ == '__main__':

	path_to_kitti = "KITTI/data_scene_flow/testing/"
	disparity = createMap(path_to_kitti,'000199_11.png')
	plt.imsave("temp.png", disparity)

	LOOKBACK_WINDOW = 2
	FRAME = 199
	uvd = np.array([[-16,-16,-16]])

	for lb in range(LOOKBACK_WINDOW):
		d = createMap(path_to_kitti, f'000{FRAME-lb}_11.png')
		u = np.tile(range(d.shape[1]), d.shape[0])
		v = np.repeat(range(d.shape[0]), d.shape[1])
		uv_disp = np.array([u,v,d.flatten()]).T
		uvd = np.concatenate((uvd, uv_disp))

	print(uvd.shape)

	uvd_noblanks = uvd[uvd[:,2] != -16]
	u = uvd_noblanks[:,0].reshape(-1,1)
	v = uvd_noblanks[:,1].reshape(-1,1)
	d = uvd_noblanks[:,2]




	line_u,line_du = ransac_U(u, d)
	line_v,line_dv = ransac_V(v,d)
	img = Image.open("temp.png")

	for i,disp in enumerate(d):

		# Should below line be an OR or AND???
		if disp <= line_du[line_u[u[i]-line_u[0]-1]-line_u[0]-1] and disp <= line_dv[line_v[v[i]-line_v[0]-1]-line_v[0]-1]:
			img.putpixel((u[i][0],v[i][0]), (255,0,0)) # red

		else:
			img.putpixel((u[i][0],v[i][0]), (0,255,0)) # green


	img.show()
















#