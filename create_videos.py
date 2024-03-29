# run_all.py
from fit_plane import mapGround
from PIL import Image
from multiprocessing import Process
from matplotlib import pyplot as plt
import os
import cv2
import numpy as np
import time


def createVideo(images_folder, disparity_folder, ground_folder, combined_folder, video_folder, video_file, n):
	FPS = 2
	FIRST_DIGITS = 6

	print("Compiling images in " + images_folder + " into a video: " + video_file)
	cur_dir = os.path.dirname(__file__)
	images_folder = os.path.join(cur_dir, images_folder)
	disparity_folder = os.path.join(cur_dir, disparity_folder)
	ground_folder = os.path.join(cur_dir, ground_folder)
	combined_folder = os.path.join(cur_dir, combined_folder)
	video_folder = os.path.join(cur_dir, video_folder)

	i = 0
	switch = False
	video = None

	while i < n:
		frame = str(i)
		num_zeros = FIRST_DIGITS - len(frame)
		frame_name = ('0'*num_zeros) + frame + '_1' + str(int(switch)) + '.png'
		print(frame_name)

		original_img = cv2.imread(images_folder+frame_name)
		disp_img = cv2.imread(disparity_folder+frame_name)
		ground_img = cv2.imread(ground_folder+frame_name)

		img_combined = np.concatenate((original_img, disp_img, ground_img), axis=0)
		height,width,layers = img_combined.shape
		size = (width,height)


		cv2.imwrite(combined_folder+frame_name, img_combined)

		if i == 0 and not switch:
			video = cv2.VideoWriter(video_folder+video_file, cv2.VideoWriter_fourcc('M','J','P','G'), FPS, size) # 
		video.write(img_combined)

		if switch:
			i += 1
		switch = not switch

	video.release()

def runRange(path_to_kitti, path_to_save, path_to_disparity, ending, i, n):
	FIRST_DIGITS = 6

	while i < n:
		frame = str(i)
		num_zeros = FIRST_DIGITS - len(frame)
		frame_name = ('0'*num_zeros) + frame + '_' + ending + '.png'
		print("Processing " + frame_name + "...")

		
		img = mapGround(path_to_kitti, path_to_disparity, frame_name)
		img.save(path_to_save+frame_name)
		print("Completed "+frame_name+"!")

		i += 1


def runAll(path_to_kitti, path_to_save, path_to_disparity):
	processes = []
	NUM_PROCESSES = 20

	for i in range(NUM_PROCESSES):
		start = 10*i
		end = start+10
		print("process "+str(i)+" starts on "+str(start)+" and ends on "+str(end))

		even_process = Process(target=runRange, args=(path_to_kitti, path_to_save, path_to_disparity, "10", start, end,))
		odd_process = Process(target=runRange, args=(path_to_kitti, path_to_save, path_to_disparity, "11", start, end,))

		even_process.start()
		odd_process.start()

		processes.append(even_process)
		processes.append(odd_process)

	for process in processes:
		process.join()



if __name__ == '__main__':
	start_time = time.time()
	images_folder = "KITTI/data_scene_flow/testing/"
	ground_frames = "ground_detected_frames/"
	disparity_frames = "disparity_mapped_frames/"

	# runAll(images_folder, ground_frames, disparity_frames)
	createVideo(images_folder+"image_2/", disparity_frames, ground_frames, "combined/", "videos/", "dataset.avi", 200)

	print("Execution time: " + str(time.time() - start_time) + " seconds")
