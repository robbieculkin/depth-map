# run_all.py
from fit_plane import mapGround
from PIL import Image
from threading import Thread
from multiprocessing import Process
import os
import cv2
import numpy as np
import glob
import time


def createVideo(images_folder, video_folder, video_file):
	img_array = []
	FPS = 2

	print("Compiling images in " + images_folder + " into a video: " + video_file)
	images_folder = os.path.join(os.path.dirname(__file__), images_folder)
	video_folder = os.path.join(os.path.dirname(__file__), video_folder)
	for file in glob.glob(images_folder+"*.png"):
		img = cv2.imread(file)
		height,width,layers = img.shape
		size = (width,height)
		img_array.append(img)

	video = cv2.VideoWriter(video_folder+video_file, 0, FPS, size)

	for i in range(len(img_array)):
		video.write(img_array[i])

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
	NUM_PHOTOS = 200
	evens_thread = Thread(target=runRange, args=(path_to_kitti, path_to_save, path_to_disparity, "10", 0, NUM_PHOTOS,))
	odds_thread = Thread(target=runRange, args=(path_to_kitti, path_to_save, path_to_disparity, "11", 0, NUM_PHOTOS,))

	evens_thread.start()
	odds_thread.start()

	evens_thread.join()
	odds_thread.join()

def runAllPlus(path_to_kitti, path_to_save, path_to_disparity):
	threads = []
	NUM_THREADS = 20

	for i in range(NUM_THREADS):
		start = 10*i
		end = start+10
		print("thread "+str(i)+" starts on "+str(start)+" and ends on "+str(end))

		even_thread = Thread(target=runRange, args=(path_to_kitti, path_to_save, path_to_disparity, "10", start, end,))
		odd_thread = Thread(target=runRange, args=(path_to_kitti, path_to_save, path_to_disparity, "11", start, end,))

		even_thread.start()
		odd_thread.start()

		threads.append(even_thread)
		threads.append(odd_thread)

	for thread in threads:
		thread.join()

def runAllPlusPlus(path_to_kitti, path_to_save, path_to_disparity):
	processes = []
	NUM_PROCESSES = 20

	for i in range(NUM_PROCESSES):
		start = 10*i
		end = start+10
		print("thread "+str(i)+" starts on "+str(start)+" and ends on "+str(end))

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

	runAllPlusPlus(images_folder, ground_frames, disparity_frames)
	createVideo(images_folder+"image_2", "videos/", "dataset.avi")
	createVideo(ground_frames, "videos/", "ground_map.avi")
	createVideo(disparity_frames, "videos/", "disparity_map.avi")

	print("Execution time: " + str(time.time() - start_time) + " seconds")
