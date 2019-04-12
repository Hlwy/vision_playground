import cv2
import argparse
import os, sys, fnmatch
import numpy as np

def add_filename_prefixs(_dir, _prefix):
	filenames = os.listdir(_dir)
	os.chdir(_dir)
	for file in filenames:
		newName = str(_prefix) + "_" + str(file)
		os.rename(file, newName)
	print("Finished")

def grab_dir_images(_dir, patterns = ['*png','*jpg'],verbose=False):
    found = []
    for root, dirs, files in os.walk(_dir):
        for pat in patterns:
            for file in files:
                if fnmatch.fnmatch(file, pat):
                    found.append(os.path.join(root, file))

    imgs = [cv2.imread(path) for path in found]
    if(verbose): print(found)
    return found, imgs

def cycle_through_images(key, _imgs, _paths, index, flags=[False]):
	n = len(_imgs)
	_flag = False
	post_recording_step = flags[0]

	if key == ord('p') or post_recording_step == True:
		index = index + 1
		if index >= n:
			index = 0
		_flag = True
		print('Next Image...')
	if key == ord('o'):
		index = index - 1
		if index < 0:
			index = n - 1
		_flag = True
		print('Previous Image...')

	new_img = np.copy(_imgs[index])
	new_img_path = _paths[index]
	return new_img, new_img_path, index, _flag

def cycle_through_filters(key, index, max_index=2):
	if key == ord('l'):
		index += 1
		if index >= max_index:
			index = 0
		print('Next Filter...')
	if key == ord('k'):
		index -= 1
		if index < 0:
			index = max_index - 1
		print('Previous Filter...')

	filter_index = index
	return filter_index


def spread_image(image, percentile=1.1):
    copy = image.copy()
    summed = np.sum(copy, axis=1)
    total = np.sum(summed)
    percentile_limit = total / percentile
    running_total = 0
    for i in range(summed.shape[0]):
        running_total += summed[i]
        if running_total > percentile_limit:
            break

    reshape_factor = copy.shape[0] / float(i)
    print(copy.shape)
    copy = cv2.resize(copy, (copy.shape[1], int(copy.shape[0]*reshape_factor)), interpolation=cv2.INTER_NEAREST)
    print(copy.shape)
#     tmpI = int(copy.shape[0]/reshape_factor)+1
#     print(tmpI)
#     copy = copy[:tmpI,:]
#     print(copy.shape)
    return copy, reshape_factor


def despread_image(image,reshape_factor):
    copy = image.copy()
    print(copy.shape)
    sz1 = copy.shape[1]
    sz2 = int(copy.shape[0]/reshape_factor)+2
#     sz2 = int(int(copy.shape[0]/reshape_factor)*reshape_factor)

    copy = cv2.resize(copy, (sz1, sz2), interpolation=cv2.INTER_NEAREST)
    sz3 = int(copy.shape[0]*reshape_factor)
#     copy = copy[:sz3,:]
    print(sz1,sz2,sz3)
    return copy


def strip_image(_img, nstrips=48, horizontal_strips=True, verbose=False):
	strips = []
	h,w = _img.shape[:2]
	if(verbose): print("[INFO] strip_image() ---- Input Image Shape: %s" % (str(_img.shape)))

	if horizontal_strips == True:
		strip_widths = h // nstrips
		axis = 0
		maxDim = h
	else:
		strip_widths = w // nstrips
		axis = 1
		maxDim = w

	if verbose: print("[INFO] strip_image() ---- Strips Info:")
	for strip_number in range(nstrips):
		dim1 = (strip_number*strip_widths)
		dim2 = (strip_number+1)*(strip_widths)
		if strip_number == (nstrips-1):
			diff = maxDim - dim2
			dim2 += diff
			if verbose: print("\tStrip [%d] MaxDim, DimDiff, Dim2: %d, %d, %d" % (strip_number, maxDim, diff, dim2))

		if horizontal_strips == True: tmp = _img[dim1:dim2, :]
		else: tmp = _img[:, dim1:dim2]

		strips.append(tmp)
		if verbose: print("\tStrip [%d] Dim1, Dim2, Strip Shape: %d, %d, %s" % (strip_number, dim1, dim2, str(tmp.shape)))
	testImg = np.concatenate(strips, axis=axis)
	if(verbose): print("[INFO] strip_image() ---- Strip Width, Resulting Image Shape: %d, %s" % (strip_widths,str(testImg.shape)))
	return strips


def histogram_sliding_filter(hist, window_size=16, flag_plot=False):
	avg_hist = np.zeros_like(hist).astype(np.int32)
	sliding_window = np.ones((window_size,))/window_size

	try:
		n, depth = hist.shape
	except:
		n = hist.shape
		depth = None

	if flag_plot == True:
		plt.figure(1)
		plt.clf()
		plt.title('Smoothed Histogram of the image')

	if depth == None:
		avg_hist = np.convolve(hist[:], sliding_window , mode='same')
		if flag_plot == True:
			plt.plot(range(avg_hist.shape[0]), avg_hist[:])
	else:
		for channel in range(depth):
			tmp_hist = np.convolve(hist[:,channel], sliding_window , mode='same')
			avg_hist[:,channel] = tmp_hist
			if flag_plot == True:
				plt.plot(range(avg_hist.shape[0]), avg_hist[:,channel])
				# plt.plot(range(avg_hist.shape[0]), avg_hist[:,1])
				# plt.plot(range(avg_hist.shape[0]), avg_hist[:,2])
	return avg_hist

def crop_bottom_half(image):
	cropped_img = image[image.shape[0]/2:image.shape[0]]
	return cropped_img

def crop_bottom_two_thirds(image):
	cropped_img = image[image.shape[0]/6:image.shape[0]]
	return cropped_img

def crop_below_pixel(image, y_pixel):
	cropped_img = image[y_pixel:image.shape[0]]
	return cropped_img

def horizontal_hist(_img, method=0):
	if method == 1:		# Take a histogram of the bottom half of the image
		hist = np.sum(_img[_img.shape[0]//2:,:], axis=0)
	elif method == 2:	# Take a histogram of the top half of the image
		hist = np.sum(_img[0:_img.shape[0]//2,:], axis=0)
	else:				# Take a histogram of the whole image
		hist = np.sum(_img[:,:], axis=0)
	return hist

def vertical_hist(_img, method=0):
	if method == 1:			# Take a histogram of the left half of the image
		hist = np.sum(_img[_img.shape[1]//2:,:], axis=1)
	elif method == 2:		# Take a histogram of the right half of the image
		hist = np.sum(_img[0:_img.shape[1]//2,:], axis=1)
	else:					# Take a histogram of the whole image
		hist = np.sum(_img[:,:], axis=1)
	return hist


def custom_hist(_img, rows=[0,0], cols=[0,0], axis=0,flag_plot=False):
	# Take a histogram of the whole image
	hist = np.sum(_img[rows[0]:rows[1],cols[0]:cols[1]], axis=axis)

	# print("Histogram Shape: ", hist.shape)

	if flag_plot == True:
		plt.figure(1)
		plt.clf()
		plt.title('Histogram of the image')
		plt.plot(range(hist.shape[0]), hist[:])
		# plt.plot(range(hist.shape[0]), hist[:,1])
		# plt.plot(range(hist.shape[0]), hist[:,2])

	return hist


#Run Main
if __name__ == "__main__" :

	# Setup commandline argument(s) structures
	ap = argparse.ArgumentParser(description='Image Segmentation')
	ap.add_argument("--prefix", "-p", type=str, default='test', metavar='NAME', help="Name of video file to parse")
	ap.add_argument("--directory", "-d", type=str, default='test', metavar='FOLDER', help="Name of video file to parse")
	# Store parsed arguments into array of variables
	args = vars(ap.parse_args())

	# Extract stored arguments array into individual variables for later usage in script
	prefix = args["prefix"]
	dir = args["directory"]

	add_filename_prefixs(dir, prefix)
