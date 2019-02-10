# Created by: Hunter Young
# Date: 10/5/17
#
# Script Description:
# 	Script is designed to take various commandline arguments making a very simple
# 	and user-friendly method to take any video file of interest as input and extract
# 	all the possible images available into a seperate folder, in addition to outputting
# 	a .csv file logging any additional useful information that is associated with each image.
#
# Current Recommended Usage: (in terminal)
# 	python parse_video.py -p /path/to/videos/home/directory -v name_of_video.mp4 -o name_of_output_data_file


import cv2
import os
import csv
import argparse

# Setup commandline argument(s) structures
ap = argparse.ArgumentParser(description='Extract images from a video file.')
ap.add_argument("--home", "-p", type=str, metavar='PATH', help="Path to directory containing video file")
ap.add_argument("--video", "-v", type=str, metavar='FILE', help="Name of video file to parse")
ap.add_argument("--outputs", "-o", type=str, metavar='NAME', default='processed_outputs', help="Name of output file containing information about parsed video")
# Store parsed arguments into array of variables
args = vars(ap.parse_args())

# Extract stored arguments array into individual variables for later usage in script
vid = args["video"]
home = args["home"]
dataFile = args["outputs"]

# Change current directory to the specified home directory of the video of interest
os.chdir(home)
# Create the directories used to store the extracted data, if they haven't already been made
if not os.path.exists("frames"):
	os.mkdir('frames')
if not os.path.exists("data"):
	os.mkdir('data')

# Initialize variables for data capturing
cap = cv2.VideoCapture(vid)	# Input video file as OpenCV VideoCapture device
idxFrame = 0	# Number of Frames extracted

# Setup the name for the output .csv file name
csvFile = "./data/" + str(dataFile) + ".csv"
# Begin logging pertinent information to the output .csv
with open(csvFile, "w") as output:
	writer = csv.writer(output, lineterminator='\n')

	# Log some initial information about the video file for referencing later
	width = cap.get(3) 		# Width of the video stream's frame (CV_CAP_PROP_FRAME_WIDTH)
	height = cap.get(4) 	# Height of the video stream's frame (CV_CAP_PROP_FRAME_HEIGHT)
	fps = cap.get(5) 		# FPS the video was captured at (CV_CAP_PROP_FPS)
	numFrames = cap.get(7) 	# Number of frames stored in the video file (CV_CAP_PROP_FRAME_COUNT)
	tmpData = [width, height, fps, numFrames,0.0]
	writer.writerow(["Width", "Height", "FPS", "# of Frames", "System Time @ Start of Video (msec)"])	# Write data header line
	writer.writerow(tmpData)									# Write out the data
	writer.writerow([])											# Create blank row for easier visualization

	# Log the data for the very first frame available
	writer.writerow(["Image Name", "Time of Frame in Video (msec)", "Frame Index"])	# Header line
	# tmpTime = cap.get(0) 		# Current position (msec) of extracted frame relative to the video file (CV_CAP_PROP_POS_MSEC)
	# tmpIdx = cap.get(1) 		# Current (0-based) index of the extracted frame relative to the video file (CV_CAP_PROP_FRAME_COUNT)
	# tmpSys = tmpTime + sysTime	# Time (msec) of Frame relative to the recording system's time (useful for cross-referencing data in recorded datalog)
	# nameImg = 'frame' + str(idxFrame) # Setup the base name of the output image file
	# tmpData = [nameImg, tmpTime, tmpSys, tmpIdx]
	# writer.writerow(tmpData)
	writer.writerow([])

	# idxFrame += 1	# Step forward the number of collected frames

	# For debugging
	print('Width, Height, FPS, # of Frames, System Time @ Start of Video (msec): %d,	%d,	%d,	%d,	%d' % (width, height, fps, numFrames, cap.get(0)))
	idxFrame = 0
	# Loop through the video file grabbing frames, collecting frame data and output to log file
	while(cap.grab()):	# Grab next frame and Loop until nothing else to grab
		tmpIdx = cap.get(1) 		# Current (0-based) index of the extracted frame relative to the video file (CV_CAP_PROP_FRAME_COUNT)
		tmpTime = float(tmpIdx) / float(fps)
		nameImg = 'frame_' + str(idxFrame) # Setup the base name of the output image file
		tmpData = [nameImg, tmpTime, tmpIdx]
		writer.writerow(tmpData)

		# Retrieve grabbed frame to extract it to .jpg file
		_, frame = cap.retrieve()
		# Name of file retrieved image is saved to
		name = './frames/' + nameImg + '.jpg'

		# Save grabbed image to file
		print('Creating ' + name + '	@ ' + str(tmpTime)) # Debugging
		cv2.imwrite(name, frame) # Save to file

		# Step forward
		idxFrame += 1

# Script Closure
cap.release()
cv2.destroyAllWindows()
