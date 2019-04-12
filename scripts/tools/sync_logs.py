# Created by: Hunter Young
# Date: 10/8/17
#
# Script Description:
#     This script is designed to essentially synchronize the data, that was recorded
#     in the datalog file, with the images that were extracted from the associated
#     video recording. The script outputs a .csv file containing all the data from
#     the datalog that matches up to the time of the useful extracted images.
#
# Current Recommended Usage: (in terminal)
# 	python parse_video.py -p /path/to/data/home/directory -d name-of-datalog-file.csv -v name-of-video-log-file.csv -o name_of_output_data_file

import cv2
import os
import csv
import argparse

# Hard-Coded values for the max and min PWM values (TODO:? incorporate commandline args for modifying these)
pwm_max = 2000
pwm_min = 1000
pwm_neutral = 1500

# Function for data interpolation
def interpolate(knownTime, prevRefTime, prevRefData, nextRefTime, nextRefData):
	refRatio = (nextRefData - prevRefData) / (nextRefTime - prevRefTime)
	data =  refRatio * (knownTime - prevRefTime) + prevRefData
	return data

# Function for normalizing PWM values
def normalize(_input, _neutral, _min, _max):
	# Slope
	m = (1 - (-1)) / (float(_max) - float(_min))
	# Y-intercept
	b = m * float(_neutral)
	# Normalized value
	out = m * float(_input) - b
	return float(out)

# Setup commandline argument(s) structures
ap = argparse.ArgumentParser(description='Data synchronization of extracted images log and the corresponding datalog.')
ap.add_argument("--home", "-p", type=str, metavar='PATH', help="Path to parent directory containing datalog, video, and extracted images")
ap.add_argument("--datalog", "-d", type=str, metavar='FILE', help="Name of datalog file that corresponds to the recorded video")
ap.add_argument("--videolog", "-v", type=str, metavar='FILE', help="Name of the log file of the extracted video images")
ap.add_argument("--output", "-o", type=str, metavar='NAME', default='synced_data', help="Name of output file containing synchronized data")
ap.add_argument("--convert", "-c", action="store_true", help="internal flag called to convert raw PWM values to a normalized value between -1 and 1")
# Store parsed arguments into array of variables
arg = ap.parse_args()
args = vars(arg)

# Extract stored arguments array into individual variables for later usage in script
home = args["home"]
vidLog = args["videolog"]
dataLog = args["datalog"]
outLog = args["output"]

# Change current directory to the specified home directory of the video of interest
os.chdir(home)

# Store file names for easier callback later
vidFile = "./data/" + str(vidLog)
dataFile = "./" + str(dataLog)
outFile = "./data/" + str(outLog) + ".csv"

# Initialize variables for data syncing
idxImg = 0	# Index of current line in video log
idxData = 0 # Index of current line in data log

# Store video log data as list for later usage
with open(vidFile, 'r') as fIn:
    vL = list(csv.reader(fIn))

# Store data log data as list for later usage
with open(dataFile, 'r') as fIn:
    dL = list(csv.reader(fIn))

# Store useful column entries in video log into arrays for later
imgNames = [i[0] for i in vL[4::]]
imgTimes = [i[2] for i in vL[4::]]

# Store useful column entries in data log into arrays for later
dataTimes = [i[1] for i in dL[1::]]
rawSpeed = [i[2] for i in dL[1::]]
rawYawRate = [i[3] for i in dL[1::]]
yawPwm = [i[4] for i in dL[1::]]
speedPwm = [i[5] for i in dL[1::]]

# Check if the first times are lined up
if imgTimes[0] < dataTimes[0]:
	# Ensure the extracted images used are those that are recorded at or after the datalog's first recorded time
	while(imgTimes[idxImg] < dataTimes[0]):
		idxImg += 1

# Store the shortened video log's data
frameNames = imgNames[idxImg::]
frameTimes = imgTimes[idxImg::]

# Reset the video log's index
idxImg = 0
# Establish an exit condition for the sync loop
idxEOF = len(frameNames) - 1

# Setup csv writer for later data output
with open(outFile, "w") as output:
	writer = csv.writer(output, lineterminator='\n')
	# Write header line for the output data sync .csv
	writer.writerow(["Time (msec)", "Image filename", "Raw Speed (m/s)", "Raw Yaw Speed (rad/s)", "Yaw Control (PWM pulsewidth us)", "Speed Control (PWM pulsewidth us)"])

	# Run through the datalogs syncing times
	while(idxImg <= idxEOF):
		tmpFrame = frameNames[idxImg]
		tmpTime = frameTimes[idxImg]

		# Check and update data logs' indices if interpolation is needed
		if(tmpTime == dataTimes[idxData]):  # If no interpolation is needed
			print "No interpolation Needed"
			tmpOutData = [tmpTime, tmpFrame, rawSpeed[idxData], rawYawRate[idxData], yawPwm[idxData], speedPwm[idxData]]
			writer.writerow(tmpOutData)
			idxImg += 1
			continue
		elif(tmpTime > dataTimes[idxData]): # Update datalog's index if needed
			print "Increasing Data Index"
			idxData += 1
		else:
			print "Interpolating..."

		# Reference Times for interpolations
		tmpRefT2 = float(dataTimes[idxData])
		tmpRefT1 = float(dataTimes[idxData - 1])

		# Initialize empty array to store each data column's interpolated value
		tmpOuts = []

		# Interpolate each datalog column
		for i in range(0,4):
			# Update Reference Data Used
			if(i == 0):
				tmpRefX = rawSpeed
			elif(i == 1):
				tmpRefX = rawYawRate
			elif(i == 2):
				tmpRefX = yawPwm
			elif(i == 3):
				tmpRefX = speedPwm

			# Extract specific reference data for interpolation
			tmpRefX2 = float(tmpRefX[idxData])
			tmpRefX1 = float(tmpRefX[idxData - 1])

			# Interpolate
			unknown = interpolate(float(tmpTime), tmpRefT1, tmpRefX1, tmpRefT2, tmpRefX2)
			# Store Interpolation
			tmpOuts.append(unknown)

		# Combine all relevent data to be written to file and save to csv
		if arg.convert: # Use normalized PWM values
			normYaw = normalize(tmpOuts[2], pwm_neutral, pwm_min, pwm_max)
			normThrottle = normalize(tmpOuts[3], pwm_neutral, pwm_min, pwm_max)
			tmpOutData = [tmpTime, tmpFrame, tmpOuts[0], tmpOuts[1], normYaw, normThrottle]
		else: # Use un-normalized PWM values
			tmpOutData = [tmpTime, tmpFrame, tmpOuts[0], tmpOuts[1], tmpOuts[2], tmpOuts[3]]

		# Save to file
		writer.writerow(tmpOutData)
		# print "Interpolating Data " + str(i)
		idxImg += 1
