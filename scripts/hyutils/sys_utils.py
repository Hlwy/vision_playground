# Created by: Hunter Young
# Date: 4/24/18
#
# Script Description:
# 	TODO
# Current Recommended Usage: (in terminal)
# 	TODO

import csv, pickle
import numpy as np
import os, sys, fnmatch

def create_new_directory(dir_name, parentDir=None, verbose=False):
    """
    Creates a new directory for storing extracted images saved to file for
    later CNN learning and post-processing.
    """
    if parentDir is not None: dir_name = os.path.join(parentDir,dir_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        if(verbose): print("Created directory \'%s\' " % dir_name)
    else: print("Directory \'%s\' already exists" % dir_name)
    return dir_name


def export_list2csv(_path, _file, _headers, _datalist):
	filenames = os.path.split(_file)
	filename = filenames[-1]
	print(filename)

	if not os.path.exists(str(_path)):
		print("Target output directory [" + str(_path) + "] does not exist --> MAKING IT NOW")
		os.makedirs(_path)

	csvFile = str(_path) + "/" + str(filename) + ".csv"
	with open(csvFile, "w") as output:
		writer = csv.writer(output, lineterminator='\n')
		writer.writerow(_headers)

		for row in range(len(_datalist)):
			tmpData = _datalist[row]
			writer.writerow(tmpData)
	print("	Data exporting to ...")

def import_csv2list(_filepath):
	data = []
	with open(_filepath, 'rb') as sd:
		r = csv.DictReader(sd)
		for line in r:
			data.append(line)
	return data


def save_pickle(path, data):
    f = open(path,"wb")
    pickle.dump(data,f)
    f.close()

def load_pickle(path):
    data = pickle.load(open(path, "rb" ))
    return data

def save_log_to_file(self, name="tmp_log.csv", data_array=None, headers=None, format=None,parent=None):
    if parent is not None: name = os.path.join(parent,name)
    if isinstance(headers, type(None)): np.savetxt(name,data_array,delimiter=",")
    else:
        if isinstance(format, type(None)): np.savetxt(name,data_array,delimiter=",", header=headers)
        else: np.savetxt(name,data_array,delimiter=",", header=headers, fmt=format)
