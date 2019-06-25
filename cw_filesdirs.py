# -*- coding: utf-8 -*-
# *****************************************************************************
# Author: Christian Wolf
#
# Helper functinos
#
# Changelog:
# 10.06.15 cw: begin development
# *****************************************************************************

from __future__ import print_function, division

import glob
import os


INVALID_FILENAME="__void__"

# *****************************************************************************
# Helper function to get img names. Do it recursively,
# also read in sub-directories.
# Files which are not images will lead to erros
# *****************************************************************************
	
def get_img_names(dir, max=-1):

	imgnames = get_img_names_recursive(dir, max)

	# Filter out the dummy names which correspond to directories 
	# (which we dealt with)
	imgnames_filtered = [x for x in imgnames if x != '__void__']

	return imgnames_filtered

# *****************************************************************************
# Recursive helper method for the above method
# *****************************************************************************

def get_img_names_recursive (dir, max=-1):

	print "Entering", dir

	# Get _ALL_ directory entries
	regex="%s/*"%dir
	entries=glob.glob(regex)

	# Collect all the directories in a list
	dirs=[]
	for i in range(len(entries)):
		fn = entries[i]
		if os.path.isdir(fn):
			dirs.append (fn)
			entries[i] = INVALID_FILENAME

	# For each directory, recursively call the method
	# and add the returned names to the current entries
	for dn in dirs:
		path = dn
		entries += get_img_names_recursive(path)

		print "max=", max, "len=",len(entries)
		if max>0 and len(entries)>max:
			break

	return entries
