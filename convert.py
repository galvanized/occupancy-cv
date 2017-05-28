import PIL
from PIL import Image
from os import listdir
import numpy as np
import time


# INCLUDE TRAILING SLASHES FOR DIRS
root_dir = ''
source_dir = root_dir + 'capture/'
output_dir = root_dir + 'data/'
output_resolution = (15,10)#(52,39)
output_filename = 'data' # .npz file in root_dir


states = ['here/','away/']

def image_convert():
	for state in states:
		source_files = listdir(source_dir + state)
		output_files = listdir(output_dir + state)

		# assume same file name
		new_files = [f for f in source_files if f not in output_files]

		for f in new_files:
			# f is a filename without its directory
			print(f)
			try:
				im = Image.open(source_dir + state + f)
				om = im.resize(output_resolution)
				om.save(output_dir + state + f)

			except OSError:
				print("Could not convert " + f)


def get_data():
	d = [[],[]]
	filecount = 0
	starttime = time.time()
	for state in states:
		source_files = listdir(source_dir + state)

		for f in source_files:
			# f is a filename without its directory
			print(f)
			try:
				im = Image.open(source_dir + state + f)
				om = im.resize(output_resolution)
				# normalize?
				l = list(om.getdata())
				channels = 3
				x = []
				y = [state == 'here/', state == 'away/']
				for i, p in enumerate(l):
					for c in range(channels):
						x.append(p[c])
				d[0].append(x)
				d[1].append(y)
				filecount += 1

			except OSError:
				print("Could not convert " + f)
	np.savez_compressed(root_dir+output_filename, xs=np.array(d[0]), ys=np.array(d[1]))
	print("Data saved.")
	print(time.time() - starttime, "seconds for", filecount, "photos.")


if __name__ == '__main__':
	get_data()