import PIL
from PIL import Image
from os import listdir
import numpy as np
import time


data_dir = 'data/' # Include trailing slash!
output_resolution = (15,10)#(52,39)
output_filename = 'data' # .npz file in data_dir


states = ['here/','away/']

def get_data():
	d = [[],[]]
	filecount = 0
	starttime = time.time()
	for state in states:
		source_files = listdir(data_dir + state)

		for f in source_files:
			# f is a filename without its directory
			print(f)
			try:
				im = Image.open(data_dir + state + f)
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
	np.savez_compressed(data_dir+output_filename, xs=np.array(d[0]), ys=np.array(d[1]))
	print("Data saved.")
	print(time.time() - starttime, "seconds for", filecount, "photos.")


if __name__ == '__main__':
	get_data()