# utils-share/zouinkhim/postprocessing/er/
path = "task2.err"

# process line segment_blockwise_er ▶:   0%|          | 0/21231161 [00:00<?, 234blocks/s, ⧗=1.86e+7, ▶=13, ✔=0, ✗=0, ∅=0] 
# and return number of blocks
def get_n_blocks_from_line(line):
    try:
        val = line.split("blocks")[0].split(" ")[-1]
        # print(val)
        return int(float(val))
    except:
        return 0

# put results in a list
result = []
# terabyte file, read file line by line
# length = 2000000
with open(path, "r") as f:
    for i, line in enumerate(f):
        # print(i, line)
        blocks = get_n_blocks_from_line(line)
        if blocks > 0:
            result.append(blocks)

# save results in a csv file and a plot

import csv
import matplotlib.pyplot as plt
import numpy as np

with open('blocks2.csv', 'w', newline='') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(result)

# read results from csv file
# with open('blocks.csv', newline='') as f:
#     reader = csv.reader(f)
#     result = list(reader)[0]
#     result = [int(i) for i in result]

# plt.hist(result, bins=100)
    
plt.plot(result)
plt.show()

plt.savefig("plot_without_smooth.png")
from scipy.ndimage.filters import gaussian_filter1d

ysmoothed = gaussian_filter1d(result, sigma=800)
plt.plot(ysmoothed)
plt.show()

# plt.plot(result)
# plt.title("Number of blocks per line")
# plt.xlabel("Number of blocks")
# plt.ylabel("Number of lines")
plt.savefig("smooth_plot_800_2.png")

plt.show()
plt.clf()
