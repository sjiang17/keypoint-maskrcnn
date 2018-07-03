import cv2
import numpy as np
from dataset_store import Dataset


bag_name = '2018-04-09-10-11-35'
local_bag_name = '/mnt/truenas/datasets/v2/2018-04-09-10-11-35'
# open a bag on NAS
bag = Dataset.open(bag_name)
# or open a local bag
# bag = Dataset(local_bag_name)

# list the topics
print bag.list_topics()

# fetch images and show
idx = 0
for data_lst in bag.fetch_aligned('/camera1/image_color/compressed', '/camera3/image_color/compressed', ts_begin='2:00'):
    for i in range(2):
        im = cv2.imdecode(np.fromstring(data_lst[i][1].data, np.uint8), cv2.IMREAD_COLOR)
        if idx > 2000:
            continue
        cv2.imwrite("video/" + str(idx) + ".jpg", im)
        idx += 1
        #cv2.imshow(str(i), im)
        #cv2.waitKey(20)
