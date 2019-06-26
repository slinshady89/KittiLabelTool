import cv2
import numpy as np
import os

from dateutil.rrule import weekday

from constants import Constants

kitti_dir = '/home/nils/nils/kitti/data_odometry_gray/dataset/'
sequence = '00'

consts = Constants()

consts.readFileLists(kitti_dir, sequence)

image = cv2.imread(consts.image_path + consts.image_names[0])
cv2.imshow("test", image)
height, width, channels = image.shape
print(image.shape)
labeled_image= np.zeros((height,width,3), np.uint8)

pt = np.array((0.0, 0.0, 0.0), dtype=np.float).reshape(3, 1)

i = 0
while i < 20:
    pose_tf = np.array(consts.poses[i]).reshape(3,4)
    # reshaping pt to 4x1
    pt = np.array((pt[0], pt[1], pt[2], 1.0), dtype = np.float).reshape(4, 1)
    # concatenating forward posetransformation
    pt = np.matmul(pose_tf, pt)
    print("pt")
    print(pt)
    print("\n")
    # projection into image coordinates
    uvw = np.matmul(consts.KRT_left, np.array((pt[0], pt[1], 0.0, 1.0), dtype = np.float).reshape(4, 1))
    print("uvw")
    print(uvw)
    print("\n")
    if pt[2] != 0:
            #         x = u / w
            #         y = v / w
            u = int(np.divide(uvw[0], uvw[2]))
            v = int(np.divide(uvw[1], uvw[2]))
            print(u, v)
            if u > 0 and u < height:
                if v > 0 and v < width:
                    print(u, v)
                    cv2.circle(labeled_image, (v, u), 2,(0, 0, 255), -1)


    i += 1

#cv2.imshow("labeled", labeled_image)

vis = cv2.addWeighted(image, 1.0, labeled_image, 1.0, 0.0)
cv2.imshow("vis", vis)
cv2.waitKey(5000) # in ms

