import cv2
import numpy as np
import os
from constants import Constants
from mathHelpers import add_ones

#kitti_dir = '/home/nils/nils/kitti/data_odometry_gray/dataset/'
kitti_dir = '/media/localadmin/New Volume/11Nils/kitti/dataset/'
sequence = '00'

consts = Constants()

consts.readFileLists(kitti_dir, sequence)

image = cv2.imread(consts.image_path + consts.image_names[0])
#cv2.imshow("test", image)
print("\nimage shape: \n")
height, width, channels = image.shape
print(image.shape)
labeled_image= np.zeros((height,width,3), np.uint8)

pt = np.array((0.0, 0.0, 0.0), dtype = np.float).reshape(3, 1)
pose_tf = np.eye(4, dtype = np.float)

i = 0
j = 0
k = 10
while i < len(consts.image_names) - 1:
    image = cv2.imread(consts.image_path + consts.image_names[i])
    j = i
    pt = np.array((0.0, 0.0, 0.0), dtype = np.float).reshape(3, 1)
    labeled_image= np.zeros((height,width,3), np.uint8)
    pose_chunk = np.eye(4, dtype = np.float)
    last_pose = pose_chunk

    while j < i + k:
        # concatenate the posetransformations first before multiplying with pt and K
        pose_chunk[:3, :4] = np.array(consts.poses[j]).reshape(3, 4)
        print("pose_chunk \n")
        print(pose_chunk)
        #pose_tf = np.matmul(pose_tf, pose_chunk)
        #print("\npose_conc \n")
        #print(pose_tf)

        # reshaping pt to 4x1
        pt = np.array((pt[0], pt[1], pt[2], 1.0), dtype = np.float).reshape(4, 1)
        # concatenating forward pose transformation to point
        pt = np.matmul(pose_chunk, pt)
        #print("pt")
        #print(pt)
        #print("\n")
        # projection into image coordinates
        uvw_l = np.matmul(consts.KRT_left, np.array((pt[0], pt[1], pt[2], 1.0), dtype = np.float).reshape(4, 1))
        uvw_r = np.matmul(consts.KRT_right, np.array((pt[0], pt[1], pt[2], 1.0), dtype = np.float).reshape(4, 1))
        #print("uvw")
        #print(uvw)
        #print("\n")
        if uvw_l[2] != 0:
            #         x = u / w
            #         y = v / w
            u = int(np.divide(uvw_l[0], uvw_l[2]))
            v = int(np.divide(uvw_l[1], uvw_l[2]))
            if u > 0 and u < width:
                if v > 0 and v < height:
                    cv2.circle(labeled_image, (u, v), 2, (0, 0, 255), -1)
            u = int(np.divide(uvw_r[0], uvw_r[2]))
            v = int(np.divide(uvw_r[1], uvw_r[2]))
            if u > 0 and u < width:
                if v > 0 and v < height:
                    cv2.circle(labeled_image, (u, v), 2, (0, 255, ), -1)
        j += 1

    vis = cv2.addWeighted(image, 1.0, labeled_image, 1.0, 0.0)
    cv2.imshow("vis", vis)
    cv2.waitKey(100)
    i += 1


#cv2.imshow("vis", vis)
#cv2.waitKey(0)   # in ms

