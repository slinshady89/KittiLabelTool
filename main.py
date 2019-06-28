import cv2
import numpy as np
import os
from constants import Constants
from mathHelpers import add_ones

kitti_dir = '/home/nils/nils/kitti/data_odometry_gray/dataset/'
#kitti_dir = '/media/localadmin/New Volume/11Nils/kitti/dataset/'
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
    labeled_image= np.zeros((height, width, 3), np.uint8)
    pose_chunk = np.eye(4, dtype = np.float)
    pose_chunk[:3, :4] = np.array(consts.poses[i]).reshape(3, 4)
    inv_image_pose = np.linalg.inv(pose_chunk)
    u_l_last = v_l_last = u_r_last = v_r_last = -1
    while j < i + k:
        # concatenate the posetransformations first before multiplying with pt and K
        if j > len(consts.image_names):
            break
        pose_chunk[:3, :4] = np.array(consts.poses[j]).reshape(3, 4)
        pose_chunk = np.matmul(inv_image_pose, pose_chunk)


        #print("pose_chunk \n")
        #print(pose_chunk)
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
            u_l = int(np.divide(uvw_l[0], uvw_l[2]))
            v_l = int(np.divide(uvw_l[1], uvw_l[2]))
            u_r = int(np.divide(uvw_r[0], uvw_r[2]))
            v_r = int(np.divide(uvw_r[1], uvw_r[2]))
            left_in = 0
            right_in = 0

            if u_l > 0 and u_l < width:
                if v_l > 0 and v_l < height:
                    left_in = 1
                    print ("\nleft")
                    print("\nu and v \n")
                    print(u_l, v_l)
                    cv2.circle(labeled_image, (u_l, v_l), 3, (0, 0, 255), -1)
            if u_r > 0 and u_r < width:
                if v_r > 0 and v_r < height:
                    print("\nright")
                    print("\nu and v \n")
                    print(u_r, v_r)
                    right_in = 1
                    cv2.circle(labeled_image, (u_r, v_r), 3, (0, 255,), -1)

            if left_in == 1 and right_in == 1:
                rect = np.array([[u_l, v_l], [u_r, v_r], [u_r_last, v_r_last], [u_l_last, v_l_last]], np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(labeled_image, [rect], (0, 255, 0))
                #cv2.rectangle(labeled_image, (u_l, u_l_last), (u_r, u_r_last), (0, 0, 255), cv2.FILLED)
            if j > i:
                u_l_last = u_l
                v_l_last = v_l
                u_r_last = u_r
                v_r_last = v_r
        j += 1

    vis = cv2.addWeighted(image, 1.0, labeled_image, 1.0, 0.0)
    cv2.imshow("vis", vis)
    cv2.waitKey(100)
    i += 1


#cv2.imshow("vis", vis)
#cv2.waitKey(0)   # in ms

