import cv2
import numpy as np
import os
from constants import Constants
from mathHelpers import add_ones, pt_in_image

kitti_dir = '/home/nils/nils/kitti/data_odometry_gray/dataset/'
#kitti_dir = '/media/localadmin/New Volume/11Nils/kitti/dataset/'
sequence = '10'

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
k = 20
while i < len(consts.image_names) - 1:
    image = cv2.imread(consts.image_path + consts.image_names[i])
    j = i
    l = i - 1
    pt = np.array((0.0, 0.0, 0.0), dtype = np.float).reshape(3, 1)
    pt_1 = np.array((0.0, 0.0, 0.0), dtype = np.float).reshape(3, 1)
    labeled_image= np.zeros((height, width, 3), np.uint8)
    pose_chunk = np.eye(4, dtype = np.float)
    pose_chunk[:3, :4] = np.array(consts.poses[i]).reshape(3, 4)

    inv_image_pose = np.linalg.inv(pose_chunk)
    pose_chunk_t1 = np.eye(4, dtype = np.float)
    if l > 1:
        pose_chunk_t1[:3, :4] = np.array(consts.poses[l]).reshape(3, 4)
        inv_image_pose_t1 = np.linalg.inv(pose_chunk_t1)

    pt_r_last = [-1, -1]
    pt_l_last = [-1, -1]
    pt_1_l_last = [-1, -1]
    pt_1_r_last = [-1, -1]
    while j < i + k:
        # concatenate the posetransformations first before multiplying with pt and K
        if j > len(consts.image_names):
            break
        I = np.eye(4)
        pose_chunk[:3, :4] = np.array(consts.poses[j]).reshape(3, 4)
        I = pose_chunk
        pose_chunk = np.matmul(inv_image_pose, pose_chunk)
        if pose_chunk[1][1] < 0.95:
            print ("\n")
            print(pose_chunk)
            print ("\n")
        inv_image_pose = np.linalg.inv(I)
        # reshaping pt to 4x1
        pt = np.array((pt[0], pt[1], pt[2], 1.0), dtype = np.float).reshape(4, 1)
        # concatenating forward pose transformation to point
        pt = np.matmul(pose_chunk, pt)

        # projection into image coordinates
        uvw_l = np.matmul(consts.KRT_left, np.array((pt[0], pt[1], pt[2], 1.0), dtype = np.float).reshape(4, 1))
        uvw_r = np.matmul(consts.KRT_right, np.array((pt[0], pt[1], pt[2], 1.0), dtype = np.float).reshape(4, 1))
        if uvw_l[2] != 0:
            #         x = u / w
            #         y = v / w
            pt_l = (int(np.divide(uvw_l[0], uvw_l[2])), int(np.divide(uvw_l[1], uvw_l[2])))
            pt_r = (int(np.divide(uvw_r[0], uvw_r[2])), int(np.divide(uvw_r[1], uvw_r[2])))
            left_in = 0
            right_in = 0

            if pt_in_image(pt_l, width, height):
                left_in = 1
                cv2.circle(labeled_image, pt_l, 3, (0, 0, 255), -1)
            if pt_in_image(pt_r, width, height):
                right_in = 1
                cv2.circle(labeled_image, pt_r, 3, (0, 255,), -1)

            if left_in == 1 and right_in == 1 and pt_in_image(pt_l_last, width, height) and pt_in_image(pt_r_last, width, height):
                rect = np.array([pt_l, pt_r, pt_r_last, pt_l_last], np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(labeled_image, [rect], (0, 255, 0))
            if j > i:
               pt_l_last = pt_l
               pt_r_last = pt_r

        j += 1
    '''
    if l > 1:
        while l < i + int(k * 0.9):
            # concatenate the posetransformations first before multiplying with pt and K
            if l > len(consts.image_names):
                break
            I = np.eye(4)
            pose_chunk_t1[:3, :4] = np.array(consts.poses[l]).reshape(3, 4)
            I = pose_chunk_t1
            pose_chunk_t1 = np.matmul(inv_image_pose_t1, pose_chunk_t1)
            inv_image_pose_t1 = np.linalg.inv(I)
            # reshaping pt to 4x1
            pt_1 = np.array((pt_1[0], pt_1[1], pt_1[2], 1.0), dtype = np.float).reshape(4, 1)
            # concatenating forward pose transformation to point
            pt_1 = np.matmul(pose_chunk_t1, pt_1)

            # projection into image coordinates
            uvw_l = np.matmul(consts.KRT_left, np.array((pt_1[0], pt_1[1], pt_1[2], 1.0), dtype = np.float).reshape(4, 1))
            uvw_r = np.matmul(consts.KRT_right, np.array((pt_1[0], pt_1[1], pt_1[2], 1.0), dtype = np.float).reshape(4, 1))
            if uvw_l[2] != 0:
                #         x = u / w
                #         y = v / w
                pt_1_l = (int(np.divide(uvw_l[0], uvw_l[2])), int(np.divide(uvw_l[1], uvw_l[2])))
                pt_1_r = (int(np.divide(uvw_r[0], uvw_r[2])), int(np.divide(uvw_r[1], uvw_r[2])))
                left_in = 0
                right_in = 0

                if pt_in_image(pt_1_l, width, height):
                    left_in = 1
                    cv2.circle(labeled_image, pt_1_l, 3, (0, 0, 255), -1)
                if pt_in_image(pt_1_r, width, height):
                    right_in = 1
                    cv2.circle(labeled_image, pt_1_r, 3, (0, 255,), -1)
                if left_in == 1 and right_in == 1:
                    rect = np.array([pt_1_l, pt_1_r, pt_1_r_last, pt_1_l_last], np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(labeled_image, [rect], (0, 0, 150))
                if l > i:
                    pt_1_l_last = pt_1_l
                    pt_1_r_last = pt_1_r
        l += 1
    '''
    vis = cv2.addWeighted(image, 1.0, labeled_image, 1.0, 0.0)
    cv2.imshow("vis", vis)
    cv2.waitKey(100)
    i += 1


#cv2.imshow("vis", vis)
#cv2.waitKey(0)   # in ms

