import cv2
import numpy as np
import os
from constants import Constants
from mathHelpers import add_ones, pt_in_image, rotationMatrixToEulerAngles

kitti_dir = '/home/nils/nils/kitti/data_odometry_gray/dataset/'
#kitti_dir = '/media/localadmin/New Volume/11Nils/kitti/dataset/'
sequence = '00'
velo_dir = '/home/nils/nils/kitti/dataset/sequences/'

# https://github.com/hunse/kitti/blob/master/kitti/velodyne.py
def load_velodyne_points(drive,  frame):
    points_path = os.path.join(drive + '/velodyne/', "%06d.bin" % frame)
    points = np.fromfile(points_path, dtype=np.float32).reshape(-1, 4)
    points = points[:, :3]  # exclude luminance
    return points

def processPointCloud(img, pointcloud, pitch):
    i = 0
    usedPoints = 0
    mean_z = 0
    height, width, channels = img.shape
    processedImg = np.zeros((height, width, 3), np.uint8)
    print(len(pointcloud))
    while i < len(pointcloud):
        # check if the x coordinate is in front of the camera at all
        if pointcloud[i][0] > 2:
            # check if it's roughly in front of the camera
            if np.abs(pointcloud[i][1]) < 15:
                #test = np.array([pc_data[i][0], pc_data[i][1], pc_data[i][2], 1]).reshape(4,1)
                pts = np.matmul(consts.P_L2C, np.array([pointcloud[i][0], pointcloud[i][1], pointcloud[i][2],1]).reshape(4,1))
                #  [u v w]' = P * [X Y Z 1]'
                if pts[2] != 0:

                    #         x = u / w
                    #         y = v / w
                    u = int(np.divide(pts[0], pts[2]))
                    v = int(np.divide(pts[1], pts[2]))
                    # TODO: recover groundplane from the point cloud instead of assuming planar driving
                    z = pointcloud[i][2] + 1.73 # + sin(pitch)

                    if u > 0 and u < width:
                        if v > 0 and v < height:
                            if z > 0.25:
                                usedPoints += 1
                                cv2.line(processedImg, (u, v), (u, 0), (0, 0, 255), thickness = 4, lineType = 8)
        i +=1
    print("\nnum used points: ")
    print(usedPoints)
    return processedImg



consts = Constants()

consts.readFileLists(kitti_dir, sequence)

image = cv2.imread(consts.image_path + consts.image_names[0])
#cv2.imshow("test", image)
print("\nimage shape: \n")
height, width, channels = image.shape
print(image.shape)

pt = np.array((-.27, 0.0, 0.0), dtype = np.float).reshape(3, 1)
pose_tf = np.eye(4, dtype = np.float)

consts.readTfLidarToCamera0(kitti_dir, sequence)

i = 0
j = 0
k = 20
while i < len(consts.image_names) - 1:
    image = cv2.imread(consts.image_path + consts.image_names[i])
    j = i
    l = i - 1
    pt = np.array((0.0, 0.0, 0.0), dtype = np.float).reshape(3, 1)
    pt_1 = np.array((0.0, 0.0, 0.0), dtype = np.float).reshape(3, 1)
    pose_chunk = np.eye(4, dtype = np.float)
    pose_chunk[:3, :4] = np.array(consts.poses[i]).reshape(3, 4)
    [yaw, pitch, roll] = rotationMatrixToEulerAngles(pose_chunk[:3, :3])
    print(yaw * 180 / 3.1415, pitch * 180 / 3.1415, roll * 180 / 3.1415)

    #points = load_velodyne_points(velo_dir + sequence, i)
    #labeled_image = processPointCloud(image, points, pitch)
    labeled_image = np.zeros(image.shape, dtype = np.uint8)

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
        # store actual pose
        I = pose_chunk
        # right multiply actual pose with inverse of the last
        pose_chunk = np.matmul(inv_image_pose, pose_chunk)
        # store inverted actual pose for the next step
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
                cv2.circle(labeled_image, pt_l, 1, (0, 255, 0), -1)
            if pt_in_image(pt_r, width, height):
                right_in = 1
                cv2.circle(labeled_image, pt_r, 1, (0, 255,0), -1)

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
    

    u = v = 0
    while u < width - 1:
        while v < height - 1:
            if (labeled_image[v, u, 1] == 0 and labeled_image[v, u, 2] == 0):
                labeled_image[v, u] = (255, 0, 0)
            v += 1
        v = 0
        u += 1
'''
    vis = cv2.addWeighted(image, 1.0, labeled_image, 1.0, 0.0)
    cv2.imshow("vis", vis)
    cv2.waitKey(100)
    i += 1


#cv2.imshow("vis", vis)
#cv2.waitKey(0)   # in ms

