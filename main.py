import cv2
import numpy as np
import os
from constants import Constants
from mathHelpers import pt_in_image, rotationMatrixToEulerAngles
import transformations as tf

#kitti_dir = '/home/nils/nils/kitti/data_odometry_gray/dataset/'
kitti_dir = '/media/localadmin/New Volume/11Nils/kitti/dataset/'
sequence = '02'
#velo_dir = '/home/nils/nils/kitti/dataset/sequences/'
velo_dir = '/media/localadmin/New Volume/11Nils/kitti/dataset/sequences/'

# https://github.com/hunse/kitti/blob/master/kitti/velodyne.py
def load_velodyne_points(drive,  frame):
    points_path = os.path.join(drive + '/velodyne/', "%06d.bin" % frame)
    points = np.fromfile(points_path, dtype=np.float32).reshape(-1, 4)
    points = points[:, :3]  # exclude luminance
    return points

def processPointCloud(img, pointcloud, pitch, roll, detections, div = 5):
    i = 0
    usedPoints = 0
    mean_z = 0
    height, width, channels = img.shape
    processedImg = np.zeros((height, width, 3), np.uint8)
    while i < len(pointcloud):
        x = pointcloud[i][0]
        y = pointcloud[i][1]
        z = pointcloud[i][2]
        # check if the x coordinate is in front of the camera at all
        if x > 0:
            # check if it's roughly in front of the camera
            if -25 < y < 25:
                # roughly check if it's higher than ground
                if -1 < z + 1.73:
                    pts = np.matmul(consts.P_L2C, np.array([x, y, z, 1]).reshape(4, 1))
                    #  [u v w]' = P * [X Y Z 1]'
                    if pts[2] != 0:

                        #         x = u / w
                        #         y = v / w
                        u = int(np.divide(pts[0], pts[2]))
                        v = int(np.divide(pts[1], pts[2]))
                        # TODO: recover groundplane from the point cloud instead of assuming planar driving

                        if 0 < u < width - 1 and 0 < v < height:
                            if not x == 0 or y == 0:
                                ground_depth = -1.73 - np.sin(pitch) / np.sqrt(x * x + y * y) - np.sin(roll) / y
                                if z > ground_depth + 0.4:
                                    if np.abs(detections[u // div, 0] - v) < 5 or detections[u // div, 0] == 0:
                                        detections[u // div, 0] = v
                                        detections[u // div, 1] += 1
                                        detections[u // div, 2] = 1 - 0.4 ** detections[u // div, 1]
                                    else:
                                        detections[u // div, 0] = v
                                        detections[u // div, 1] = 1
                                        detections[u // div, 2] = 1 - 0.4 ** detections[u // div, 1]


        i += 1
    mean_prob = 0
    for u in range(0, len(detections) - 1):
        mean_prob += detections[u, 2]
        v = int(detections[u, 0])
        if detections[u, 2] > 0.5:
            cv2.rectangle(processedImg, (u * div, v + 1), ((u + 1) * div, height), (255, 0, 0), 3)
            cv2.rectangle(processedImg, (u * div, v), ((u + 1) * div, 0), (0, 0, 255), 3)
    print(mean_prob / len(detections))
    return processedImg, detections



consts = Constants()

consts.readFileLists(kitti_dir, sequence)

image = cv2.imread(consts.image_path + consts.image_names[0])
#cv2.imshow("test", image)
print("\nimage shape: \n")
height, width, channels = image.shape
print(image.shape)

# point of IMU
pt = np.array((-.27 - 0.81, -.32, 0.93), dtype = np.float).reshape(3, 1)
pose_tf = np.eye(4, dtype = np.float)

consts.readTfLidarToCamera0(kitti_dir, sequence)

i = 0
j = 0
# look ahead
k = 40

# should result in turn of 30 deg around X-Axis
test_R = np.array([1, 0, 0, 0, np.sqrt(3) / 2, 0.5, 0, -0.5, np.sqrt(3) / 2]).reshape(3, 3)
print(test_R)
print(rotationMatrixToEulerAngles(test_R)*180 / 3.14159)
test = np.eye(4)
test[:3, :3] = test_R #* np.array([0, 0, 1, -1, 0, 0, 0, -1, 0]).reshape(3, 3)
print(test)
alpha, beta, gamma = tf.euler_from_matrix(test, 'rzyx')
print("\n zyx system \n")
print(alpha * 180 / 3.1415, beta * 180 / 3.1415, gamma * 180 / 3.1415)

detections = np.zeros((width // 5, 3), np.float)
q_C2Veh = [0.500, -0.500, 0.500, -0.500]

while i < len(consts.image_names) - 1:
    image = cv2.imread(consts.image_path + consts.image_names[i])
    j = i
    pt = np.array((0.0, 0.0, 0.0), dtype = np.float).reshape(3, 1)
    pose_chunk = np.eye(4, dtype = np.float)
    pose_chunk[:3, :4] = np.array(consts.poses[i]).reshape(3, 4)

    quat = tf.quaternion_from_matrix(pose_chunk)
    yaw, roll, pitch = tf.euler_from_quaternion(quat, 'ryzx') #ryzx or rxzy
    print(yaw * 180 / 3.1415, roll * 180 / 3.1415, pitch * 180 / 3.1415)

    last_detections = detections
    points = load_velodyne_points(velo_dir + sequence, i)
    labeled_image, detections = processPointCloud(image, points, pitch, roll, detections, 5)
    #labeled_image = np.zeros(image.shape, dtype = np.uint8)

    #'''
    for k in range(0, len(detections)):
        if detections[k, 1] == last_detections[k, 1]:
            if detections[k, 1] > 0:
                detections[k, 2] -= 0.4 ** detections[k, 1]
                detections[k, 1] -= 1
    #'''

    inv_image_pose = np.linalg.inv(pose_chunk)
    #test = np.matmul(consts.C2Veh, pose_chunk[:3, :3])
    #y, p, r = rotationMatrixToEulerAngles(pose_chunk[:3, :3])
    #print(y * 180 / 3.14159, p * 180 / 3.14159, r * 180 / 3.14159)

    pt_r_last = [-1, -1]
    pt_l_last = [-1, -1]
    while j < i + k:
        # concatenate the posetransformations first before multiplying with pt and K
        if j > len(consts.image_names) - 1:
            break
        I = np.eye(4)
        pose_chunk[:3, :4] = np.array(consts.poses[j]).reshape(3, 4)
        # store actual pose
        I = pose_chunk
        # right multiply actual pose with inverse of the image pose
        pose_chunk = np.matmul(inv_image_pose, pose_chunk)
        # reshaping pt to 4x1
        pt = np.array((0.0, 0.0, 0.0, 1), dtype = np.float).reshape(4, 1)

        # concatenating forward pose transformation to point
        pt = np.matmul(pose_chunk, pt)

        # projection into image coordinates
        uvw_l = np.matmul(consts.KRT_left, pt)
        uvw_r = np.matmul(consts.KRT_right, pt)
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

            if pt_in_image(pt_l_last, width, height) and pt_in_image(pt_r_last, width, height):
                if not pt_l_last == (-1, -1) or pt_r_last == (-1, -1):
                    rect = np.array([pt_l, pt_r, pt_r_last, pt_l_last], np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(labeled_image, [rect], (0, 255, 0))
            if j > i:
               pt_l_last = pt_l
               pt_r_last = pt_r

        j += 1
    vis = cv2.addWeighted(image, 1.0, labeled_image, 1.0, 0.0)
    cv2.imshow("gt", image)
    cv2.imshow("vis", vis)
    cv2.waitKey(100)
    i += 1
