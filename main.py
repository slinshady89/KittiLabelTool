
import cv2
import numpy as np
import os
import pypcd

from argparser import argparser

from constants import Constants
from mathHelpers import pt_in_image, rotationMatrixToEulerAngles
import transformations as tf

#kitti_dir = '/home/nils/nils/kitti/data_odometry_gray/dataset/'
kitti_dir = '/media/localadmin/BigBerta/11Nils/kitti/dataset/'
sequence = '00'


# https://github.com/hunse/kitti/blob/master/kitti/velodyne.py
def load_velodyne_points(drive, sequence, frame):
    points_path = os.path.join(drive + "sequences/"  + sequence + '/velodyne/', "%06d.bin" % frame)
    points = np.fromfile(points_path, dtype=np.float32).reshape(-1, 4)
    points = points[:, :3]  # exclude luminance
    return points


# https://github.com/hunse/kitti/blob/master/kitti/velodyne.py
def load_velodyne_pcd(drive, sequence, frame):
    points_path = os.path.join(drive + "sequences/"  + sequence + '/velo_no_floor/', "%06d.pcd" % frame)
    pc = pypcd.PointCloud.from_path(points_path)
    points = np.array([np.array(xi) for xi in pc.pc_data])
    return points



def create_working_dir(args, folder_name):
    path = args.base_path + 'sequences/' + args.sequence + folder_name
    try:
        os.mkdir(path)
    except OSError as e:
        if e.errno != 17:
            print('OSError: %s', e)
    else:
        print("Successfully created the directory %s " % path)
    return path


def print2img(img, text, position):

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    fontColor = (255, 0, 255)
    lineType = 2

    cv2.putText(img, text,
                position,
                font,
                fontScale,
                fontColor,
                lineType)
    return img


def processPointCloud(img, pointcloud, coeffs,  detections, div = 5, consts = Constants()):
    i = 0
    usedPoints = 0
    mean_z = 0

    not_drivable_color = (0, 0, 255)
    height, width, _ = img.shape

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
                            # height of the lidar relative to the street on a plane
                            # if vehicle has pitch and roll angles these change the vehicle relative groundplane
                            # for very small lateral distances the roll influence is wrongly increased
                            # ground_depth = -1.73 - np.sin(pitch) / np.sqrt(x * x + y * y) - y * np.arctan(roll)
                            norm = np.sqrt(coeffs[0] * coeffs[0] + coeffs[1] * coeffs[1] + coeffs[2] * coeffs[2])
                            dist_to_ground_depth = (coeffs[0] * x + coeffs[1] * y + coeffs[2] * z + coeffs[3]) / norm
                            if np.abs(dist_to_ground_depth) > 0.3:
                                if u // div >= len(detections):
                                    u = div * (len(detections) - 1)
                                if detections[u // div, 0] == 0:
                                    detections[u // div, 0] = v
                                    detections[u // div, 1] += 1
                                    detections[u // div, 2] = 1 - 0.4 ** detections[u // div, 1]

                                if np.abs(detections[u // div, 0] - v) < 10:
                                    #detections[u // div, 0] = v
                                    detections[u // div, 1] += 1
                                    detections[u // div, 2] = 1 - 0.4 ** detections[u // div, 1]
                                else:
                                    detections[u // div, 0] = v
                                    detections[u // div, 1] = 1
                                    detections[u // div, 2] = 1 - 0.4 ** detections[u // div, 1]

        i += 1
    mean_prob = 0
    draw_left = True

    # draw rectangles from the lower left bounder of the left reflection until the column of the next reflection
    for u in range(0, len(detections) - 1):
        # accumulate probability of existence for every grid cell
        mean_prob += detections[u, 2]
        v = int(detections[u, 0])
        th_detect = 0.94  # > 1-0.4**4
        if detections[u, 2] > th_detect:
            left_refl = np.array([u * div, v + 1])
            right_refl = np.array([u * div, v + 1])

            # search for the next reflection which probability of existence is higher than th_detect (0.85)
            for m in range(u + 1, len(detections) - 1):
                if detections[m, 2] > th_detect:
                    right_refl = np.array([m * div, v + 1])
                    break
            # draw from left side of the image until the first reflection
            if draw_left:
                refl_rect = np.array([[0, 0], [left_refl[0], 0], left_refl, [0, left_refl[1]]], np.int32)
                cv2.fillPoly(img, [refl_rect], not_drivable_color)
                draw_left = False


            bottom_left = left_refl
            top_left = [left_refl[0], 0]
            bottom_right = right_refl
            top_right = [right_refl[0], 0]
            refl_rect = np.array([top_left, bottom_left, bottom_right, top_right], np.int32)
            cv2.fillPoly(img, [refl_rect], not_drivable_color)

    # fill the free space right of the most right reflection
    refl_rect = np.array([bottom_right, top_right, [width - 1, 0], [width - 1, bottom_right[0]]], np.int32)
    cv2.fillPoly(img, [refl_rect], not_drivable_color)

    # print mean probability of existence for all grid cells
    #print(mean_prob / len(detections))

    return img, detections


def main(args):
    label_path = create_working_dir(args, args.label_dir)
    # trafo_path = create_working_dir(args, '/trafos')

    consts = Constants()

    consts.readFileLists(args.base_path, args.sequence)

    image = cv2.imread(consts.image_path + consts.image_names[0])
    #cv2.imshow("test", image)
    # print("\nimage shape: \n")
    height, width, channels = image.shape
    # print(image.shape)

    # point of IMU
    pt = np.array((-.0, .06, 1.65), dtype = np.float).reshape(3, 1)
    pose_tf = np.eye(4, dtype = np.float)

    consts.readTfLidarToCamera0(args.base_path, args.sequence)

    i = 0
    # look ahead
    k = 100

    calc_path = True
    draw_path = True

    initial_pose = np.eye(4, dtype = np.float)
    initial_pose[:3, :4] = np.array(consts.poses[0]).reshape(3, 4)

    divisor = 5

    detections = np.zeros((width // divisor, 3), np.float)
    print("\nSequence %02d:\n" % int(args.sequence))

    while i < len(consts.image_names) - 1:

        plane_coeffs = np.loadtxt(kitti_dir + 'sequences/00/norms/%06d.txt' % i, dtype = np.float)
        image = cv2.imread(consts.image_path + consts.image_names[i])
        j = i
        pt = np.array((0.0, 0.0, 0.0), dtype = np.float).reshape(3, 1)
        pose_chunk = np.eye(4, dtype = np.float)
        pose_chunk[:3, :4] = np.array(consts.poses[i]).reshape(3, 4)
        movement = np.matmul(np.linalg.inv(initial_pose), pose_chunk)
        yaw, pitch, roll = tf.euler_from_matrix(movement, 'ryzx') #ryzx or rxzy
        #print(yaw * 180 / 3.1415, roll * 180 / 3.1415, pitch * 180 / 3.1415)


        h, w, c = image.shape
        blue_img = np.zeros((h, w, 3), np.uint8)

        blue_rect = np.array([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]], np.int32)
        cv2.fillPoly(blue_img, [blue_rect], (255, 0, 0))

        last_detections = detections
        points = load_velodyne_points(args.base_path, args.sequence, i+1)
        #labeled_image = np.zeros(image.shape, dtype = np.uint8)

        # aging of detections
        for l in range(0, len(detections)):
            if detections[l, 1] == last_detections[l, 1]:
                if detections[l, 1] > 0:
                    detections[l, 2] -= 0.4 ** detections[l, 1]
                    detections[l, 1] -= 1
                if detections[l, 1] == 0:
                    detections[l, 0] = 0
                    detections[l, 2] = 0.0


        inv_image_pose = np.linalg.inv(pose_chunk)

        pt_r_last = [-1, -1]
        pt_l_last = [-1, -1]
        if calc_path:
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
                if draw_path:
                    # projection into image coordinates
                    uvw_l = np.matmul(consts.KRT_left, pt)
                    uvw_r = np.matmul(consts.KRT_right, pt)
                    if uvw_l[2] != 0:
                        #         x = u / w
                        #         y = v / w
                        pt_l = (int(np.divide(uvw_l[0], uvw_l[2])), int(np.divide(uvw_l[1], uvw_l[2])))
                        pt_r = (int(np.divide(uvw_r[0], uvw_r[2])), int(np.divide(uvw_r[1], uvw_r[2])))

                        if pt_in_image(pt_l_last, width, height) and pt_in_image(pt_r_last, width, height):
                            if not (pt_l_last == (-1, -1) or pt_r_last == (-1, -1)):
                                rect = np.array([pt_l, pt_r, pt_r_last, pt_l_last], np.int32).reshape((-1, 1, 2))
                                cv2.fillPoly(blue_img, [rect], (0, 255, 0))
                        if j > i:
                           pt_l_last = pt_l
                           pt_r_last = pt_r
                j += 1

        # blue_img, detections = processPointCloud(blue_img, points, pitch, roll, detections, divisor, consts)
        blue_img, detections = processPointCloud(blue_img, points, plane_coeffs, detections, divisor, consts)
        img_name = os.path.join("%06d.png" % i)
        # print(label_path + img_name)
        # cv2.imwrite(label_path + img_name, blue_img)

        y, p, r = tf.euler_from_matrix(pose_chunk, 'ryzx') #ryzx or rxzy

        print('\r\033[1A\033[0K %d of %d' % (i, len(consts.image_names) - 1))
        # if args.visu == 1:
        ypr = 'yaw: %.3f | pitch: %.3f | roll: %.3f' % (y, p, r)

        vis = cv2.addWeighted(image, 1.0, blue_img, 1.0, 0.0)
        # print2img(vis, ypr, (20, 20))

        cv2.imshow('vis', vis)
        cv2.waitKey(10)

        i += 1


if __name__ == "__main__" :
    try:
        args = argparser()
        main(args)
    except KeyboardInterrupt:
        print("Finished")

