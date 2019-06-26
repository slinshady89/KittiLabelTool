import numpy as np
from mathHelpers import isRotationMatrix, euler_to_quaternion_rad, eulerAnglesToRotationMatrixRad, quaternion_to_euler_rad, add_ones
import os, fnmatch

class Constants():
    def __init__(self):

        ## quaternion rotation of coordinate system from world to camera system
        self.quats =[-0.5, 0.5, -0.5, 0.5]

        self.theta = np.array(quaternion_to_euler_rad(self.quats), dtype=np.float32).reshape(3,1)

        ## camera intrinsics matrix
        #     [fx'  0  cx']
        # K = [ 0  fy' cy']
        #     [ 0   0   1 ]
        self.K = np.array([718.856, 0.0, 607.1928, 0.0, 718.856, 185.2157, 0.0, 0.0, 1.0], dtype=np.float32).reshape(3,3)

        self.R = eulerAnglesToRotationMatrixRad(self.theta)

        ## Translation from left & right wheel to camera
        self.T_left = np.array([-1.68, 0.8, 1.65], dtype=np.float32).reshape(3,1)
        self.T_right = np.array([-1.68, -0.8, 1.65], dtype=np.float32).reshape(3,1)

        ## Transformationmatrices from left & right wheel to camera
        self.RT_left = np.column_stack((self.R,self.T_left))
        self.RT_right = np.column_stack((self.R,self.T_right))

        ## Projection matrices from left & right wheel to image coordinates
        self.KRT_left = np.matmul(self.K, self.RT_left)
        self.KRT_right = np.matmul(self.K, self.RT_right)

        print("Projection Matrices: \n")
        print("Left: \n")
        print(self.KRT_left)
        print("\nRight: \n")
        print(self.KRT_right)
        print("\n")

        ## projection matrix
        #     [fx'  0  cx' Tx]
        # P = [ 0  fy' cy' Ty]
        #     [ 0   0   1   0]
        self.P = np.float32([7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, 0.000000000000e+00,
                             0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02, 0.000000000000e+00,
                             0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00]).reshape(3, 4)

        self.image_names = []
        self.poses = []
        self.image_path = []



    def readFileLists(self, img_path, sequence):
        self.image_path = img_path + "sequences/" + sequence + "/image_0/"
        list_of_files = os.listdir(self.image_path)
        for image_name in list_of_files:
            self.image_names.append(image_name)

        self.image_names.sort()
        self.poses = np.loadtxt(img_path + "poses/" + sequence + ".txt")


    # TODO: make this working
    def getImageName(self, idx):
        return self.img_path + self.image_names[idx]

    def getPclName(self, idx):
        return self.img_path + self.pcl_names[idx]
