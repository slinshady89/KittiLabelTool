import numpy as np
from mathHelpers import isRotationMatrix, euler_to_quaternion_rad, eulerAnglesToRotationMatrixRad, quaternion_to_euler_rad
import os, fnmatch

class Constants():
    def __init__(self):

        ## quaternion rotation of coordinate system from world to camera system
        self.quats =[0.499, -0.499, 0.501, 0.501]

        ## quaternion rotation of coordinate system from world to camera system
        ## quaternion rotation of coordinate system from world to camera system
        #self.quats =[-0.5, 0.5, -0.5, 0.5]

        self.theta = np.array(quaternion_to_euler_rad(self.quats),dtype=np.float32).reshape(3,1)

        ## camera intrinsics matrix
        #     [fx'  0  cx']
        # K = [ 0  fy' cy']
        #     [ 0   0   1 ]
        self.K = np.array([335.639852470912, 0.0, 400.0, 0.0, 335.639852470912, 300.0, 0.0, 0.0, 1.0],dtype=np.float32).reshape(3,3)

        self.R = eulerAnglesToRotationMatrixRad(self.theta)
        print(self.R)

        ## Translation from lidar to camera coordinate system
        self.T = np.array([-0.02, -0.4,-2],dtype=np.float32).reshape(3,1)
        ## Translation for world-coord to camera
        #self.T = np.array([2.0, -0.4, 2.0],dtype=np.float32).reshape(3,1)

        self.RT = np.column_stack((self.R,self.T))
        self.KRT = np.matmul(self.K, self.RT)

        ## projection matrix
        #     [fx'  0  cx' Tx]
        # P = [ 0  fy' cy' Ty]
        #     [ 0   0   1   0]
        self.P = np.float32([335.639852470912, 0.0, 400.0, 0.0, 0.0, 335.639852470912, 300.0, 0.0, 0.0, 0.0, 1.0, 0.0]).reshape(3, 4)

        if os.path.exists('/home/localadmin/nils/imageseries/1/capture/'):
            self.img_path = '/home/localadmin/nils/imageseries/1/capture/'  # '/home/nils/nils/imageseries/1/capture/'
            self.label_path = '/home/localadmin/nils/imageseries/1/labels_backwards/'  # '/home/nils/nils/imageseries/1/labels/'
        else:
            self.img_path =  '/home/nils/nils/imageseries/1/capture/'
            self.label_path =  '/home/nils/nils/imageseries/1/labels/'


        self.image_names = []
        self.pcl_names = []


    def readFileLists(self):
        list_of_files = os.listdir(self.img_path)
        pattern_png = "*.png"


        image_chunk = []
        pcl_chunk = []
        pose_names = []

        ## find all images in the folder with corresponding point clouds and
        for entry in list_of_files:
            if fnmatch.fnmatch(entry, pattern_png):
                corresponding_pcl = entry.replace('.png', '.pcd')
                if corresponding_pcl in list_of_files:
                    pcl_chunk.append(corresponding_pcl)
                else:
                    continue
                image_chunk.append(entry)

        self.pcl_names = pcl_chunk.sort()
        self.image_names = image_chunk.sort()

        return pcl_chunk, image_chunk

    # TODO: make this working
    def getImageName(self, idx):
        return self.img_path + self.image_names[idx]

    def getPclName(self, idx):
        return self.img_path + self.pcl_names[idx]
