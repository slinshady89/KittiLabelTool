import os
import argparse


#kitti_dir = '/home/nils/nils/kitti/data_odometry_gray/dataset/'
sequence = '00'

def argparser() :
    # command line argments
    parser = argparse.ArgumentParser(description = "KITTI Label Tool for semantic labeling with lidar, camera and "
                                                   "transformation data")
    parser.add_argument("--base_path",
                        default = '/media/localadmin/BigBerta/11Nils/kitti/dataset/',
                        help = "train list path")
    parser.add_argument("--sequence",
                        default = '00',
                        help = "Sequence in case of KITTI data")
    parser.add_argument("--label_dir",
                        default = '/labels/',
                        help = "folder to save labels in sequence")
    parser.add_argument("--visu",
                        default = 1,
                        type = int,
                        help = "0 if no visu"
                                "1 if visu")
    args = parser.parse_args()

    return args