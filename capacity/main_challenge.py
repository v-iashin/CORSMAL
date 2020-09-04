##################################################################################
# This work is an extension of LoDE code developed by Ricardo Sanchez Matilla
# (Email: ricardo.sanchezmatilla@qmul.ac.uk)
#        Author: Francesca Palermo
#         Email: f.palermo@qmul.ac.uk
#         Date: 2020/09/03
# Centre for Intelligent Sensing, Queen Mary University of London, UK
#
##################################################################################
# License
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0
# International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
##################################################################################


# System libs
import glob
import sys
import argparse

# Numeric libs
import cv2
import numpy as np
import torch
import torchvision

from numpy import linalg as LA

import shutil
import os

import pickle
import re
import csv

from tqdm import tqdm


# Computer Vision libs
from libs._3d.projection import *
from libs.detection.detection import imageSegmentation

# Additional Scripts
import utilities
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#object_set = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
#object_set = ['10', '11', '12']
object_set = ['10', '11', '12']

training_set = [1, 2, 3, 4, 5, 6, 7, 8, 9]
testing_set = [10, 11, 12]

modality_set = ['rgb', 'ir', 'depth']



camera_id_set = ['1', '2', '3', '4']

class LoDE:
    def __init__(self, args):

        self.args = args

        self.c1 = dict.fromkeys(['rgb', 'seg', 'intrinsic', 'extrinsic'])
        self.c2 = dict.fromkeys(['rgb', 'seg', 'intrinsic', 'extrinsic'])

        self.dataset_path = 'dataset'

        self.output_path = 'results'
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        #ADDED METHOD to extract the frames from the video
        print('Extract frames from bigger database')
        utilities.extract_frames(object_set, modality_set)

        f = open('{}/estimation.csv'.format(self.output_path), 'w', newline='')
        #f.write('fileName\theight[mm]\twidth[mm]\tcapacity[mL]\n')
        with f:
            writer = csv.writer(f)
            writer.writerow(['fileName','height[mm]','width[mm]','capacity[mL]'])
            #f.write('fileName\theight[mm]\twidth[mm]\tcapacity[mL]\n')
        f.close()

        # Load object detection model
        self.detectionModel = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        self.detectionModel.eval()
        self.detectionModel.cuda()

    def getObjectDimensions(self, file_id):

        f = open('{}/estimation.csv'.format(self.output_path), 'a+', newline='')
        try:
            centroid1, contour1 = getCentroid(self.c1['seg'])
            centroid2, contour2 = getCentroid(self.c2['seg'])

            centroid = cv2.triangulatePoints(self.c1['extrinsic']['projMatrix'], self.c2['extrinsic']['projMatrix'],
                                             centroid1, centroid2).transpose()
            centroid /= centroid[:, -1].reshape(-1, 1)
            centroid = centroid[:, :-1].reshape(-1)

            height, width, visualization, capacity = getObjectDimensions(self.c1, self.c2, centroid, self.args.draw)
            cv2.imwrite('{}/id{}_{}.png'.format(self.output_path, args.object, file_id), visualization)


            # f.write(
            #     'id{}_{}.png\t{:.2f}\t{:.2f}\t{:.2f}\n'.format(self.args.object,
            #                                            file_id,
            #                                               height,
            #                                               width,
            #                                               capacity))

            with f:
                writer = csv.writer(f)
                writer.writerow(['id{}_{}.png'.format(self.args.object,file_id), height , width, capacity])
                # f.write('fileName\theight[mm]\twidth[mm]\tcapacity[mL]\n')
            f.close()

            print('{}/id{}_{} ---- DONE'.format(self.output_path, args.object, file_id))
        except:
            # f.write(
            #     'id{}_{}.png\t0\t0\t0\n'.format(self.args.object,
            #                                                    file_id))
            with f:
                writer = csv.writer(f)
                writer.writerow(['id{}_{}.png'.format(self.args.object,file_id), '0' , '0', '749'])
                # f.write('fileName\theight[mm]\twidth[mm]\tcapacity[mL]\n')
            f.close()

            print('Error measuring id{}_{}'.format(self.args.object, file_id))
        f.close()

    def readData(self, image_path, file_id):
        # Read images from Camera 1
        image_c1_path = image_path+'/id{}_{}_c1_rgb.png'.format(args.object, file_id)

        if os.path.exists(image_c1_path):
            print('Image '  + image_c1_path + ' ---  FOUND' )
            self.c1['rgb'] = cv2.imread(image_c1_path)
            self.c1['seg'] = imageSegmentation(self.detectionModel, self.c1['rgb'])
        else:
            print('Image '  + image_c1_path + ' ---  MISSING')
            return

        # Read images from Camera 2
        image_c2_path = image_path+'/id{}_{}_c2_rgb.png'.format(args.object, file_id)

        if os.path.exists(image_c2_path):
            print('Image ' + image_c2_path + ' ---  FOUND')
            self.c2['rgb'] = cv2.imread(image_c2_path)
            self.c2['seg'] = imageSegmentation(self.detectionModel, self.c2['rgb'])
        else:
            print('Image ' + image_c2_path + ' ---  MISSING')
            return

    # Read calibration file for the chosen setup
    def readCalibration(self, calibration_path, file_id):
        calibration_c1_path = calibration_path+'/{}_c1_calib.pickle'.format(file_id)

        if not os.path.exists(calibration_c1_path):
            print('COMBINATION OF PARAMETERS FOR CALIBRATION DOES NOT EXISTS')
            return
        else:
            with open(calibration_c1_path, 'rb') as f:
                calibration = pickle.load(f, encoding="latin1")
                c1_intrinsic = calibration[0]
                c1_extrinsic = calibration[1]

        calibration_c2_path = calibration_path+'/{}_c2_calib.pickle'.format(file_id)

        if not os.path.exists(calibration_c2_path):
            print('COMBINATION OF PARAMETERS FOR CALIBRATION DOES NOT EXISTS')
            return
        else:
            with open(calibration_c2_path, 'rb') as f:
                calibration = pickle.load(f, encoding="latin1")
                c2_intrinsic = calibration[0]
                c2_extrinsic = calibration[1]

        self.c1['intrinsic'] = c1_intrinsic['rgb']
        self.c1['extrinsic'] = c1_extrinsic['rgb']
        self.c2['intrinsic'] = c2_intrinsic['rgb']
        self.c2['extrinsic'] = c2_extrinsic['rgb']

    def run(self):

        calibration_path = './dataset/calibration/{}'.format(args.object)
        image_path = './dataset/images/{}'.format(args.object)

        # path exists?
        if not os.path.isdir(calibration_path):
            print("Can't find path "+calibration_path)
            return

        # list all files in the calibration, so we will have a list
        # match their "id" part, to later find the same case in other folders
        file_pattern = r"([\w\d_]+)_c1_calib.pickle"
        file_id_list = [re.match(file_pattern, f).group(1) for f in os.listdir(calibration_path) if re.match(file_pattern, f)]

        for fid in file_id_list:
            # Read camera calibration files
            self.readCalibration(calibration_path, fid)
            # Main loop
            self.readData(image_path, fid)
            self.getObjectDimensions(fid)

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--object', type=int, default=0)
    parser.add_argument('--scenario', type=int, default=0)
    parser.add_argument('--filling_type', type=int, default=0)
    parser.add_argument('--filling_level', type=int, default=0)
    parser.add_argument('--background', type=int, default=0)
    parser.add_argument('--lighting', type=int, default=0)
    parser.add_argument('--draw', default=False, action='store_true', help='Output visual results in ./results')
    args = parser.parse_args()

    print('Executing...')
    lode = LoDE(args)
    for args.object in object_set:
        lode.run()
    print('Completed!')
