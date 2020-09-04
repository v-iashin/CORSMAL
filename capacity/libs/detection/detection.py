################################################################################## 
#        Author: Ricardo Sanchez Matilla
#         Email: ricardo.sanchezmatilla@qmul.ac.uk
#  Created Date: 2020/02/13
# Modified Date: 2020/02/28

# Centre for Intelligent Sensing, Queen Mary University of London, UK
# 
################################################################################## 
# License
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0
# International License. To view a copy of this license, visit 
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to 
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
##################################################################################
#
import numpy as np
import copy
import cv2

import torch
import torchvision.transforms as T

device = 'cuda' if torch.cuda.is_available() else 'cpu'

trf = T.Compose([T.ToPILImage(), T.ToTensor()])

def imageSegmentation(detectionModel, _img):
	labels_to_avoid = [1, 15, 62, 63, 67, 68, 69, 77]
	img = copy.deepcopy(_img)
	img = img[:, :, [2, 1, 0]]

	output = detectionModel([trf(img).to(device)])[0]
	seg = None
	for i in range(0, len(output['labels'])):
		#TODO. This is the original code. It is quite dificult ot reapply this kind of coding -- https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
		if output['scores'][i] >= 0.5 and output['labels'][i] in [44, 46, 47, 84, 86]:	#bottle, wine glass, cup, book, vase
			seg = np.uint8(255.*output['masks'][i,:,:,:].detach().cpu().numpy().squeeze())
			seg = ((seg >= 128)*255).astype(np.uint8)
			break
	# Return the most confident
	return seg