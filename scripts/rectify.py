import torch
import numpy as np
import cv2

from packnet_sfm.geometry.camera import Camera
from packnet_sfm.geometry.camera_ucm import UCMCamera
from packnet_sfm.geometry.camera_eucm import EUCMCamera
from packnet_sfm.geometry.camera_ds import DSCamera

from packnet_sfm.utils.rectify import to_perspective
from packnet_sfm.datasets.augmentations import resize_image, to_tensor, center_crop_image

def center_crop(img, dim):
	"""Returns center cropped image
	Args:
	img: image to be center cropped
	dim: dimensions (width, height) to be cropped
	"""
	width, height = img.shape[1], img.shape[0]

	# process crop width and height for max available dimension
	crop_height = dim[0] if dim[0]<img.shape[1] else img.shape[1]
	crop_width = dim[1] if dim[1]<img.shape[0] else img.shape[0] 
	mid_x, mid_y = int(width/2), int(height/2)
	cw2, ch2 = int(crop_width/2), int(crop_height/2) 
	crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
	return crop_img

if __name__=='__main__':
    # ucm_I = torch.tensor([235.4,  245.1,  186.5,  132.6,  0.650]) # gt
    # ucm_I = torch.tensor([236.1,  246.9,  178.3,  146.7,  0.635]) # learned
    # ucm_I = torch.tensor([112.5976,  93.0813,  199.6491,  195.1849,  0.5])
    # ucm_I = torch.tensor([320.0, 305.0, 318.0, 277.0, 0.52])
    ucm_I = torch.tensor([320.0, 305.0, 320.0, 240.0, 0.52])
    ucm_I = ucm_I.unsqueeze(0)

    # eucm_I = torch.tensor([235.64381137951174, 245.38803860055288, 186.44431894063212, 132.64829510142745, 0.5966287792627975, 1.1122253956511319], dtype=torch.double)
    # eucm_I = torch.tensor([237.78, 247.71, 186.66, 129.09, 0.598, 1.075]) # learned
    # eucm_I = eucm_I.unsqueeze(0)

    # ds_I = torch.tensor([183.9, 191.5, 186.7, 132.8, -0.208, 0.560]) # gt
    # # ds_I = torch.tensor([187.2, 195.9, 188.6, 138.9, -0.227, 0.569]) # learned
    # ds_I = ds_I.unsqueeze(0)

    ucm_cam = UCMCamera(I=ucm_I)
    # eucm_cam = EUCMCamera(I=eucm_I)
    # ds_cam = DSCamera(I=ds_I)
    
    # img = cv2.imread('scripts/MH_01.png')
    img = cv2.imread('/data/datasets/tri_fisheye_01/000000000.png')
    center_crop_shape = (720, 960)
    img = center_crop(img, center_crop_shape)
    print(img.shape)
    img = cv2.resize(img, (480,640))
    out = to_perspective(ucm_cam, img, img_size=(480,640), f=0.25)
    cv2.imwrite('outs/test_tss.png', out)


# threedpoints_tensor = torch.tensor(threedpoints[i].T, dtype=torch.double).unsqueeze(0)
# ucm_imgpoints2 = ucm_cam.project(threedpoints_tensor).squeeze(0).numpy()
# eucm_imgpoints2 = eucm_cam.project(threedpoints_tensor).squeeze(0).numpy()
# ds_imgpoints2 = ds_cam.project(threedpoints_tensor).squeeze(0).numpy()