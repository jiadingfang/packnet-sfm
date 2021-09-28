import torch
import numpy as np
import cv2

from packnet_sfm.geometry.camera import Camera
from packnet_sfm.geometry.camera_ucm import UCMCamera
from packnet_sfm.geometry.camera_eucm import EUCMCamera
from packnet_sfm.geometry.camera_ds import DSCamera

from packnet_sfm.utils.rectify import to_perspective

if __name__=='__main__':
    # ucm_I = torch.tensor([235.4,  245.1,  186.5,  132.6,  0.650]) # gt
    # ucm_I = torch.tensor([236.1,  246.9,  178.3,  146.7,  0.635]) # learned
    ucm_I = torch.tensor([112.5976,  93.0813,  199.6491,  195.1849,  0.5])
    ucm_I = ucm_I.unsqueeze(0)

    # eucm_I = torch.tensor([235.64381137951174, 245.38803860055288, 186.44431894063212, 132.64829510142745, 0.5966287792627975, 1.1122253956511319], dtype=torch.double)
    eucm_I = torch.tensor([237.78, 247.71, 186.66, 129.09, 0.598, 1.075]) # learned
    eucm_I = eucm_I.unsqueeze(0)

    ds_I = torch.tensor([183.9, 191.5, 186.7, 132.8, -0.208, 0.560]) # gt
    # ds_I = torch.tensor([187.2, 195.9, 188.6, 138.9, -0.227, 0.569]) # learned
    ds_I = ds_I.unsqueeze(0)

    ucm_cam = UCMCamera(I=ucm_I)
    eucm_cam = EUCMCamera(I=eucm_I)
    ds_cam = DSCamera(I=ds_I)
    
    # img = cv2.imread('scripts/MH_01.png')
    img = cv2.imread('/data/datasets/omnicam/session1/000000000.png')
    print('image')
    print(img.shape)
    img = cv2.resize(img, (384,384))
    out = to_perspective(ucm_cam, img, img_size=(384,384), f=0.25)
    cv2.imwrite('scripts/test_omnicam.png', out)


# threedpoints_tensor = torch.tensor(threedpoints[i].T, dtype=torch.double).unsqueeze(0)
# ucm_imgpoints2 = ucm_cam.project(threedpoints_tensor).squeeze(0).numpy()
# eucm_imgpoints2 = eucm_cam.project(threedpoints_tensor).squeeze(0).numpy()
# ds_imgpoints2 = ds_cam.project(threedpoints_tensor).squeeze(0).numpy()