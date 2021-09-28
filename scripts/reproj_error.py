from packnet_sfm.geometry.camera_fov import FOVCamera
import torch
import cv2
import numpy as np
import os
import glob

from packnet_sfm.geometry.camera import Camera
from packnet_sfm.geometry.camera_ucm import UCMCamera
from packnet_sfm.geometry.camera_eucm import EUCMCamera
from packnet_sfm.geometry.camera_ds import DSCamera

from packnet_sfm.geometry.pose import Pose

# Define the dimensions of checkerboard
CHECKERBOARD = (7, 6)


# stop the iteration when specified
# accuracy, epsilon, is reached or
# specified number of iterations are completed.
criteria = (cv2.TERM_CRITERIA_EPS +
			cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# Vector for 3D points
threedpoints = []

# Vector for 2D points
twodpoints = []


# 3D points real world coordinates
objectp3d = np.zeros((1, CHECKERBOARD[0]
					* CHECKERBOARD[1],
					3), np.float32)
objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0],
							0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None


# Extracting path of individual image stored
# in a given directory. Since no path is
# specified, it will take current directory
# jpg files alone
data_dir = '/data/datasets/cam_checkerboard_tiny_384x256'
save_dir = '/data/datasets/cam_checkerboard_tiny_384x256'
data_dir = '/data/datasets/cam_checkerboard_tiny'
save_dir = '/data/datasets/cam_checkerboard_tiny'

images = glob.glob(os.path.join(data_dir, '*.png'))

resize_shape = (384, 256)

print('num of images')
print(len(images))

for filename in images:
    image = cv2.imread(filename)
    image = image[16:464, 25:727] # center crop to (702, 448)
    image = cv2.resize(image, resize_shape)
    # print(image.shape)

    grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    # If desired number of corners are
    # found in the image then ret = true
    ret, corners = cv2.findChessboardCorners(
                    grayColor, CHECKERBOARD,
                    cv2.CALIB_CB_ADAPTIVE_THRESH
                    + cv2.CALIB_CB_FAST_CHECK +
                    cv2.CALIB_CB_NORMALIZE_IMAGE)

    # If desired number of corners can be detected then,
    # refine the pixel coordinates and display
    # them on the images of checker board
    if ret == True:
        threedpoints.append(objectp3d)

        # Refining pixel coordinates
        # for given 2d points.
        corners2 = cv2.cornerSubPix(
            grayColor, corners, (11, 11), (-1, -1), criteria)

        twodpoints.append(corners2)

h, w = image.shape[:2]
print('h,w')
print(h,w)

# Perform camera calibration by
# passing the value of above found out 3D points (threedpoints)
# and its corresponding pixel coordinates of the
# detected corners (twodpoints)
ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(threedpoints, twodpoints, grayColor.shape[::-1], None, None)

poses = []

for i in range(len(r_vecs)):
    r_mat = cv2.Rodrigues(r_vecs[i])[0]
    t_vec = t_vecs[i][:,0]
    T = np.identity(4)
    T[:3,:3] = r_mat
    T[:3, 3] = t_vec
    # print('Transformation matrix')
    # print(T)
    # print(r_mat)
    # print(t_vec)
    T_tensor = torch.tensor(T)
    # print('pose type')
    # print(T_tensor.dtype)
    poses.append(Pose(T_tensor))

    # image_fn = images[i]
    # T_name = image_fn.split('/')[-1].split('.')[0] + '.npy'
    # save_path = os.path.join(save_dir, T_name)
    # np.save(save_path, T)


# Displaying required output
print(" Camera matrix:")
print(matrix)

print("\n Distortion coefficient:")
print(distortion)

print("\n Rotation Vectors:")
print(r_vecs)

print("\n Translation Vectors:")
print(t_vecs)


K = torch.tensor(matrix, dtype=torch.double)
K = K.unsqueeze(0)

# ucm_I = torch.tensor([235.4,  245.1,  186.5,  132.6,  0.650]) # gt
ucm_I = torch.tensor([236.1,  246.9,  178.3,  146.7,  0.635]) # learned
ucm_I = ucm_I.unsqueeze(0)

# eucm_I = torch.tensor([235.64381137951174, 245.38803860055288, 186.44431894063212, 132.64829510142745, 0.5966287792627975, 1.1122253956511319], dtype=torch.double)
eucm_I = torch.tensor([237.78, 247.71, 186.66, 129.09, 0.598, 1.075]) # learned
eucm_I = eucm_I.unsqueeze(0)

# fov_I = torch.tensor([218.73300148419895, 227.78847410828527, 186.54493477129685, 132.9030698368847, 0.9238658091225103]) # gt
fov_I = torch.tensor([222.5, 232.9, 187.9, 140.9, 0.91]) # learned
fov_I = fov_I.unsqueeze(0)

# ds_I = torch.tensor([183.9, 191.5, 186.7, 132.8, -0.208, 0.560]) # gt
ds_I = torch.tensor([187.2, 195.9, 188.6, 138.9, -0.227, 0.569]) # learned
ds_I = ds_I.unsqueeze(0)

opencv_mean_error = 0
pinhole_mean_error = 0
ucm_mean_error = 0
eucm_mean_error = 0
fov_mean_error = 0
ds_mean_error = 0

# print('numer of 3d points')
# print(len(threedpoints))

# Gordon's
# matrix = np.array([[250.2, 0, 187.2], [0, 261.3, 132.8], [0,0,1]])
# distortion = np.array([[-0.283, 0.074, 0.0, 0.0, 0.0]]) 

# distortion = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])  # debugging

for i in range(len(threedpoints)):
# for i in range(1):

    # print('2d points shape')
    # print(twodpoints[i].shape)
    # print(twodpoints[i])

    imgpoints2, _ = cv2.projectPoints(threedpoints[i], r_vecs[i], t_vecs[i], matrix, distortion) # opencv calib
    # opencv_error = cv2.norm(twodpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    opencv_error = np.linalg.norm(twodpoints[i] - imgpoints2) / len(imgpoints2)
    opencv_mean_error += opencv_error

    # print('imgpoints2')
    # print(imgpoints2.shape)
    # print(imgpoints2)

    pose = poses[i]
    # pinhole_cam = Camera(K=K, Tcw=pose.inverse())
    # eucm_cam = EUCMCamera(I=eucm_I, Tcw=pose.inverse())
    pinhole_cam = Camera(K=K, Tcw=pose)
    ucm_cam = UCMCamera(I=ucm_I, Tcw=pose)
    eucm_cam = EUCMCamera(I=eucm_I, Tcw=pose)
    fov_cam = FOVCamera(I=fov_I, Tcw=pose)
    ds_cam = DSCamera(I=ds_I, Tcw=pose)

    # print('3d points shape')
    # print(threedpoints[i].shape)

    threedpoints_tensor = torch.tensor(threedpoints[i].T, dtype=torch.double).unsqueeze(0)

    # print('3d points tensor shape')
    # print(threedpoints_tensor.shape)
    # print('3d points tensor type')
    # print(threedpoints_tensor.dtype)

    pinhole_imgpoints2 = pinhole_cam.project(threedpoints_tensor).squeeze(0).numpy()
    ucm_imgpoints2 = ucm_cam.project(threedpoints_tensor).squeeze(0).numpy()
    eucm_imgpoints2 = eucm_cam.project(threedpoints_tensor).squeeze(0).numpy()
    fov_imgpoints2 = fov_cam.project(threedpoints_tensor).squeeze(0).numpy()
    ds_imgpoints2 = ds_cam.project(threedpoints_tensor).squeeze(0).numpy()
    
    # print('eucm_imgpoints2 shape')
    # print(eucm_imgpoints2.shape)
    # print('twodpoints[i] shape')
    # print(twodpoints[i].shape)
    # print('eucm_imgpoints2')
    # print(eucm_imgpoints2)
    # print('pinhole_imgpoints2')
    # print(pinhole_imgpoints2)

    # eucm_error = cv2.norm(twodpoints[i], eucm_imgpoints2, cv2.NORM_L2)/len(eucm_imgpoints2)
    pinhole_error = np.linalg.norm(twodpoints[i] - pinhole_imgpoints2) / len(pinhole_imgpoints2)
    ucm_error = np.linalg.norm(twodpoints[i] - ucm_imgpoints2) / len(ucm_imgpoints2)
    eucm_error = np.linalg.norm(twodpoints[i] - eucm_imgpoints2) / len(eucm_imgpoints2)
    fov_error = np.linalg.norm(twodpoints[i] - fov_imgpoints2) / len(fov_imgpoints2)
    ds_error = np.linalg.norm(twodpoints[i] - ds_imgpoints2) / len(ds_imgpoints2)

    pinhole_mean_error += pinhole_error
    ucm_mean_error += ucm_error
    eucm_mean_error += eucm_error
    fov_mean_error += fov_error
    ds_mean_error += ds_error

    # print('r_vecs[i]')
    # print(r_vecs[i])
    # print('t_vecs[i]')
    # print(t_vecs[i])
    # print('pose')
    # print(pose.mat)
    

print( "total opencv error: {}".format(opencv_mean_error/len(threedpoints)))
print( "total pinhole error: {}".format(pinhole_mean_error/len(threedpoints)))
print( "total ucm error: {}".format(ucm_mean_error/len(threedpoints)) )
print( "total eucm error: {}".format(eucm_mean_error/len(threedpoints)))
print( "total fov error: {}".format(fov_mean_error/len(threedpoints)))
print( "total ds error: {}".format(ds_mean_error/len(threedpoints)))
