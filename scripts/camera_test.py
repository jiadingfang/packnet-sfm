import torch
from packnet_sfm.geometry.camera import Camera
from packnet_sfm.geometry.camera_ucm import UCMCamera
from packnet_sfm.geometry.camera_eucm import EUCMCamera

K = torch.eye(3)
K[0,0] = 5
K[1,1] = 8
K[0,2] = 96
K[1,2] = 320
K = K.unsqueeze(0)

# I_ucm = torch.tensor([1,1,0,0,0])
I_ucm = torch.tensor([5,8,96,320,0])
I_ucm = I_ucm.unsqueeze(0)

# I_eucm = torch.tensor([1,1,0,0,0,1])
I_eucm = torch.tensor([5,8,96,320,0,1])
I_eucm = I_eucm.unsqueeze(0)

camera = Camera(K=K)
ucm_camera = UCMCamera(I=I_ucm)
eucm_camera = EUCMCamera(I=I_eucm)

x = torch.rand(1,3,1,1)
d = torch.rand(1,1,1,1)

print('projection')
print(camera.project(x))
print(ucm_camera.project(x))
print(eucm_camera.project(x))
print()
print('unprojection')
print(camera.reconstruct(d))
print(ucm_camera.reconstruct(d))
print(eucm_camera.reconstruct(d))