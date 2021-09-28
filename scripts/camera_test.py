import torch
from packnet_sfm.geometry.camera import Camera
from packnet_sfm.geometry.camera_ucm import UCMCamera
from packnet_sfm.geometry.camera_eucm import EUCMCamera
from packnet_sfm.geometry.camera_ds import DSCamera

K = torch.eye(3)
K[0,0] = 9.037596e+02
K[1,1] = 9.019653e+02
K[0,2] = 6.957519e+02
K[1,2] = 2.242509e+02
K = K.unsqueeze(0)

# I_ucm = torch.tensor([1,1,0,0,0])
I_ucm = torch.tensor([9.037596e+02, 9.019653e+02, 6.957519e+02, 2.242509e+02, 0.5])
I_ucm = I_ucm.unsqueeze(0)

# I_eucm = torch.tensor([1,1,0,0,0,1])
I_eucm = torch.tensor([9.037596e+02, 9.019653e+02, 6.957519e+02, 2.242509e+02, 0.5, 1])
I_eucm = I_eucm.unsqueeze(0)

I_ds = torch.tensor([9.037596e+02, 9.019653e+02, 6.957519e+02, 2.242509e+02, 0, 0.5])
I_ds = I_ds.unsqueeze(0)

camera = Camera(K=K)
ucm_camera = UCMCamera(I=I_ucm)
eucm_camera = EUCMCamera(I=I_eucm)
ds_camera = DSCamera(I=I_ds)

x = torch.rand(1,3,1,1)
d = torch.rand(1,1,1,1)

print('projection')
# print(camera.project(x))
print(ucm_camera.project(x))
print(eucm_camera.project(x))
print(ds_camera.project(x))
print()
print('unprojection')
# print(camera.reconstruct(d))
print(ucm_camera.reconstruct(d))
print(eucm_camera.reconstruct(d))
print(ds_camera.reconstruct(d))