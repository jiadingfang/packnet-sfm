import torch
from packnet_sfm.geometry.camera import Camera
from packnet_sfm.geometry.camera_ucm import UCMCamera

K = torch.eye(3)
K[0,0] = 5
K[1,1] = 8
K[0,2] = 96
K[1,2] = 320
K = K.unsqueeze(0)

# I = torch.tensor([1,1,0,0,0])
I = torch.tensor([5,8,96,320,0])
I = I.unsqueeze(0)

camera = Camera(K=K)
ucm_camera = UCMCamera(I=I)

x = torch.rand(1,3,1,1)
d = torch.rand(1,1,1,1)

print(camera.project(x))
print(ucm_camera.project(x))
print(camera.reconstruct(d))
print(ucm_camera.reconstruct(d))