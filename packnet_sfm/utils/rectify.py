import torch
import numpy as np
import cv2

def warp_img(img, img_pts, valid_mask):
    # Remap
    img_pts = img_pts.astype(np.float32)
    out = cv2.remap(
        img, img_pts[..., 0], img_pts[..., 1], cv2.INTER_LINEAR
    )
    # out[~valid_mask] = 0.0
    return out

def to_perspective(cam, img, img_size=(512, 512), f=0.25):
    # Generate 3D points
    h, w = img_size
    z = f * min(img_size)
    x = np.arange(w) - w / 2
    y = np.arange(h) - h / 2
    x_grid, y_grid = np.meshgrid(x, y, indexing="xy")
    point3D = np.stack([x_grid, y_grid, np.full_like(x_grid, z)], axis=-1)
    point3D = np.moveaxis(point3D, -1, 0)
    point3D_tensor = torch.tensor(point3D, dtype=torch.float).unsqueeze(0)

    # Project on image plane
    img_pts = cam.project(point3D_tensor).squeeze(0).numpy() # [B,H,W,2]
    img_pts[:,:,0] = (img_pts[:,:,0] + 1) * w / 2 # unnormalize
    img_pts[:,:,1] = (img_pts[:,:,1] + 1) * h / 2
    valid_mask = np.full_like(x_grid, True) # all true mask for now
    out = warp_img(img, img_pts, valid_mask)
    return out