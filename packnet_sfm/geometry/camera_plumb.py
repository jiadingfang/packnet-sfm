# Copyright 2020 Toyota Research Institute.  All rights reserved.

from functools import lru_cache
import torch
import torch.nn as nn

from packnet_sfm.geometry.pose import Pose
from packnet_sfm.geometry.camera_utils import scale_intrinsics
from packnet_sfm.utils.image import image_grid

########################################################################################################################

class PlumbCamera(nn.Module):
    """
    Differentiable camera class implementing reconstruction and projection
    functions for a pinhole plumb model.
    """
    def __init__(self, I, Tcw=None):
        """
        Initializes the Camera class

        Parameters
        ----------
        I : torch.Tensor [5]
            Camera intrinsics parameter vector
        Tcw : Pose
            Camera -> World pose transformation
        """
        super().__init__()
        self.I = I
        self.Tcw = Pose.identity(len(I)) if Tcw is None else Tcw

    def __len__(self):
        """Batch size of the camera intrinsics"""
        return len(self.I)

    def to(self, *args, **kwargs):
        """Moves object to a specific device"""
        self.I = self.I.to(*args, **kwargs)
        self.Tcw = self.Tcw.to(*args, **kwargs)
        return self

    @staticmethod
    def calc_distortion(k1, k2, r):
        return 1 + k1 ** 2 + k2 ** 4

########################################################################################################################

    @property
    def fx(self):
        """Focal length in x"""
        return self.I[:, 0].unsqueeze(1).unsqueeze(2)

    @property
    def fy(self):
        """Focal length in y"""
        return self.I[:, 1].unsqueeze(1).unsqueeze(2)

    @property
    def cx(self):
        """Principal point in x"""
        return self.I[:, 2].unsqueeze(1).unsqueeze(2)

    @property
    def cy(self):
        """Principal point in y"""
        return self.I[:, 3].unsqueeze(1).unsqueeze(2)

    @property
    def k1(self):
        """k1 in radial distortion model"""
        return self.I[:, 4].unsqueeze(1).unsqueeze(2)

    @property
    def k2(self):
        """k2 in radial distortion model"""
        return self.I[:, 5].unsqueeze(1).unsqueeze(2)

    @property
    @lru_cache()
    def Twc(self):
        """World -> Camera pose transformation (inverse of Tcw)"""
        return self.Tcw.inverse()

########################################################################################################################

    def reconstruct(self, depth, frame='w'):
        """
        Reconstructs pixel-wise 3D points from a depth map.

        Parameters
        ----------
        depth : torch.Tensor [B,1,H,W]
            Depth map for the camera
        frame : 'w'
            Reference frame: 'c' for camera and 'w' for world

        Returns
        -------
        points : torch.tensor [B,3,H,W]
            Pixel-wise 3D points
        """
        B, C, H, W = depth.shape
        assert C == 1

        # Create flat index grid
        grid = image_grid(B, H, W, depth.dtype, depth.device, normalized=False)  # [B,3,H,W]
        flat_grid = grid.view(B, 3, -1)  # [B,3,HW]

        # Estimate the outward rays in the camera frame
        fx, fy, cx, cy, k1, k2 = self.fx, self.fy, self.cx, self.cy, self.k1, self.k2

        if torch.any(torch.isnan(k1)):
            raise ValueError('k1 is nan')

        u = grid[:,0,:,:]
        v = grid[:,1,:,:]

        xd = (u - cx) / fx
        yd = (v - cy) / fy

        rd = torch.sqrt(xd ** 2 + yd ** 2)
        r = rd
        # fixed point iteration
        for _ in range(4):
            r = rd / self.calc_distortion(k1, k1, r)

        x = xd / self.calc_distortion(k1, k2, r)
        y = yd / self.calc_distortion(k1, k2, r)
        z = torch.ones_like(x)

        xnorm = torch.stack(( x, y, z ), dim=1)

        # Scale rays to metric depth
        Xc = xnorm * depth

        # If in camera frame of reference
        if frame == 'c':
            return Xc
        # If in world frame of reference
        elif frame == 'w':
            return self.Twc @ Xc
        # If none of the above
        else:
            raise ValueError('Unknown reference frame {}'.format(frame))

    def project(self, X, frame='w'):
        """
        Projects 3D points onto the image plane

        Parameters
        ----------
        X : torch.Tensor [B,3,H,W]
            3D points to be projected
        frame : 'w'
            Reference frame: 'c' for camera and 'w' for world

        Returns
        -------
        points : torch.Tensor [B,H,W,2]
            2D projected points that are within the image boundaries
        """
        B, C, H, W = X.shape
        assert C == 3

        # Project 3D points onto the camera image plane
        if frame == 'c':
            # Xc = self.K.bmm(X.view(B, 3, -1))
            X = X
        elif frame == 'w':
            # Xc = self.K.bmm((self.Tcw @ X).view(B, 3, -1))
            X = (self.Tcw @ X)
        else:
            raise ValueError('Unknown reference frame {}'.format(frame))
        
        fx, fy, cx, cy, k1, k2 = self.fx, self.fy, self.cx, self.cy, self.k1, self.k2
        x, y, z = X[:,0,:], X[:,1,:], X[:,2,:]

        xu = x / z
        yu = y / z
        ru = torch.sqrt(xu ** 2 + yu ** 2)
        distortion = self.calc_distortion(k1, k2, ru) # radial distortion
        xd = xu * distortion
        yd = yu * distortion

        Xnorm = fx * xd + cx
        Xnorm = 2 * Xnorm / (W) - 1
        Ynorm = fy * yd + cy
        Ynorm = 2 * Ynorm / (H) - 1

        # Return pixel coordinates
        return torch.stack([Xnorm, Ynorm], dim=-1).view(B, H, W, 2)
