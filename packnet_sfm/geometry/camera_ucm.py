# Copyright 2020 Toyota Research Institute.  All rights reserved.

from functools import lru_cache
import torch
import torch.nn as nn

from packnet_sfm.geometry.pose import Pose
from packnet_sfm.geometry.camera_utils import scale_intrinsics
from packnet_sfm.utils.image import image_grid

########################################################################################################################

class UCMCamera(nn.Module):
    """
    Differentiable camera class implementing reconstruction and projection
    functions for a pinhole model.
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
        self.Tcw = Pose.identity(1) if Tcw is None else Tcw

    def __len__(self):
        """Batch size of the camera intrinsics"""
        return len(self.I)

    def to(self, *args, **kwargs):
        """Moves object to a specific device"""
        self.I = self.I.to(*args, **kwargs)
        self.Tcw = self.Tcw.to(*args, **kwargs)
        return self

########################################################################################################################

    @property
    def fx(self):
        """Focal length in x"""
        return self.I[:, 0]

    @property
    def fy(self):
        """Focal length in y"""
        return self.I[:, 1]

    @property
    def cx(self):
        """Principal point in x"""
        return self.I[:, 2]

    @property
    def cy(self):
        """Principal point in y"""
        return self.I[:, 3]

    @property
    def alpha(self):
        """alpha in UCM model"""
        return self.I[:, 4]

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
        fx, fy, cx, cy, alpha = self.fx, self.fy, self.cx, self.cy, self.alpha
        assert abs(fx) > 1e-5
        assert abs(fy) > 1e-5
        assert abs(alpha) > 1e-5
        assert abs(1-alpha) > 1e-5

        u = grid[:,0,:,:]
        v = grid[:,1,:,:]

        mx = (u - cx) / fx * (1 - alpha)
        my = (v - cy) / fy * (1 - alpha)
        r_square = mx ** 2 + my ** 2
        xi = alpha / (1 - alpha)
        coeff = (xi + torch.sqrt(1 + (1 - xi ** 2) * r_square)) / (1 + r_square)
        
        x = coeff * mx
        y = coeff * my
        z = coeff * 1 - xi
        xnorm = torch.stack((x,y,z), dim=1)

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
            X = X.view(B, 3, -1)
        elif frame == 'w':
            # Xc = self.K.bmm((self.Tcw @ X).view(B, 3, -1))
            X = (self.Tcw @ X).view(B, 3, -1)
        else:
            raise ValueError('Unknown reference frame {}'.format(frame))
        
        # TODO: check 3d points for valid projection

        d = torch.norm(X, dim=1)
        x, y, z = X[:,0,:], X[:,1,:], X[:,2,:]
        fx, fy, cx, cy, alpha = self.fx, self.fy, self.cx, self.cy, self.alpha

        # print('d = {}'.format(d))
        # print('x = {}'.format(x))
        # print('y = {}'.format(y))
        # print('z = {}'.format(z))

        # print('fx = {}'.format(fx))
        # print('fy = {}'.format(fy))
        # print('cx = {}'.format(cx))
        # print('cy = {}'.format(cy))
        # print('alpha = {}'.format(alpha))

        assert abs(fx) > 1e-5
        assert abs(fy) > 1e-5
        assert abs(alpha) > 1e-5
        assert abs(1-alpha) > 1e-5
        
        Xnorm = fx * x / (alpha * d + (1 - alpha) * z) + cx
        Ynorm = fy * y / (alpha * d + (1 - alpha) * z) + cy

        assert abs(alpha * d + (1 - alpha) * z) > 1e-5

        # Clamp out-of-bounds pixels
        # Xmask = ((Xnorm > 1) + (Xnorm < -1)).detach()
        # Xnorm[Xmask] = 2.
        # Ymask = ((Ynorm > 1) + (Ynorm < -1)).detach()
        # Ynorm[Ymask] = 2.

        # Return pixel coordinates
        return torch.stack([Xnorm, Ynorm], dim=-1).view(B, H, W, 2)
