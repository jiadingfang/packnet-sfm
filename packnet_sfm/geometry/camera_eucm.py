# Copyright 2020 Toyota Research Institute.  All rights reserved.

from functools import lru_cache
import torch
import torch.nn as nn

from packnet_sfm.geometry.pose import Pose
from packnet_sfm.geometry.camera_utils import scale_intrinsics
from packnet_sfm.utils.image import image_grid

########################################################################################################################

class EUCMCamera(nn.Module):
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
        self.Tcw = Pose.identity(len(I)) if Tcw is None else Tcw

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
    def alpha(self):
        """alpha in EUCM model"""
        return self.I[:, 4].unsqueeze(1).unsqueeze(2)

    @property
    def beta(self):
        """beta in EUCM model"""
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
        fx, fy, cx, cy, alpha, beta = self.fx, self.fy, self.cx, self.cy, self.alpha, self.beta

        if torch.any(torch.isnan(alpha)):
            raise ValueError('alpha is nan')

        u = grid[:,0,:,:]
        v = grid[:,1,:,:]
        
        # print('u')
        # print(u.shape)
        # print('cx')
        # print(cx.shape)
        # print('fx')
        # print(fx.shape)
        # print('alpha')
        # print(alpha.shape)

        mx = (u - cx) / fx
        my = (v - cy) / fy
        r_square = mx ** 2 + my ** 2
        mz = (1 - beta * alpha ** 2 * r_square) / (alpha * torch.sqrt(1 - (2 * alpha - 1) * beta * r_square) + (1 - alpha))
        # xi = alpha / (1 - alpha)

        # coeff = (xi + torch.sqrt(1 + (1 - xi ** 2) * r_square)) / (1 + r_square)
        coeff = 1 / torch.sqrt(mx ** 2 + my ** 2 + mz ** 2)

        # print('fx, fy ,cx, cy, alpha, beta')
        # print(fx, fy, cx, cy, alpha, beta)
        # print(fx.requires_grad)
        # print('u')
        # print(torch.any(torch.isnan(u)))
        # print(u.requires_grad)
        # print('v')
        # print(torch.any(torch.isnan(v)))
        # print('mx')
        # print(mx.requires_grad)
        # print(torch.any(torch.isnan(mx)))
        # print('my')
        # print(torch.any(torch.isnan(my)))
        # print('r_square')
        # print(r_square.requires_grad)
        # print(torch.any(torch.isnan(r_square)))
        # print('xi')
        # print('xi')
        # print(xi.requires_grad)
        # print(torch.any(torch.isnan(xi)))
        # print('coeff')
        # print(coeff.requires_grad)
        # print(torch.any(torch.isnan(coeff)))
        
        x = coeff * mx
        y = coeff * my
        # z = coeff * 1 - xi
        z = coeff * mz
        z = z.clamp(min=1e-5)

        x_norm = x / z
        y_norm = y / z
        z_norm = z / z
        xnorm = torch.stack(( x_norm, y_norm, z_norm ), dim=1)

        # print('x')
        # print(torch.any(torch.isnan(x)))
        # print('y')
        # print(torch.any(torch.isnan(y)))
        # print('z')
        # print(torch.any(torch.isnan(z)))
        # print('x/z')
        # print(torch.any(torch.isnan(x/z)))
        # print('y/z')
        # print(torch.any(torch.isnan(y/z)))
        # print('z/z')
        # print(torch.any(torch.isnan(z/z)))
        # print('xnorm')
        # print(torch.any(torch.isnan(xnorm)))
        # print(x[torch.isnan(z/z)])
        # print(y[torch.isnan(z/z)])
        # print(z[torch.isnan(z/z)])
        # print(x_unnorm[torch.isnan(z/z)])
        # print(y_unnorm[torch.isnan(z/z)])
        # print(z_unnorm[torch.isnan(z/z)])

        # Scale rays to metric depth
        Xc = xnorm * depth

        # print('xnorm')
        # print(xnorm.requires_grad)

        # print('depth')
        # print(depth.requires_grad)

        # print('Xc')
        # print(Xc.shape)
        # # print(Xc.requires_grad)

        # print('Twc')
        # print(self.Twc.shape)
        # print(self.Twc.item())

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
        
        fx, fy, cx, cy, alpha, beta = self.fx, self.fy, self.cx, self.cy, self.alpha, self.beta
        x, y, z = X[:,0,:], X[:,1,:], X[:,2,:]
        # d = torch.norm(X, dim=1)
        d = torch.sqrt( beta * ( x ** 2 + y ** 2 ) + z ** 2 )
        z = z.clamp(min=1e-5)

        # print('X')
        # print(X.shape)
        # print('fx, fy, cx, cy, alpha, beta')
        # print(fx, fy, cx, cy, alpha, beta)
        # print('d')
        # print(d.shape)
        # # print(d.requires_grad)
        # print('fx')
        # print(fx.shape)
        # # print(fx.requires_grad)
        # print('z')
        # print(z.shape)
        # # print(z.requires_grad)
        
        Xnorm = fx * x / (alpha * d + (1 - alpha) * z) + cx
        Xnorm = 2 * Xnorm / (W) - 1
        Ynorm = fy * y / (alpha * d + (1 - alpha) * z) + cy
        Ynorm = 2 * Ynorm / (H) - 1

        # Return pixel coordinates
        return torch.stack([Xnorm, Ynorm], dim=-1).view(B, H, W, 2)
