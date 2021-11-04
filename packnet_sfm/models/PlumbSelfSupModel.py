# Copyright 2020 Toyota Research Institute.  All rights reserved.

from packnet_sfm.models.PlumbSfmModel import PlumbSfmModel
from packnet_sfm.losses.plumb_multiview_photometric_loss import PlumbMultiViewPhotometricLoss
from packnet_sfm.models.model_utils import merge_outputs

import numpy as np
from PIL import Image

class PlumbSelfSupModel(PlumbSfmModel):
    """
    Model that inherits a depth and pose network from SfmModel and
    includes the photometric loss for self-supervised training.

    Parameters
    ----------
    kwargs : dict
        Extra parameters
    """
    def __init__(self, **kwargs):
        # Initializes SfmModel
        super().__init__(**kwargs)
        self.counter = 0
        # Initializes the photometric loss
        self._photometric_loss = PlumbMultiViewPhotometricLoss(**kwargs)

    @property
    def logs(self):
        """Return logs."""
        return {
            **super().logs,
            **self._photometric_loss.logs,
            **self.I_dict
        }

    def self_supervised_loss(self, image, ref_images, inv_depths, poses,
                             intrinsics, return_logs=False, progress=0.0):
        """
        Calculates the self-supervised photometric loss.

        Parameters
        ----------
        image : torch.Tensor [B,3,H,W]
            Original image
        ref_images : list of torch.Tensor [B,3,H,W]
            Reference images from context
        inv_depths : torch.Tensor [B,1,H,W]
            Predicted inverse depth maps from the original image
        poses : list of Pose
            List containing predicted poses between original and context images
        intrinsics : torch.Tensor [B,3,3]
            Camera intrinsics
        return_logs : bool
            True if logs are stored
        progress :
            Training progress percentage

        Returns
        -------
        output : dict
            Dictionary containing a "loss" scalar a "metrics" dictionary
        """
        return self._photometric_loss(
            image, ref_images, inv_depths, intrinsics, intrinsics, poses,
            return_logs=return_logs, progress=progress)

    def forward(self, batch, return_logs=False, progress=0.0):
        """
        Processes a batch.

        Parameters
        ----------
        batch : dict
            Input batch
        return_logs : bool
            True if logs are stored
        progress :
            Training progress percentage

        Returns
        -------
        output : dict
            Dictionary containing a "loss" scalar and different metrics and predictions
            for logging and downstream usage.
        """
        # Calculate predicted depth and pose output
        output = super().forward(batch, return_logs=return_logs)

        I = output['intrinsics'][0,:]
        self.I_dict = {'fx': I[0].item(), 'fy': I[1].item(), 'cx': I[2].item(), 'cy': I[3].item(), 'k1': I[4].item(), 'k2': I[5].item()}
        batch_I = output['intrinsics']

        # I_0 = output['intrinsics'][0,0,:]
        # I_1 = output['intrinsics'][0,1,:]
        # self.I_dict = {'fx_0': I_0[0].item(), 'fy_0': I_0[1].item(), 'cx_0': I_0[2].item(), 'cy_0': I_0[3].item(), 'alpha_0': I_0[4].item(), 'beta_0' : I_0[5].item(),
        #                 'fx_1': I_1[0].item(), 'fy_1': I_1[1].item(), 'cx_1': I_1[2].item(), 'cy_1': I_1[3].item(), 'alpha_1': I_1[4].item(), 'beta_1': I_1[5].item()}
        
        # print('batch intrinsic types')
        # print(batch['intrinsic_type'])
        # print(batch.keys())

        # B = len(batch['idx'])
        # batch_I = torch.zeros((B,5))
        # for i in range(B):
        #     if batch['intrinsic_type'][i] == 'euroc':
        #         batch_I[i, :] = I_0
        #     elif batch['intrinsic_type'][i] == 'omnicam':
        #         batch_I[i, :] = I_1
        #     else:
        #         raise ValueError('only implement for euroc and omnicam')

        # B = len(batch['idx'])
        # batch_I = torch.zeros((B,5))
        # for i in range(B):
        #     if batch['intrinsic_type'][i] == 'euroc':
        #         batch_I[i, :] = I_0
        #     elif batch['intrinsic_type'][i] == 'kitti':
        #         batch_I[i, :] = I_1
        #     else:
        #         raise ValueError('only implement for euroc and kitti')
        
        # print('batch_I')
        # print(batch_I)

        if self.counter % 100 == 0:
            print()
            print(I)
            # print(batch['intrinsics'])
            # print(I_0)
            # print(I_1)
            # print(batch_I)
        self.counter += 1

        if not self.training:
            # If not training, no need for self-supervised loss
            return output
        else:
            # Otherwise, calculate self-supervised loss
            # self_sup_output = self.self_supervised_loss(
            #     batch['rgb_original'], batch['rgb_context_original'],
            #     output['inv_depths'], output['poses'], batch_I,
            #     return_logs=return_logs, progress=progress)

            self_sup_output = self.self_supervised_loss(
                batch['rgb'], batch['rgb_context'],
                output['inv_depths'], output['poses'], batch_I,
                return_logs=return_logs, progress=progress)
                
            # Return loss and metrics
            return {
                'loss': self_sup_output['loss'],
                **merge_outputs(output, self_sup_output),
            }