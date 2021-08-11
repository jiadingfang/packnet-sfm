# Copyright 2020 Toyota Research Institute.  All rights reserved.

from packnet_sfm.models.UCMSfmModel import UCMSfmModel
# from packnet_sfm.losses.multiview_photometric_loss import MultiViewPhotometricLoss
from packnet_sfm.losses.ucm_multiview_photometric_loss import UCMMultiViewPhotometricLoss
from packnet_sfm.models.model_utils import merge_outputs

import numpy as np
from PIL import Image

class UCMSelfSupModel(UCMSfmModel):
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
        self._photometric_loss = UCMMultiViewPhotometricLoss(**kwargs)

    @property
    def logs(self):
        """Return logs."""
        return {
            **super().logs,
            **self._photometric_loss.logs
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
        # exp_name = 'omnicam_128x128'
        # exp_name = 'omnicam_384x384'
        # print()
        
        if self.counter % 100 == 0:
            print()
            print(output['intrinsics'][0])
            # print(batch['intrinsics'][0])
            # inv_depths = output['inv_depths']
            # print(output['intrinsics'])
            # print('saving inv_depths')
            # # print(len(inv_depths))
            # # print(inv_depths)
            # # print(inv_depths[0].shape)

            # img_save_path_0 = 'results/depths_{}/png/{}_0.png'.format(exp_name, self.counter)
            # img_save_path_1 = 'results/depths_{}/png/{}_1.png'.format(exp_name, self.counter)
            # img_save_path_2 = 'results/depths_{}/png/{}_2.png'.format(exp_name, self.counter)
            # img_save_path_3 = 'results/depths_{}/png/{}_3.png'.format(exp_name, self.counter)

            # inv_depth_0_copy = inv_depths[0].clone()
            # raw_inv_depth_0 = inv_depth_0_copy.detach().cpu().numpy()[0][0]
            # inv_depth_0 = (raw_inv_depth_0 - np.min(raw_inv_depth_0)) / np.max(raw_inv_depth_0) * 255
            # inv_depth_0 = inv_depth_0.astype(np.uint8)

            # inv_depth_1_copy = inv_depths[1].clone()
            # raw_inv_depth_1 = inv_depth_1_copy.detach().cpu().numpy()[0][0]
            # inv_depth_1 = (raw_inv_depth_1 - np.min(raw_inv_depth_1)) / np.max(raw_inv_depth_1) * 255
            # inv_depth_1 = inv_depth_1.astype(np.uint8)

            # inv_depth_2_copy = inv_depths[2].clone()
            # raw_inv_depth_2 = inv_depth_2_copy.detach().cpu().numpy()[0][0]
            # inv_depth_2 = (raw_inv_depth_2 - np.min(raw_inv_depth_2)) / np.max(raw_inv_depth_2) * 255
            # inv_depth_2 = inv_depth_2.astype(np.uint8)

            # inv_depth_3_copy = inv_depths[3].clone()
            # raw_inv_depth_3 = inv_depth_3_copy.detach().cpu().numpy()[0][0]
            # inv_depth_3 = (raw_inv_depth_3 - np.min(raw_inv_depth_3)) / np.max(raw_inv_depth_3) * 255
            # inv_depth_3 = inv_depth_3.astype(np.uint8)

            # im_0 = Image.fromarray(inv_depth_0)
            # im_1 = Image.fromarray(inv_depth_1)
            # im_2 = Image.fromarray(inv_depth_2)
            # im_3 = Image.fromarray(inv_depth_3)

            # im_0.save(img_save_path_0)
            # im_1.save(img_save_path_1)
            # im_2.save(img_save_path_2)
            # im_3.save(img_save_path_3)

            # npz_save_path_0 = 'results/depths_{}/npz/{}_0'.format(exp_name, self.counter)
            # npz_save_path_1 = 'results/depths_{}/npz/{}_1'.format(exp_name, self.counter)
            # npz_save_path_2 = 'results/depths_{}/npz/{}_2'.format(exp_name, self.counter)
            # npz_save_path_3 = 'results/depths_{}/npz/{}_3'.format(exp_name, self.counter)

            # np.save(npz_save_path_0, raw_inv_depth_0)
            # np.save(npz_save_path_1, raw_inv_depth_1)
            # np.save(npz_save_path_2, raw_inv_depth_2)
            # np.save(npz_save_path_3, raw_inv_depth_3)

        self.counter += 1

        if not self.training:
            # If not training, no need for self-supervised loss
            return output
        else:
            # Otherwise, calculate self-supervised loss
            self_sup_output = self.self_supervised_loss(
                batch['rgb_original'], batch['rgb_context_original'],
                output['inv_depths'], output['poses'], output['intrinsics'],
                return_logs=return_logs, progress=progress)

            # print(output)
            # print(self_sup_output)
            
            # Return loss and metrics
            return {
                'loss': self_sup_output['loss'],
                **merge_outputs(output, self_sup_output),
            }
