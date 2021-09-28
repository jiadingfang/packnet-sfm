# Copyright 2020 Toyota Research Institute.  All rights reserved.

import os
import torch
from torch import autograd
# from torch.autograd.functional import jacobian
import horovod.torch as hvd
from packnet_sfm.trainers.base_trainer import BaseTrainer, sample_to_cuda
from packnet_sfm.utils.config import prep_logger_and_checkpoint
from packnet_sfm.utils.logging import print_config
from packnet_sfm.utils.logging import AvgMeter

from packnet_sfm.geometry.camera_utils import view_synthesis
from packnet_sfm.geometry.camera_eucm import EUCMCamera
from packnet_sfm.utils.depth import inv2depth
from packnet_sfm.geometry.pose import Pose


class HorovodTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        hvd.init()
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 1)))
        torch.cuda.set_device(hvd.local_rank())
        torch.backends.cudnn.benchmark = True

        self.avg_loss = AvgMeter(50)
        self.dtype = kwargs.get("dtype", None)  # just for test for now

    @property
    def proc_rank(self):
        return hvd.rank()

    @property
    def world_size(self):
        return hvd.size()

    def fit(self, module):

        # Prepare module for training
        module.trainer = self
        # Update and print module configuration
        prep_logger_and_checkpoint(module)
        print_config(module.config)

        # Send module to GPU
        module = module.to('cuda:{}'.format(int(module.config.gpu.idx)))
        # Configure optimizer and scheduler
        module.configure_optimizers()

        # print('module optimizer')
        # print(module.optimizer)

        # print('module parameters')
        # # counter = 0
        # for name, parameter in module.named_parameters():
        #     if  'depth_net.encoder' in name:
        #         print(name)
        #         print(parameter.shape)
        #     counter += 1

        # for name, param in module.named_parameters():
        #     if param.requires_grad and 'intrinsic_vector' in name:
        #         # param.requires_grad = False
        #         print(param.data)
        #         param.data = torch.tensor([-2.0, -2.0, -2.0, -2.0, 0.0, 1.0])
        #         param.data = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        #         print(param.data)

        # for name, param in module.named_parameters():
        #     if ('depth' in name or 'pose' in name) and ('intrinsic' not in name):
        #         print(name)
        #     print(param.requires_grad)


        # Create distributed optimizer
        compression = hvd.Compression.none
        optimizer = hvd.DistributedOptimizer(module.optimizer,
            named_parameters=module.named_parameters(), compression=compression)
        scheduler = module.scheduler

        # Get train and val dataloaders
        train_dataloader = module.train_dataloader()
        val_dataloaders = module.val_dataloader()

        # Epoch loop
        for epoch in range(module.current_epoch, self.max_epochs):
            
            # Freeze intrinsic vector in first 10 epochs
            if epoch < 50:
                for name, param in module.named_parameters():
                    if param.requires_grad and 'intrinsic_vector' in name:
                        param.requires_grad = False
            else:
                for name, param in module.named_parameters():
                    if not param.requires_grad and 'intrinsic_vector' in name:
                        param.requires_grad = True

            # Freeze depth and pose networks after 50 epochs
            # if epoch >= 49:
            #     for name, param in module.named_parameters():
            #         if param.requires_grad and ('depth' in name or 'pose' in name) and ('intrinsic' not in name):
            #             param.requires_grad = False

            # print('check requires_grad')
            # for name, param in module.named_parameters():
            #     if param.requires_grad:
            #         print(name)

            # Train
            self.train(train_dataloader, module, optimizer)
            # Validation
            validation_output = self.validate(val_dataloaders, module)
            # Check and save model
            self.check_and_save(module, validation_output)
            # Update current epoch
            module.current_epoch += 1
            # Take a scheduler step
            scheduler.step()

    def train(self, dataloader, module, optimizer):
        # Set module to train
        module.train()
        # Shuffle dataloader sampler
        if hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(module.current_epoch)
        # Prepare progress bar
        progress_bar = self.train_progress_bar(
            dataloader, module.config.datasets.train)
        # Start training loop
        outputs = []
        # For all batches
        for i, batch in progress_bar:
            # Reset optimizer
            optimizer.zero_grad()
            # Send samples to GPU and take a training step
            batch = sample_to_cuda(batch, int(module.config.gpu.idx))
            output = module.training_step(batch, i)
            # Backprop through loss and take an optimizer step
            output['loss'].backward()
            optimizer.step()

            # # print(output.keys())
            # image = output['image']
            # print(image.shape)
            # context = output['context']
            # print(len(context))
            # # print(context[0].shape)
            # inv_depth = output['inv_depths'][0].detach()
            # # print(len(inv_depths))
            # print(inv_depth.shape)
            # poses = [Pose(pose.mat.detach()) for pose in output['poses']]
            # print(len(poses))
            # I = output['I'].detach()
            # print(I.shape)

            # def warp_ref_image(inv_depth, ref_image, I, ref_I, pose):
            #     B, _, H, W = ref_image.shape
            #     device = ref_image.get_device()
            #     # Generate camera
            #     _, _, DH, DW = inv_depth.shape
            #     scale_factor = DW / float(W)
            #     cam = EUCMCamera(I=I).to(device)
            #     ref_cam = EUCMCamera(I=ref_I, Tcw=pose).to(device)
            #     # View synthesis
            #     depths = inv2depth(inv_depth)
            #     # ref_images = match_scales(ref_image, inv_depths, self.n)
            #     ref_warped = view_synthesis(
            #         ref_image, depths, ref_cam, cam,
            #         padding_mode='zeros')
            #     # Return warped reference image
            #     return ref_warped

            # def calc_ba_loss(image, context, inv_depth, poses, I):
            #     B, _, H, W = image.shape
            #     n_pts = 100
            #     sample_points = torch.randperm(H * W)[:n_pts]
            #     for j, (ref_image, pose) in enumerate(zip(context, poses)):
            #         def calc_error_vec(I):
            #             # Calculate warped images
            #             ref_warped = warp_ref_image(inv_depth, ref_image, I, I, pose)
            #             # photometric loss and on first batch item
            #             photometric_loss = torch.norm(ref_warped - image, dim=1)[0].view(-1)
            #             # photometric loss on selected points
            #             pts_photometric_loss = photometric_loss[sample_points]

            #             return pts_photometric_loss

            #             # calculate Jacobian
            #             J = jacobian(calc_error_vec, I)
            #             print(J)
            #     return
            # ba_loss = calc_ba_loss(image, context, inv_depth, poses, I)

            # with autograd.detect_anomaly():
            #     output = module.training_step(batch, i)
            #     # Backprop through loss and take an optimizer step
            #     print('output loss')
            #     print(output['loss'])
            #     output['loss'].backward()

            # for name, param in module.named_parameters():
            #     if name == 'model.depth_net.intrinsic_decoder.intrinsic_vector':
            #         print()
            #         print('intrinsic vector')
            #         print('data')
            #         print(param.data)
            #         print('grad')
            #         print(param.grad)
            #         print('requires grad')
            #         print(param.requires_grad)

            
            # Append output to list of outputs
            output['loss'] = output['loss'].detach()
            outputs.append(output)
            # Update progress bar if in rank 0
            if self.is_rank_0:
                progress_bar.set_description(
                    'Epoch {} | Avg.Loss {:.4f}'.format(
                        module.current_epoch, self.avg_loss(output['loss'].item())))
        # Return outputs for epoch end
        return module.training_epoch_end(outputs)

    def validate(self, dataloaders, module):
        # Set module to eval
        module.eval()
        # Start validation loop
        all_outputs = []
        # For all validation datasets
        for n, dataloader in enumerate(dataloaders):
            # Prepare progress bar for that dataset
            progress_bar = self.val_progress_bar(
                dataloader, module.config.datasets.validation, n)
            outputs = []
            # For all batches
            for i, batch in progress_bar:
                # print('batch')
                # print(batch.keys())
                # print(batch['rgb'].shape)
                # print(batch['rgb_context'][0].shape)
                # print(batch['rgb_context'][1].shape)
                # Send batch to GPU and take a validation step
                batch = sample_to_cuda(batch, int(module.config.gpu.idx))
                output = module.validation_step(batch, i, n)
                # Append output to list of outputs
                outputs.append(output)
            # Append dataset outputs to list of all outputs
            all_outputs.append(outputs)
        # Return all outputs for epoch end
        return module.validation_epoch_end(all_outputs)

    def test(self, module):
        # Send module to GPU
        module = module.to('cuda:{}'.format(int(module.config.gpu.idx)), dtype=self.dtype)
        # Get test dataloaders
        test_dataloaders = module.test_dataloader()
        # Run evaluation
        self.evaluate(test_dataloaders, module)

    @torch.no_grad()
    def evaluate(self, dataloaders, module):
        # Set module to eval
        module.eval()
        # Start evaluation loop
        all_outputs = []
        # For all test datasets
        for n, dataloader in enumerate(dataloaders):
            # Prepare progress bar for that dataset
            progress_bar = self.val_progress_bar(
                dataloader, module.config.datasets.test, n)
            outputs = []
            # For all batches
            for i, batch in progress_bar:
                # Send batch to GPU and take a test step
                batch = sample_to_cuda(batch, int(module.config.gpu.idx), self.dtype)
                output = module.test_step(batch, i, n)
                # Append output to list of outputs
                outputs.append(output)
            # Append dataset outputs to list of all outputs
            all_outputs.append(outputs)
        # Return all outputs for epoch end
        return module.test_epoch_end(all_outputs)
