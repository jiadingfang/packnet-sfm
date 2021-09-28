import torch

from packnet_sfm.utils.image import image_grid
from packnet_sfm.geometry.camera_ucm import UCMCamera

def reduce_photometric_loss(photometric_losses, mask=None):
    """
    Combine the photometric loss from all context images

    Parameters
    ----------
    photometric_losses : list of torch.Tensor [B,3,H,W]
        Pixel-wise photometric losses from the entire context

    mask : torch.bool [B,1,H,W]
        mask for valid pixels in UCM model

    Returns
    -------
    photometric_loss : torch.Tensor [1]
        Reduced photometric loss
    """
    # Reduce function
    def reduce_function(losses, mask=None):
        # print(self.photometric_reduce_op)
        # print('losses')
        # print(len(losses))
        # print(losses[0].shape)
        # print(torch.cat(losses, 1).shape)
        # print(torch.cat(losses, 1).min(1, True)[0].shape)

        # if self.photometric_reduce_op == 'mean':
        #     if mask is None:
        #         return sum([l.mean() for l in losses]) / len(losses)
        #     else:
        #         return sum([torch.masked_select(l, mask).mean() for l in losses]) / len(losses)
        # elif self.photometric_reduce_op == 'min':
            # if mask is None:
            #     return torch.cat(losses, 1).min(1, True)[0].mean()
            # else:

        # print('torch.cat(losses, 1)')
        # print(torch.cat(losses, 1))
        # print('torch.cat(losses, 1).min(1, True)[0]')
        # print(torch.cat(losses, 1).min(1, True)[0])
        # print('torch.masked_select(torch.cat(losses, 1).min(1, True)[0], mask)')
        # print(torch.masked_select(torch.cat(losses, 1).min(1, True)[0], mask))

        return torch.masked_select(torch.cat(losses, 1).min(1, True)[0], mask).mean()
        # else:
        #     raise NotImplementedError(
        #         'Unknown photometric_reduce_op: {}'.format(self.photometric_reduce_op))

    n = 1
    # Reduce photometric loss
    photometric_loss = sum([reduce_function(photometric_losses[i], mask=mask)
                            for i in range(n)]) / n
    # # Store and return reduced photometric loss
    # self.add_metric('photometric_loss', photometric_loss)
    return photometric_loss

def calc_pixel_mask(image, I):
    """
    Calculates unprojection valid mask for valid pixels in UCM model according equation (15) in the double sphere paper

    Parameters
    ----------
    image : torch.Tensor [B,3,H,W]
        Original image
    I : torch.Tensor [B, 5]
        Camera intrinsics

    Returns
    -------
    mask : torch.bool [B,1,H,W]
        mask of the batch images
    """
    B, C, H, W = image.shape

    grid = image_grid(B, H, W, image.dtype, 'cpu', normalized=False)  # [B,1,H,W]

    fx = I[:, 0].unsqueeze(1).unsqueeze(2)
    fy = I[:, 1].unsqueeze(1).unsqueeze(2)
    cx = I[:, 2].unsqueeze(1).unsqueeze(2)
    cy = I[:, 3].unsqueeze(1).unsqueeze(2)
    alpha = I[:, 4].unsqueeze(1).unsqueeze(2)

    u = grid[:,0,:,:]
    v = grid[:,1,:,:]

    mx = (u - cx) / fx * (1 - alpha)
    my = (v - cy) / fy * (1 - alpha)
    r_square = mx ** 2 + my ** 2

    print('u')
    print(u)
    print('v')
    print(v)
    print('r_square')
    print(r_square)

    mask = (r_square <= (1 - alpha) ** 2 / (2 * alpha - 1)) | (alpha <= 1/2)
    mask = mask.detach().unsqueeze(1)

    # print('mask')
    # print(mask.shape)
    # print(mask.dtype)
    # print(torch.sum(torch.logical_not(mask)))

    return mask

I_0 = torch.tensor([307.4048, 200.0247, 242.0782, 163.3021, 0.8899])
I_0 = I_0.unsqueeze(0)
# I_1 = torch.tensor([253.9979, 267.6426, 204.2654, 194.4133, 0.517])
# I_1 = I_1.unsqueeze(0)

ucm_cam_0 = UCMCamera(I=I_0)
# ucm_cam_1 = UCMCamera(I=I_1)

# H = 1
# W = 1
# H = 2
# W = 3
# H = 128
# W = 192
H = 256
W = 384

x = torch.rand(1,3,H,W)
d = torch.rand(1,1,H,W)

mask = calc_pixel_mask(d, I_0)
# mask = calc_pixel_mask(d, I_1)
print('mask')
print(mask)
print()

photometric_losses = [[torch.rand(1,1,H,W)]]
photometric_loss = reduce_photometric_loss(photometric_losses, mask)
print('photometric_loss')
print(photometric_loss)
print()

print('projection')
# print(camera.project(x))
print(ucm_cam_0.project(x))
# print(ucm_cam_1.project(x))
print()
print('unprojection')
# print(camera.reconstruct(d))
print(ucm_cam_0.reconstruct(d))
# print(ucm_cam_1.reconstruct(d))
