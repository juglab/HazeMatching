"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics.image.fid import FrechetInceptionDistance as FID_score
import pyiqa


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normalize_batch(imgs):
    imgs_flat = imgs.view(imgs.size(0), -1)
    imgs_min = imgs_flat.min(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
    imgs_max = imgs_flat.max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
    return (imgs - imgs_min) / (imgs_max - imgs_min + 1e-8)

def normalize_to_neg1_1(imgs):
    imgs_flat = imgs.view(imgs.size(0), -1)
    imgs_min = imgs_flat.min(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
    imgs_max = imgs_flat.max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
    return 2 * (imgs - imgs_min) / (imgs_max - imgs_min + 1e-8) - 1

def lpips(gt, posterior_samples, net_type='alex', batch_size=50):
    '''
    Calculate LPIPS score between the ground truth and the prediction in memory-safe chunks.
    Returns mean LPIPS over the batch.
    '''
    # ensure cpu tensors and no grad
    posterior_samples = posterior_samples.detach().cpu()
    gt = torch.tensor(gt).clone().detach().cpu()
    gt = normalize_to_neg1_1(gt)

    posterior_samples = normalize_to_neg1_1(posterior_samples)
    posterior_samples = posterior_samples.repeat(1, 3, 1, 1)
    gt = gt.repeat(1, 3, 1, 1)

    lpips_obj = LPIPS(net_type=net_type, normalize=False).to(device)
    scores = []
    with torch.no_grad():
        N = gt.shape[0]
        for i in range(0, N, batch_size):
            g = gt[i:i+batch_size].to(device)
            p = posterior_samples[i:i+batch_size].to(device)
            s = lpips_obj(g, p)
            scores.append(s.detach().cpu())
    # cleanup
    del lpips_obj
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    scores = torch.tensor(scores)
    return float(scores.mean().item())


def fid_score(gt, posterior_samples, batch_size=50):
    '''
    Calculate FID score between the ground truth and the prediction in memory-safe chunks.
    Returns scalar FID.
    '''
    posterior_samples = posterior_samples.detach().cpu()
    gt = torch.tensor(gt).clone().detach().cpu()
    gt = normalize_batch(gt)

    posterior_samples = normalize_batch(posterior_samples)
    posterior_samples = posterior_samples.repeat(1, 3, 1, 1)
    gt = gt.repeat(1, 3, 1, 1)
    
    fid = FID_score(feature=768, normalize=True).to(device)
    with torch.no_grad():

        # update real (gt) in chunks
        N = gt.shape[0]
        for i in range(0, N, batch_size):
            fid.update(gt[i:i+batch_size].to(device), real=True)

        # update fake (posterior_samples) in chunks
        M = posterior_samples.shape[0]
        for i in range(0, M, batch_size):
            fid.update(posterior_samples[i:i+batch_size].to(device), real=False)

        result = fid.compute()

    # cleanup
    del fid
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return float(result.item())

#make a straigghtness function that computes the 

def straightness_score(A_values):
    """
    Computes Bs based on the differences between consecutive As.

    Parameters:
        A_values (torch.Tensor): Tensor of A values with shape (num_steps, ...).

    Returns:
        torch.Tensor: Tensor of B values with shape (num_steps-1, ...).
    """
    B_values = []
    for i in range(A_values.shape[0] - 1):
        B = A_values[i] - A_values[i + 1]  # Difference between consecutive As
        B_values.append(B)
    B_values = torch.stack(B_values)
    sum_B = torch.sum(B_values, dim=0)
    straightness_score = torch.mean(sum_B)
    return straightness_score

def NIQE(input):
    """
    Computes the NIQE score of the input image.

    Parameters:
        input (torch.Tensor): Tensor of the input image with shape (N,C, ...).

    Returns:
        float: NIQE score of the input image.
    """

    for i in range(input.shape[0]):
        img = input[i]
        img = (img - img.min()) / (img.max() - img.min())
        input[i] = img
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    niqe_metric = pyiqa.create_metric('niqe', device=device)
    niqe_score = niqe_metric(input)
    return niqe_score

def FSIM(input, target):
    """
    Computes the FSIM score of the input image.
    Input and target should be in the range [0,1].

    Parameters:
        input (torch.Tensor): Tensor of the input image with shape (N,C, ...).
        target (torch.Tensor): Tensor of the target image with shape (N,C, ...).
        https://github.com/chaofengc/IQA-PyTorch/blob/61cf2fd2cd78cb2150204aed996c7b2092858145/pyiqa/archs/fsim_arch.py#L438
    Returns:
        float: FSIM score of the input image. [0,1]
    """

    # for i in range(input.shape[0]):
    #     img = input[i]
    #     img = (img - img.min()) / (img.max() - img.min())
    #     input[i] = img
    input = normalize_batch(input)
    # for i in range(target.shape[0]):
    #     img = target[i]
    #     img = (img - img.min()) / (img.max() - img.min())
    #     target[i] = img
    #move the tensors to the device
    input = input.to(device)
    target = target.to(device)
    fsim_metric = pyiqa.create_metric('fsim', device=device, chromatic=False)
    fsim_score = fsim_metric(input, target)
    return fsim_score

def entropy(input):
    """
    Computes the entropy of the input image.

    Parameters:
        input (torch.Tensor): Tensor of the input image with shape (N,C, ...).

    Returns:
        float: Entropy of the input image.
    """
    # for i in range(input.shape[0]):
    #     img = input[i]
    #     img = (img - img.min()) / (img.max() - img.min())
    #     input[i] = img
    input = normalize_batch(input)
    #move the tensors to the device
    input = input.to(device)
    entropy_metric = pyiqa.create_metric('entropy', device=device)
    entropy_score = entropy_metric(input)
    return entropy_score

def GMSD(input, target):
    """
    Computes the GMSD score of the input image.
    Input and target should be in the
    range [0,1].
    """
    # for i in range(input.shape[0]):
    #     img = input[i]
    #     img = (img - img.min()) / (img.max() - img.min())
    #     input[i] = img
    input = normalize_batch(input)
    input = input.repeat(1, 3, 1, 1)
    # for i in range(target.shape[0]):
    #     img = target[i]
    #     img = (img - img.min()) / (img.max() - img.min())
    #     target[i] = img
    target = target.repeat(1, 3, 1, 1)
    #move the tensors to the device
    input = input.to(device)
    target = target.to(device)
    gmsd_metric = pyiqa.create_metric('gmsd', device=device)
    gmsd_score = gmsd_metric(input, target)
    return gmsd_score


def extract_patches_inner(image, patch_size=128, crop_size=64):
    """
    Extract patches and record crop offsets such that border patches are cropped
    on the appropriate side.
    Returns:
        patches: numpy array of shape (num_patches, channels, patch_size, patch_size)
        coords: list of tuples (crop_top, crop_left, pos_i, pos_j) where pos_* is the placement in full image.
    """
    assert image.ndim == 3
    C, H, W = image.shape
    effective = (patch_size - crop_size) // 2

    # Determine grid positions for cropped region (full image)
    pos_i_list = list(range(0, H - crop_size + 1, crop_size))
    pos_j_list = list(range(0, W - crop_size + 1, crop_size))
    
    patches = []
    coords = []
    for pos_i in pos_i_list:
        for pos_j in pos_j_list:
            # Compute extraction window starting indices with border adjustment
            i0 = pos_i - effective
            j0 = pos_j - effective
            if i0 < 0:
                i0 = 0
            if j0 < 0:
                j0 = 0
            if i0 + patch_size > H:
                i0 = H - patch_size
            if j0 + patch_size > W:
                j0 = W - patch_size
            # Compute in-patch crop offsets
            crop_top = pos_i - i0
            crop_left = pos_j - j0
            patch = image[:, i0:i0+patch_size, j0:j0+patch_size]
            patches.append(patch)
            coords.append((crop_top, crop_left, pos_i, pos_j))
    patches = np.stack(patches)
    return patches, coords


def reconstruct_image_inner(patches, coords, image_shape, patch_size=128, crop_size=64):
    """
    Reconstruct full image from patches.
    patches: numpy array of shape (num_samples, num_steps, num_patches, channels, patch_size, patch_size)
    coords: list of tuples (crop_top, crop_left, pos_i, pos_j)
    image_shape: tuple (num_samples, channels, H, W) for the full image.
    For each patch, extract its inner crop based on stored offsets and place it in full image.
    """
    # Unpack image shape: here num_samples is from patches if multi-step exists.
    num_samples, num_steps, num_patches, channels, _, _ = patches.shape
    _, H, W = image_shape
    full_image = np.zeros((num_samples, num_steps, channels, H, W), dtype=patches.dtype)
    
    for idx, (crop_top, crop_left, pos_i, pos_j) in enumerate(coords):
        # Extract crop region from patch
        # For all samples and steps
        patch_crop = patches[:, :, idx, :, crop_top:crop_top+crop_size, crop_left:crop_left+crop_size]
        # Place cropped patch in full image at its designated location
        full_image[:, :, :, pos_i:pos_i+crop_size, pos_j:pos_j+crop_size] = patch_crop
    return full_image

def extract_patches_inner_metrics(image, patch_size=128):
    """
    Extract non-overlapping patches from the image.
    Returns:
        patches: numpy array of shape (num_patches, channels, patch_size, patch_size)
        coords: list of tuples (pos_i, pos_j) where pos_* is the placement in full image.
    """
    assert image.ndim == 3
    C, H, W = image.shape

    # Determine grid positions for non-overlapping patches
    pos_i_list = list(range(0, H, patch_size))
    pos_j_list = list(range(0, W, patch_size))
    
    patches = []
    coords = []
    for pos_i in pos_i_list:
        for pos_j in pos_j_list:
            # Ensure the patch is within the image boundaries
            if pos_i + patch_size <= H and pos_j + patch_size <= W:
                patch = image[:, pos_i:pos_i+patch_size, pos_j:pos_j+patch_size]
                patches.append(patch)
                coords.append((pos_i, pos_j))
    patches = np.stack(patches)
    return patches, coords


