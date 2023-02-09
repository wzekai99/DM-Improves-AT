import numpy as np

import torch
import torch.nn.functional as F

def cutmix(images, labels, alpha=1.0, beta=1.0, num_classes=10):
    """
    Apply CutMix to a batch of images.
    Arguments:
        image (torch.FloatTensor): images.
        labels (torch.LongTensor): target labels.
        alpha (float): parameter for cut ratio.
        beta (float): parameter for cut ratio.
        num_classes (int): number of target classes.
    Returns:
        augmented batch of images and labels.
    """
    batch_size, _, height, width = images.shape
    labels = F.one_hot(labels, num_classes)
    
    lam = np.random.beta(alpha, beta)
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.array(width * cut_rat, dtype=np.int32)
    cut_h = np.array(height * cut_rat, dtype=np.int32)
    box_coords = _random_box(height, width, cut_h, cut_w)
    
    # Adjust lambda.
    lam = 1. - (box_coords[2] * box_coords[3] / (height * width))
    idx = np.random.permutation(batch_size)

    def _cutmix(x, y):
        images_a = x
        images_b = x[idx, :, :, :]
        y = lam * y + (1. - lam) * y[idx, :]
        x = _compose_two_images(images_a, images_b, box_coords)
        return x, y

    return _cutmix(images, labels)


def _random_box(height, width, cut_h, cut_w):
    """
    Return a random box within the image size.
    """
    minval_h = 0
    minval_w = 0
    maxval_h = height
    maxval_w = width
    
    i = np.random.randint(minval_h, maxval_h, dtype=np.int32)
    j = np.random.randint(minval_w, maxval_w, dtype=np.int32)
    bby1 = np.clip(i - cut_h // 2, 0, height)
    bbx1 = np.clip(j - cut_w // 2, 0, width)
    h = np.clip(i + cut_h // 2, 0, height) - bby1
    w = np.clip(j + cut_w // 2, 0, width) - bbx1
    return np.array([bby1, bbx1, h, w])


def _compose_two_images(images, image_permutation, bbox):
    """
    Mix two images.
    """
    def _single_compose_two_images(image1, image2):
        _, height, width = image1.shape
        mask = _window_mask(bbox, (height, width))
        return image1 * (1. - mask) + image2 * mask
    
    new_images = [_single_compose_two_images(image1, image2) for image1, image2 in zip(images, image_permutation)]
    return torch.stack(new_images, dim=0)


def _window_mask(destination_box, size):
    """
    Compute window mask.
    """
    height_offset, width_offset, h, w = destination_box
    h_range = np.reshape(np.arange(size[0]), [1, size[0], 1])
    w_range = np.reshape(np.arange(size[1]), [1, 1, size[1]])
    return np.logical_and(
        np.logical_and(height_offset <= h_range, h_range < height_offset + h), 
        np.logical_and(width_offset <= w_range, w_range < width_offset + w)
    ).astype(np.float32)