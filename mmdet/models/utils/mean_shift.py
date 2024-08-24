# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def ball_kernel(Z, X, kappa, metric='cosine'):
    """Computes pairwise ball kernel (without normalizing constant) (note this
    is kernel as defined in non-parametric statistics, not a kernel as in RKHS)

    @param Z: a [n x d] torch.FloatTensor of NORMALIZED datapoints - the seeds
    @param X: a [m x d] torch.FloatTensor of NORMALIZED datapoints - the points

    @return: a [n x m] torch.FloatTensor of pairwise ball kernel computations,
             without normalizing constant
    """
    if metric == 'euclidean':
        distance = Z.unsqueeze(1) - X.unsqueeze(0)
        distance = torch.norm(distance, dim=2)
        kernel = torch.exp(-kappa * torch.pow(distance, 2))
    elif metric == 'cosine':
        kernel = torch.exp(kappa * torch.mm(Z, X.t()))
    return kernel


def get_label_mode(array):
    """Computes the mode of elements in an array. Ties don't matter. Ties are
    broken by the smallest value (np.argmax defaults)

    @param array: a numpy array
    """
    labels, counts = np.unique(array, return_counts=True)
    mode = labels[np.argmax(counts)].item()
    return mode


def connected_components(Z, epsilon, metric='cosine', device='cuda'):
    """For the connected components, we simply perform a nearest neighbor
    search in order: for each point, find the points that are up to epsilon
    away (in cosine distance) these points are labeled in the same cluster.

    @param Z: a [n x d] torch.FloatTensor of NORMALIZED datapoints

    @return: a [n] torch.LongTensor of cluster labels
    """
    n, d = Z.shape

    K = 0
    cluster_labels = torch.ones(n, dtype=torch.long, device=device) * -1
    for i in range(n):
        if cluster_labels[i] == -1:

            if metric == 'euclidean':
                distances = Z.unsqueeze(1) - Z[i:i + 1].unsqueeze(
                    0)  # a are points, b are seeds
                distances = torch.norm(distances, dim=2)
            elif metric == 'cosine':
                distances = 0.5 * (1 - torch.mm(Z, Z[i:i + 1].t()))
            component_seeds = distances[:, 0] <= epsilon

            # If at least one component already has a label, then use the mode of the label
            if torch.unique(cluster_labels[component_seeds]).shape[0] > 1:
                temp = cluster_labels[component_seeds].numpy()
                temp = temp[temp != -1]
                label = torch.tensor(get_label_mode(temp))
            else:
                label = torch.tensor(K)
                K += 1  # Increment number of clusters

            cluster_labels[component_seeds] = label

    return cluster_labels


def seed_hill_climbing_ball(X, Z, kappa, max_iters=10, metric='cosine'):
    """Runs mean shift hill climbing algorithm on the seeds. The seeds climb
    the distribution given by the KDE of X.

    @param X: a [n x d] torch.FloatTensor of d-dim unit vectors
    @param Z: a [m x d] torch.FloatTensor of seeds to run mean shift from
    @param dist_threshold: parameter for the ball kernel
    """
    n, d = X.shape
    m = Z.shape[0]

    for _iter in range(max_iters):

        # Create a new object for Z
        new_Z = Z.clone()

        W = ball_kernel(Z, X, kappa, metric=metric)

        # use this allocated weight to compute the new center
        new_Z = torch.mm(W, X)  # Shape: [n x d]

        # Normalize the update
        if metric == 'euclidean':
            summed_weights = W.sum(dim=1)
            summed_weights = summed_weights.unsqueeze(1)
            summed_weights = torch.clamp(summed_weights, min=1.0)
            Z = new_Z / summed_weights
        elif metric == 'cosine':
            Z = F.normalize(new_Z, p=2, dim=1)

    return Z


def mean_shift_with_seeds(X,
                          Z,
                          kappa,
                          max_iters=10,
                          metric='cosine',
                          embedding_alpha=0.02):
    """Runs mean-shift.

    @param X: a [n x d] torch.FloatTensor of d-dim unit vectors
    @param Z: a [m x d] torch.FloatTensor of seeds to run mean shift from
    @param dist_threshold: parameter for the von Mises-Fisher distribution
    """

    Z = seed_hill_climbing_ball(
        X, Z, kappa, max_iters=max_iters, metric=metric)

    # Connected components
    cluster_labels = connected_components(
        Z, 2 * embedding_alpha, metric=metric)  # Set epsilon = 0.1 = 2*alpha

    return cluster_labels, Z


def select_smart_seeds(X,
                       num_seeds,
                       return_selected_indices=False,
                       init_seeds=None,
                       num_init_seeds=None,
                       metric='cosine'):
    """Selects seeds that are as far away as possible.

    @param X: a [n x d] torch.FloatTensor of d-dim unit vectors
    @param num_seeds: number of seeds to pick
    @param init_seeds: a [num_seeds x d] vector of initial seeds
    @param num_init_seeds: the number of seeds already chosen.
                           the first num_init_seeds rows of init_seeds have been chosen already

    @return: a [num_seeds x d] matrix of seeds
             a [n x num_seeds] matrix of distances
    """
    n, d = X.shape
    selected_indices = -1 * torch.ones(num_seeds, dtype=torch.long)

    # Initialize seeds matrix
    if init_seeds is None:
        seeds = torch.empty((num_seeds, d), device=X.device)
        num_chosen_seeds = 0
    else:
        seeds = init_seeds
        num_chosen_seeds = num_init_seeds

    # Keep track of distances
    distances = torch.empty((n, num_seeds), device=X.device)

    if num_chosen_seeds == 0:  # Select first seed if need to
        selected_seed_index = np.random.randint(0, n)
        selected_indices[0] = selected_seed_index
        selected_seed = X[selected_seed_index, :]
        seeds[0, :] = selected_seed
        if metric == 'euclidean':
            distances[:, 0] = torch.norm(X - selected_seed.unsqueeze(0), dim=1)
        elif metric == 'cosine':
            distances[:, 0] = 0.5 * (
                1 - torch.mm(X, selected_seed.unsqueeze(1))[:, 0])
        num_chosen_seeds += 1
    else:  # Calculate distance to each already chosen seed
        for i in range(num_chosen_seeds):
            if metric == 'euclidean':
                distances[:, i] = torch.norm(X - seeds[i:i + 1, :], dim=1)
            elif metric == 'cosine':
                distances[:,
                          i] = 0.5 * (1 -
                                      torch.mm(X, seeds[i:i + 1, :].t())[:, 0])

    # Select rest of seeds
    for i in range(num_chosen_seeds, num_seeds):
        # Find the point that has the furthest distance from the nearest seed
        distance_to_nearest_seed = torch.min(
            distances[:, :i], dim=1)[0]  # Shape: [n]
        selected_seed_index = torch.argmax(distance_to_nearest_seed)
        selected_indices[i] = selected_seed_index
        selected_seed = torch.index_select(X, 0, selected_seed_index)[0, :]
        seeds[i, :] = selected_seed

        # Calculate distance to this selected seed
        if metric == 'euclidean':
            distances[:, i] = torch.norm(X - selected_seed.unsqueeze(0), dim=1)
        elif metric == 'cosine':
            distances[:, i] = 0.5 * (
                1 - torch.mm(X, selected_seed.unsqueeze(1))[:, 0])

    return_tuple = (seeds, )
    if return_selected_indices:
        return_tuple += (selected_indices, )
    return return_tuple


def mean_shift_smart_init(X,
                          kappa,
                          num_seeds=100,
                          max_iters=10,
                          metric='cosine'):
    """Runs mean shift with carefully selected seeds.

    @param X: a [n x d] torch.FloatTensor of d-dim unit vectors
    @param dist_threshold: parameter for the von Mises-Fisher distribution
    @param num_seeds: number of seeds used for mean shift clustering

    @return: a [n] array of cluster labels
    """

    n, d = X.shape
    seeds, selected_indices = select_smart_seeds(
        X, num_seeds, return_selected_indices=True, metric=metric)
    seed_cluster_labels, updated_seeds = mean_shift_with_seeds(
        X, seeds, kappa, max_iters=max_iters, metric=metric)

    # Get distances to updated seeds
    if metric == 'euclidean':
        distances = X.unsqueeze(1) - updated_seeds.unsqueeze(
            0)  # a are points, b are seeds
        distances = torch.norm(distances, dim=2)
    elif metric == 'cosine':
        distances = 0.5 * (1 - torch.mm(X, updated_seeds.t())
                           )  # Shape: [n x num_seeds]

    # Get clusters by assigning point to closest seed
    closest_seed_indices = torch.argmin(distances, dim=1)  # Shape: [n]
    cluster_labels = seed_cluster_labels[closest_seed_indices]

    # assign zero to the largest cluster
    num = len(torch.unique(seed_cluster_labels))
    count = torch.zeros(num, dtype=torch.long)
    for i in range(num):
        count[i] = (cluster_labels == i).sum()
    label_max = torch.argmax(count)
    if label_max != 0:
        index1 = cluster_labels == 0
        index2 = cluster_labels == label_max
        cluster_labels[index1] = label_max
        cluster_labels[index2] = 0

    return cluster_labels, selected_indices


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '{:.3f} ({:.3f})'.format(self.val, self.avg)


def clustering_features(features, num_seeds=100):
    metric = 'cosine'
    height = features.shape[2]
    width = features.shape[3]
    out_label = torch.zeros((features.shape[0], height, width))

    # mean shift clustering
    kappa = 20
    selected_pixels = []
    for j in range(features.shape[0]):
        X = features[j].view(features.shape[1], -1)
        X = torch.transpose(X, 0, 1)
        cluster_labels, selected_indices = mean_shift_smart_init(
            X, kappa=kappa, num_seeds=num_seeds, max_iters=10, metric=metric)
        out_label[j] = cluster_labels.view(height, width)
        selected_pixels.append(selected_indices)
    return out_label, selected_pixels


import random


def convert_segmentation_to_rgb(image):
    if len(image.shape) == 3:
        image = image.squeeze(0)

    # Assume `image` is of shape (H, W)
    H, W = image.shape

    # Create a blank RGB tensor with shape (H, W, 3)
    rgb_image = torch.zeros((H, W, 3), dtype=torch.uint8)

    # Get all unique segmentation indices from the image
    unique_indices = torch.unique(image)

    # Create a color map for the unique indices
    color_map = {}
    for idx in unique_indices:
        # Assign a random color to each unique index
        if idx == 0:
            color_map[idx.item()] = torch.tensor([0, 0, 0], dtype=torch.uint8)
        else:
            color_map[idx.item()] = torch.tensor([
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            ],
                                                 dtype=torch.uint8)

    # Convert each index in the image to its corresponding RGB color
    for idx in unique_indices:
        rgb_image[image == idx] = color_map[idx.item()]

    return rgb_image


def mask_to_tight_box_numpy(mask):
    """Return bbox given mask.

    @param mask: a [H x W] numpy array
    """
    a = np.transpose(np.nonzero(mask))
    bbox = np.min(a[:, 1]), np.min(a[:, 0]), np.max(a[:, 1]), np.max(a[:, 0])
    return bbox  # x_min, y_min, x_max, y_max


def mask_to_tight_box_pytorch(mask):
    """Return bbox given mask.

    @param mask: a [H x W] torch tensor
    """
    a = torch.nonzero(mask)
    bbox = torch.min(a[:,
                       1]), torch.min(a[:,
                                        0]), torch.max(a[:,
                                                         1]), torch.max(a[:,
                                                                          0])
    return bbox  # x_min, y_min, x_max, y_max


def mask_to_tight_box(mask):
    if type(mask) == torch.Tensor:
        return mask_to_tight_box_pytorch(mask)
    elif type(mask) == np.ndarray:
        return mask_to_tight_box_numpy(mask)
    else:
        raise Exception(
            'Data type {type(mask)} not understood for mask_to_tight_box...')


def crop_rois(rgb, initial_masks, device='cuda'):

    N, H, W = initial_masks.shape
    crop_size = 224  # cfg.TRAIN.SYN_CROP_SIZE
    padding_percentage = 0.25

    mask_ids = torch.unique(initial_masks[0])
    if mask_ids[0] == 0:
        mask_ids = mask_ids[1:]
    num = mask_ids.shape[0]
    rgb_crops = torch.zeros((num, 3, crop_size, crop_size), device=device)
    rois = torch.zeros((num, 4), device=device)
    mask_crops = torch.zeros((num, crop_size, crop_size), device=device)

    for index, mask_id in enumerate(mask_ids):
        mask = (initial_masks[0] == mask_id).float()  # Shape: [H x W]
        x_min, y_min, x_max, y_max = mask_to_tight_box(mask)
        x_padding = int(
            torch.round((x_max - x_min).float() * padding_percentage).item())
        y_padding = int(
            torch.round((y_max - y_min).float() * padding_percentage).item())

        # pad and be careful of boundaries
        x_min = max(x_min - x_padding, 0)
        x_max = min(x_max + x_padding, W - 1)
        y_min = max(y_min - y_padding, 0)
        y_max = min(y_max + y_padding, H - 1)
        rois[index, 0] = x_min
        rois[index, 1] = y_min
        rois[index, 2] = x_max
        rois[index, 3] = y_max

        # crop
        rgb_crop = rgb[0, :, y_min:y_max + 1,
                       x_min:x_max + 1]  # [3 x crop_H x crop_W]
        mask_crop = mask[y_min:y_max + 1, x_min:x_max + 1]  # [crop_H x crop_W]

        # resize
        new_size = (crop_size, crop_size)
        rgb_crop = F.upsample_bilinear(
            rgb_crop.unsqueeze(0), new_size)[0]  # Shape: [3 x new_H x new_W]
        rgb_crops[index] = rgb_crop
        mask_crop = F.upsample_nearest(
            mask_crop.unsqueeze(0).unsqueeze(0),
            new_size)[0, 0]  # Shape: [new_H, new_W]
        mask_crops[index] = mask_crop

    return rgb_crops, mask_crops, rois


# labels_crop is the clustering labels from the local patch
def match_label_crop(initial_masks, labels_crop, out_label_crop, rois):
    num = labels_crop.shape[0]
    for i in range(num):
        mask_ids = torch.unique(labels_crop[i])
        for index, mask_id in enumerate(mask_ids):
            mask = (labels_crop[i] == mask_id).float()
            overlap = mask * out_label_crop[i]
            percentage = torch.sum(overlap) / torch.sum(mask)
            if percentage < 0.5:
                labels_crop[i][labels_crop[i] == mask_id] = -1

    # sort the local labels
    sorted_ids = []
    for i in range(num):
        x_min = rois[i, 0]
        y_min = rois[i, 1]
        x_max = rois[i, 2]
        y_max = rois[i, 3]
        orig_H = y_max - y_min + 1
        orig_W = x_max - x_min + 1
        roi_size = orig_H * orig_W
        sorted_ids.append((i, roi_size))

    sorted_ids = sorted(sorted_ids, key=lambda x: x[1], reverse=True)
    sorted_ids = [x[0] for x in sorted_ids]

    # combine the local labels
    refined_masks = torch.zeros_like(initial_masks).float()
    count = 0
    for index in sorted_ids:

        mask_ids = torch.unique(labels_crop[index])
        if mask_ids[0] == -1:
            mask_ids = mask_ids[1:]

        # mapping
        label_crop = torch.zeros_like(labels_crop[index])
        for mask_id in mask_ids:
            count += 1
            label_crop[labels_crop[index] == mask_id] = count

        # resize back to original size
        x_min = int(rois[index, 0].item())
        y_min = int(rois[index, 1].item())
        x_max = int(rois[index, 2].item())
        y_max = int(rois[index, 3].item())
        orig_H = int(y_max - y_min + 1)
        orig_W = int(x_max - x_min + 1)
        mask = label_crop.unsqueeze(0).unsqueeze(0).float()
        resized_mask = F.upsample_nearest(mask, (orig_H, orig_W))[0, 0]

        # Set refined mask
        h_idx, w_idx = torch.nonzero(resized_mask).t()
        refined_masks[0, y_min:y_max + 1, x_min:x_max + 1][
            h_idx, w_idx] = resized_mask[
                h_idx,
                w_idx]  #.cpu()  # in mean shift mask Transformer, disable cpu()

    return refined_masks, labels_crop


# filter labels on zero depths
def filter_labels_depth(labels, depth, threshold):
    labels_new = labels.clone()
    for i in range(labels.shape[0]):
        label = labels[i]
        mask_ids = torch.unique(label)
        if mask_ids[0] == 0:
            mask_ids = mask_ids[1:]

        for index, mask_id in enumerate(mask_ids):
            mask = (label == mask_id).float()
            roi_depth = depth[i, 2][label == mask_id]
            depth_percentage = torch.sum(
                roi_depth > 0).float() / torch.sum(mask)
            if depth_percentage < threshold:
                labels_new[i][label == mask_id] = 0

    return labels_new


# filter labels inside boxes
def filter_labels(labels, bboxes):
    labels_new = labels.clone()
    height = labels.shape[1]
    width = labels.shape[2]
    for i in range(labels.shape[0]):
        label = labels[i]
        bbox = bboxes[i].numpy()

        bbox_mask = torch.zeros_like(label)
        for j in range(bbox.shape[0]):
            x1 = max(int(bbox[j, 0]), 0)
            y1 = max(int(bbox[j, 1]), 0)
            x2 = min(int(bbox[j, 2]), width - 1)
            y2 = min(int(bbox[j, 3]), height - 1)
            bbox_mask[y1:y2, x1:x2] = 1

        mask_ids = torch.unique(label)
        if mask_ids[0] == 0:
            mask_ids = mask_ids[1:]

        for index, mask_id in enumerate(mask_ids):
            mask = (label == mask_id).float()
            percentage = torch.sum(mask * bbox_mask) / torch.sum(mask)
            if percentage > 0.8:
                labels_new[i][label == mask_id] = 0

    return labels_new


def normalize_descriptor(res, stats=None):
    """Normalizes the descriptor into RGB color space.

    :param res: numpy.array [H,W,D]
        Output of the network, per-pixel dense descriptor
    :param stats: dict, with fields ['min', 'max', 'mean'], which are used to normalize descriptor
    :return: numpy.array
        normalized descriptor
    """

    if stats is None:
        res_min = res.min()
        res_max = res.max()
    else:
        res_min = np.array(stats['min'])
        res_max = np.array(stats['max'])

    normed_res = np.clip(res, res_min, res_max)
    eps = 1e-10
    scale = (res_max - res_min) + eps
    normed_res = (normed_res - res_min) / scale
    return normed_res


import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.io
from PIL import Image


def get_color_mask(object_index, nc=None):
    """Colors each index differently. Useful for visualizing semantic masks.

    @param object_index: a [H x W] numpy array of ints from {0, ..., nc-1}
    @param nc: total number of colors. If None, this will be inferred by masks
    """
    object_index = object_index.astype(int)

    if nc is None:
        NUM_COLORS = object_index.max() + 1
    else:
        NUM_COLORS = nc

    cm = plt.get_cmap('gist_rainbow')
    colors = [cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)]

    color_mask = np.zeros(object_index.shape + (3, )).astype(np.uint8)
    for i in np.unique(object_index):
        if i == 0 or i == -1:
            continue
        color_mask[object_index == i, :] = np.array(colors[i][:3]) * 255

    return color_mask


def build_matrix_of_indices(height, width):
    """Builds a [height, width, 2] numpy array containing coordinates.

    @return: 3d array B s.t. B[..., 0] contains y-coordinates, B[..., 1] contains x-coordinates
    """
    return np.indices((height, width), dtype=np.float32).transpose(1, 2, 0)


def visualize_segmentation(im,
                           masks,
                           nc=None,
                           return_rgb=False,
                           save_dir=None):
    """Visualize segmentations nicely. Based on code from:
    https://github.com/roytseng-
    tw/Detectron.pytorch/blob/master/lib/utils/vis.py.

    @param im: a [H x W x 3] RGB image. numpy array of dtype np.uint8
    @param masks: a [H x W] numpy array of dtype np.uint8 with values in {0, ..., K}
    @param nc: total number of colors. If None, this will be inferred by masks
    """
    from matplotlib.patches import Polygon

    masks = masks.astype(int)
    im = im.copy()

    if not return_rgb:
        fig = plt.figure()
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.axis('off')
        fig.add_axes(ax)
        ax.imshow(im)

    # Generate color mask
    if nc is None:
        NUM_COLORS = masks.max() + 1
    else:
        NUM_COLORS = nc

    cm = plt.get_cmap('gist_rainbow')
    colors = [cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)]

    if not return_rgb:
        # matplotlib stuff
        fig = plt.figure()
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.axis('off')
        fig.add_axes(ax)

    # Mask
    imgMask = np.zeros(im.shape)

    # Draw color masks
    for i in np.unique(masks):
        if i == 0:  # background
            continue

        # Get the color mask
        color_mask = np.array(colors[i][:3])
        w_ratio = .4
        for c in range(3):
            color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio
        e = (masks == i)

        # Add to the mask
        imgMask[e] = color_mask

    # Add the mask to the image
    imgMask = (imgMask * 255).round().astype(np.uint8)
    im = cv2.addWeighted(im, 0.5, imgMask, 0.5, 0.0)

    # Draw mask contours
    for i in np.unique(masks):
        if i == 0:  # background
            continue

        # Get the color mask
        color_mask = np.array(colors[i][:3])
        w_ratio = .4
        for c in range(3):
            color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio
        e = (masks == i)

        # Find contours
        try:
            contour, hier = cv2.findContours(
                e.astype(np.uint8).copy(), cv2.RETR_CCOMP,
                cv2.CHAIN_APPROX_NONE)
        except:
            im2, contour, hier = cv2.findContours(
                e.astype(np.uint8).copy(), cv2.RETR_CCOMP,
                cv2.CHAIN_APPROX_NONE)

        # Plot the nice outline
        for c in contour:
            if save_dir is None and not return_rgb:
                polygon = Polygon(
                    c.reshape((-1, 2)),
                    fill=False,
                    facecolor=color_mask,
                    edgecolor='w',
                    linewidth=1.2,
                    alpha=0.5)
                ax.add_patch(polygon)
            else:
                cv2.drawContours(im, contour, -1, (255, 255, 255), 2)

    if save_dir is None and not return_rgb:
        ax.imshow(im)
        return fig
    elif return_rgb:
        return im
    elif save_dir is not None:
        # Save the image
        PIL_image = Image.fromarray(im)
        PIL_image.save(save_dir)
        return PIL_image


def _vis_minibatch_segmentation_final(image,
                                      label,
                                      out_label=None,
                                      out_label_refined=None,
                                      features=None,
                                      ind=None,
                                      selected_pixels=None,
                                      bbox=None):

    im_blob = image.cpu().numpy()
    num = im_blob.shape[0]
    height = im_blob.shape[2]
    width = im_blob.shape[3]

    if label is not None:
        label_blob = label.cpu().numpy()
    if out_label is not None:
        out_label_blob = out_label.cpu().numpy()
    if out_label_refined is not None:
        out_label_refined_blob = out_label_refined.cpu().numpy()

    m = 2
    n = 2
    for i in range(num):
        # image
        im = im_blob[i, :3, :, :].copy()
        # For RGB-D with UCN backbone
        im = im.transpose((1, 2, 0)) * 255.0
        im += [123.675, 116.280, 103.530]  # cfg.PIXEL_MEANS
        # For ResNet50 backbone
        # im = im.transpose((1, 2, 0)) * [58.395, 57.120, 57.375] + [123.675, 116.280, 103.530]
        im = im[:, :, (2, 1, 0)]
        im = np.clip(im, 0, 255)
        im = im.astype(np.uint8)
        fig = plt.figure()
        start = 1
        ax = fig.add_subplot(m, n, start)
        start += 1
        plt.imshow(im)
        ax.set_title('image')
        plt.axis('off')

        # feature
        if features is not None:
            im_feature = torch.cuda.FloatTensor(height, width, 3)
            for j in range(3):
                im_feature[:, :, j] = torch.sum(features[i, j::3, :, :], dim=0)
            im_feature = normalize_descriptor(
                im_feature.detach().cpu().numpy())
            im_feature *= 255
            im_feature = im_feature.astype(np.uint8)
            ax = fig.add_subplot(m, n, start)
            start += 1
            plt.imshow(im_feature)
            ax.set_title('feature map')
            plt.axis('off')

        # initial seeds
        if selected_pixels is not None:
            ax = fig.add_subplot(m, n, start)
            start += 1
            plt.imshow(im)
            ax.set_title('initial seeds')
            plt.axis('off')
            selected_indices = selected_pixels[i]
            for j in range(len(selected_indices)):
                index = selected_indices[j]
                y = index / width
                x = index % width
                plt.plot(x, y, 'ro', markersize=2.0)

        # intial mask
        mask = out_label_blob[i, :, :]
        im_label = visualize_segmentation(im, mask, return_rgb=True)
        ax = fig.add_subplot(m, n, start)
        start += 1
        plt.imshow(im_label)
        ax.set_title('initial label')
        plt.axis('off')

        # refined mask
        if out_label_refined is not None:
            mask = out_label_refined_blob[i, :, :]
            im_label = visualize_segmentation(im, mask, return_rgb=True)
            ax = fig.add_subplot(m, n, start)
            start += 1
            plt.imshow(im_label)
            ax.set_title('refined label')
            plt.axis('off')
        elif label is not None:
            # show gt label
            mask = label_blob[i, 0, :, :]
            im_label = visualize_segmentation(im, mask, return_rgb=True)
            ax = fig.add_subplot(m, n, start)
            start += 1
            plt.imshow(im_label)
            ax.set_title('gt label')
            plt.axis('off')

        if ind is not None:
            # mng = plt.get_current_fig_manager()
            # mng.resize(*mng.window.maxsize())
            # plt.pause(0.001)
            # plt.show(block=False)
            # filename = 'output/images/%06d.png' % ind
            # fig.savefig(filename)
            # plt.close()
            filename = './output/images/%06d.png' % ind
            fig.savefig(filename, bbox_inches='tight', dpi=1200)
            plt.close()
        else:
            plt.show()


# test a single sample
def test_sample(sample, network, network_crop, visualize=True):

    # construct input
    image = sample['image_color'].cuda()

    if 'label' in sample:
        label = sample['label'].cuda()
    else:
        label = None

    # run network
    features = network(image, label).detach()
    out_label, selected_pixels = clustering_features(features, num_seeds=100)

    # zoom in refinement
    out_label_refined = None
    if network_crop is not None:
        rgb_crop, out_label_crop, rois = crop_rois(image, out_label.clone(),
                                                   depth)
        if rgb_crop.shape[0] > 0:
            features_crop = network_crop(rgb_crop, out_label_crop)
            labels_crop, selected_pixels_crop = clustering_features(
                features_crop)
            out_label_refined, labels_crop = match_label_crop(
                out_label, labels_crop.cuda(), out_label_crop, rois)

    if visualize:
        bbox = None
        _vis_minibatch_segmentation_final(
            image,
            label,
            out_label,
            out_label_refined,
            features,
            selected_pixels=selected_pixels,
            bbox=bbox)
    return out_label, out_label_refined
