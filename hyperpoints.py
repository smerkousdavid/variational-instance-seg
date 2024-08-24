import torch

FEAT_DIM = 9
NUM_POINTS = 125  # max number of points


def unit_vector_rows(features: torch.Tensor,
                     eps: float = 1e-6,
                     vectorize: bool = True) -> torch.Tensor:
    """Converts the batched feature matrices into a new family of feature
    matrices that are vectorized with unit norms.

    Args:
      features (torch.Tensor): [B, N*P] or [B, N, P] batched feature matrix
      eps (float): if feature matrix has close to zero Frobenius norm add a small constant to prevent zero division
      vectorize (bool): if the shape is [B, N, P] (assumed to be a gram matrix) then vectorize the grams. If False then it will just throw an error
    Returns:
      torch.Tensor: [B, N*P]
    """

    if features.ndim != 2:
        raise ValueError('Expected matrix of [B, P]')

    # get Frobenius of norm each gram, ie euclidean of vectorized form
    norms = torch.linalg.vector_norm(features, ord=2, dim=1, keepdim=True)

    # create unit norm gram matrices
    return features.divide(norms + eps)


def pairwise_cossim(grams: torch.Tensor,
                    eps: float = 1e-6,
                    reduction: str = 'upper',
                    detach_right: bool = False) -> torch.Tensor:
    """Computes pairwise cosine similarity between batched grams.

    Args:
      grams (torch.Tensor): [B, N, P] batched gram matrix
      eps (float): if gram matrix has close to zero Frobenius norm add a small constant to prevent zero division

    Returns:
      torch.Tensor: vector of shape [B*(B-1)/2] with the pairwise cossine similarity
    """
    B = grams.shape[0]
    unit_vec_grams = unit_vector_rows(grams, eps)

    # compute pairwise dot products (just upper triangle of inner prod)
    gram_pd = torch.matmul(
        unit_vec_grams,
        unit_vec_grams.t().detach() if detach_right else unit_vec_grams.t())

    # get indices of upper right triangle
    # rows, cols = torch.triu_indices(B, B, offset=1)

    # only return relevent pairwise cossim values
    # gram_pd = gram_pd[rows, cols]
    if reduction == 'upper':
        # gram_pd = gram_pd[~torch.eye(*gram_pd.shape,dtype = torch.bool)]
        rows, cols = torch.triu_indices(B, B, offset=1)
        gram_pd = gram_pd[rows, cols]
    return gram_pd


def mean_hyperspherical_energy(features: torch.Tensor,
                               s: float = 0.0,
                               half_space: bool = False,
                               eps: float = 3e-5,
                               arc_eps: float = 3e-4,
                               offset: float = 0.0,
                               reduction: str = 'mean',
                               remove_diag: bool = True,
                               detach_right: bool = False,
                               abs_vals: bool = False,
                               use_exp=True):
    """Calculates the mean of the minimum geodesic hyperspherical energy.

    Args:
      features (torch.Tensor): model batched features of shape [B, N, P] or [B, N*P] where B is number of models, N is batch size of input, P is the number of features
      s (float): the Riesz s kernel parameter (for geodesic dist)
      half_space (bool): when True add negated gram vectors to use half-space MHE, if "centered" features are always positive this should be set to False.
      eps (float): value to prevent zero division with/normalize some division operators
      arc_eps (float): value to prevent arccos(theta) have nan/inf results (and their gradients exploding) by smoothing out theta just slightly by theta/(1 + arc_eps)
    Returns:
      torch.Tensor: mean of the minimum geodesic hyperspherical energy
    """

    # create unit vector grams
    B = features.shape[0]
    # N = grams.shape[1]
    unit_vec_grams = unit_vector_rows(features, eps=eps)

    # double gram features but with negated
    if half_space:
        unit_vec_grams = torch.concat(
            [unit_vec_grams, -unit_vec_grams.detach()], dim=0)

    # cossim (cos(theta)) between vectors
    costheta = torch.matmul(
        unit_vec_grams,
        unit_vec_grams.t().detach() if detach_right else unit_vec_grams.t())

    if abs_vals:
        costheta = torch.abs(costheta)

    # rows, cols = torch.triu_indices(B, B, offset=1)
    # gram_pd = geodesics[rows, cols]  # now just a tensor of the isolated pairwise
    # return (1.0 / (torch.arccos(gram_pd) + 1e-5)).mean()

    # get off diagonal geodesic elements only (i != j)
    if remove_diag:
        off_diag = costheta[~torch.eye(*costheta.shape, dtype=torch.bool)]
    else:
        off_diag = costheta
    # nij_geodesics = torch.arccos((off_diag - offset) / (1.0 + arc_eps))
    nij_geodesics = torch.arccos(off_diag / (1.0 + arc_eps))

    # print('APS', 1.0 + arc_eps)
    # print('GEOD', nij_geodesics, 'COSTHE', off_diag / (1.0 + arc_eps), torch.abs(off_diag / (1.0 + arc_eps)) > 1.0)

    # calculate energy from geodesic using rho
    # when s=1.0 this is similar to Coloumb's force with unit mass/force constant
    # if 0 we just get log of inverse
    if s == 0.0 or s == 1.0:
        energy = (1.0 + eps) / (nij_geodesics + eps)

        # apply logarithm on energy
        if s == 0.0:
            energy = torch.log(energy + eps)
    else:
        energy = torch.pow(
            nij_geodesics, -torch.tensor(
                s, device=nij_geodesics.device, dtype=nij_geodesics.dtype))
        # energy = (1.0 + eps) / (torch.pow(nij_geodesics + eps, torch.tensor(s, device=nij_geodesics.device, dtype=nij_geodesics.dtype)) + eps)
        # energy = torch.exp(-(3.6 * nij_geodesics))  # - 0.2))
        if use_exp:
            energy = torch.exp(-(s * nij_geodesics))  # - 0.2))
        else:
            energy = (1.0 + eps) / (
                torch.pow(
                    nij_geodesics + eps,
                    torch.tensor(
                        s,
                        device=nij_geodesics.device,
                        dtype=nij_geodesics.dtype)) + eps)

        # print('ENERGYYY', energy)

    # print('geod', torch.any(nij_geodesics < (1.0 - 1e-10)))
    # print('fin', torch.isfinite(torch.sum(energy)))

    # sum and normalize by number of pairings, which is just
    # the mean of all energies
    if reduction == 'mean':
        return energy.mean()
    return energy


# create a random set of features
feature_anchors = torch.zeros(NUM_POINTS, FEAT_DIM).normal_(
    mean=0.0, std=1.0).requires_grad_(True)

import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D

# now minimize hyperspherical energy
optim = torch.optim.Adam([feature_anchors], lr=0.05)
sched = torch.optim.lr_scheduler.StepLR(optim, step_size=1000, gamma=0.1)

for i in range(7000):
    optim.zero_grad()

    with torch.no_grad():
        feature_anchors.data = unit_vector_rows(feature_anchors.data, eps=0.0)

    energy = mean_hyperspherical_energy(
        feature_anchors,
        s=2.0,
        eps=1e-5,
        arc_eps=1e-5,
        reduction='mean',
        remove_diag=True,
        detach_right=True,
        use_exp=False)
    energy.backward()
    optim.step()
    print(
        'ITER', i, 'ENERGY', energy.item(), 'MARGIN COSIM',
        pairwise_cossim(
            feature_anchors, eps=0.0, reduction='upper',
            detach_right=True).max().item())
    sched.step()

print('Finished optimizing... now saving points')
with torch.no_grad():
    discrete_anchors = unit_vector_rows(feature_anchors.detach(), eps=0.0)

    # save the points and the expected margin between the points
    margin = pairwise_cossim(
        discrete_anchors, eps=0.0, reduction='upper',
        detach_right=True).max() - 1e-5
    print('FINAL MARGIN', margin.item())

    torch.save({
        'anchors': discrete_anchors,
        'margin': margin
    }, f'discrete_points_{FEAT_DIM}.pth')

# test render when feat dim is 3
if FEAT_DIM == 3:
    #Set colours and render
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create a sphere
    import numpy as np
    r = 1
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0 * pi:100j]
    x = r * sin(phi) * cos(theta)
    y = r * sin(phi) * sin(theta)
    z = r * cos(phi)
    ax.plot_surface(
        x, y, z, rstride=1, cstride=1, color='c', alpha=0.6, linewidth=0)

    # split feat into x, y, z
    x = feature_anchors[:, 0].detach().numpy()
    y = feature_anchors[:, 1].detach().numpy()
    z = feature_anchors[:, 2].detach().numpy()
    ax.scatter(x, y, z, color='k', s=20)
    fig.savefig('sphere.png')
