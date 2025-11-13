import sys
if "DROID-SLAM/droid_slam" not in sys.path: sys.path.append("DROID-SLAM/droid_slam")

# Generic imports
import torch

# Droid slam imports
from droid_slam.geom.graph_utils import graph_to_edge_list


def geodesic_loss(Gt, pred, graph, gamma=0.9) -> torch.Tensor:
    """
    :param Gt: (4, 4, N) Tensor of absolute poses (ground truth)
    :param pred: List of (4, 4, N) Tensors of absolute pose estimates at each iteration
    :param graph: Pose graph
    :param gamma: Decay factor for loss weight at each iteration
    :return: Geodesic loss between the estimated poses and the ground truth poses
    """

    # relative pose
    ii, jj, kk = graph_to_edge_list(graph)
    dP = Gt[:, jj] * Gt[:, ii].inv() # Convert to relative pose (From i -> j) Because P is the absolute pose (0 -> t)

    n = len(pred)
    geodesic_loss = 0.0

    # We loop over all iterations with a decaying loss weight. This forces the network to improve its estimates at each
    # iteration. Effectively learning to iteratively optimize its pose estimates.
    for i in range(n):
        w = gamma ** (n - i - 1) # Decay weight
        dG = pred[i][:, jj] * pred[i][:, ii].inv() # Convert to relative pose displacement


        # pose error
        # log maps to tangent space at identity so that its norm is the geodesic distance
        d = (dG * dP.inv()).log() # (Compute the displacement between the estimated relative pose and the ground truth relative pose)

        tau, phi = d.split([3, 3], dim=-1)
        geodesic_loss += w * (
                tau.norm(dim=-1).mean() +
                phi.norm(dim=-1).mean())

    return geodesic_loss