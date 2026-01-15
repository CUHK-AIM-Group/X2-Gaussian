import itertools
import logging as log
from typing import Optional, Union, List, Dict, Sequence, Iterable, Collection, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_normalized_directions(directions):
    """SH encoding must be in the range [0, 1]

    Args:
        directions: batch of directions
    """
    return (directions + 1.0) / 2.0


def normalize_aabb(pts, aabb):
    return (pts - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1.0


# MODIFICATION 1: Implement a dedicated 1D linear interpolation function.
def interpolate_1d_linear(grid: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """
    Performs linear interpolation on a 1D grid.

    Args:
        grid (torch.Tensor): The 1D grid of features, shape [1, C, R].
        coords (torch.Tensor): The coordinates to sample, shape [N, 1]. Assumed to be in [-1, 1].

    Returns:
        torch.Tensor: The interpolated features, shape [N, C].
    """
    c, r = grid.shape[1], grid.shape[2]
    n = coords.shape[0]

    # Un-normalize coordinates from [-1, 1] to [0, R-1]
    # This corresponds to align_corners=True behavior in grid_sample
    coords_unnorm = (coords.squeeze(-1) + 1) / 2 * (r - 1)

    # Clamp coordinates to be within the grid boundaries (mimics padding_mode='border')
    coords_unnorm = torch.clamp(coords_unnorm, 0, r - 1)

    # Find the integer and fractional parts of the coordinates
    t_floor = coords_unnorm.floor().long()
    t_ceil = coords_unnorm.ceil().long()
    
    # Calculate interpolation weights (alpha)
    alpha = (coords_unnorm - t_floor).unsqueeze(1) # Shape [N, 1] for broadcasting

    # Fetch feature vectors from the grid at the floor and ceil indices
    # Grid shape is [1, C, R], so we squeeze and transpose for easy indexing.
    grid_flat = grid.squeeze(0).transpose(0, 1) # Shape [R, C]
    feat_floor = grid_flat[t_floor] # Shape [N, C]
    feat_ceil = grid_flat[t_ceil]   # Shape [N, C]

    # Perform the linear interpolation
    interp_feat = (1.0 - alpha) * feat_floor + alpha * feat_ceil
    
    return interp_feat



def grid_sample_wrapper(grid: torch.Tensor, coords: torch.Tensor, align_corners: bool = True) -> torch.Tensor:
    grid_dim = coords.shape[-1]

    if grid.dim() == grid_dim + 1:
        # no batch dimension present, need to add it
        grid = grid.unsqueeze(0)
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)

    # if grid_dim == 1:
    #     # Use our custom 1D interpolation function for the time axis
    #     interp = interpolate_1d_linear(grid, coords.squeeze(0))
    #     # The output of interpolate_1d_linear is already [N, C], which is what we need.
    #     # We might need to unsqueeze if a batch dimension is expected by the caller.
    #     if grid.shape[0] > 1: # If there was a batch dimension originally
    #          interp = interp.unsqueeze(0)
    #     return interp
    # elif grid_dim == 2 or grid_dim == 3:
    #     grid_sampler = F.grid_sample
    # else:
    #     raise NotImplementedError(f"Grid-sample was called with {grid_dim}D data but is only "
    #                               f"implemented for 1D, 2D and 3D data.")

    if grid_dim == 2 or grid_dim == 3:
        grid_sampler = F.grid_sample
    else:
        raise NotImplementedError(f"Grid-sample was called with {grid_dim}D data but is only "
                                  f"implemented for 2 and 3D data.")

    coords = coords.view([coords.shape[0]] + [1] * (grid_dim - 1) + list(coords.shape[1:]))
    B, feature_dim = grid.shape[:2]
    n = coords.shape[-2]
    interp = grid_sampler(
        grid,  # [B, feature_dim, reso, ...]
        coords,  # [B, 1, ..., n, grid_dim]
        align_corners=align_corners,
        mode='bilinear', padding_mode='border')
    interp = interp.view(B, feature_dim, n).transpose(-1, -2)  # [B, n, feature_dim]
    interp = interp.squeeze()  # [B?, n, feature_dim?]
    return interp

def init_grid_param(
        grid_nd: int,
        in_dim: int,
        out_dim: int,
        reso: Sequence[int],
        a: float = 0.1,
        b: float = 0.5):
    assert in_dim == len(reso), "Resolution must have same number of elements as input-dimension"
    has_time_planes = in_dim == 4
    assert grid_nd <= in_dim
    coo_combs = list(itertools.combinations(range(in_dim), grid_nd))   # [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    grid_coefs = nn.ParameterList()
    for ci, coo_comb in enumerate(coo_combs):
        # breakpoint()
        new_grid_coef = nn.Parameter(torch.empty(
            [1, out_dim] + [reso[cc] for cc in coo_comb[::-1]]
        ))
        if has_time_planes and 3 in coo_comb:  # Initialize time planes to 1
            nn.init.ones_(new_grid_coef)
        else:
            nn.init.uniform_(new_grid_coef, a=a, b=b)
        grid_coefs.append(new_grid_coef)

    return grid_coefs


# MODIFICATION 2: Rewrite the interpolation function for the new logic.
def interpolate_ms_features(pts: torch.Tensor,
                            ms_grids_spatial: Collection[Iterable[nn.Module]],
                            ms_grids_time: Collection[Iterable[nn.Module]],
                            concat_features: bool,
                            num_levels: Optional[int],
                            ) -> torch.Tensor:
    """
    Optimized interpolation function.
    Interpolates features from 3 spatial planes and 1 time axis,
    then multiplies the results.
    """
    spatial_coords = pts[..., :3]  # x, y, z
    time_coords = pts[..., 3:]   # t

    # Define coordinate combinations for the 3 spatial planes (xy, xz, yz)
    coo_combs_spatial = list(itertools.combinations(range(3), 2))

    if num_levels is None:
        num_levels = len(ms_grids_spatial)
    
    multi_scale_interp = [] if concat_features else 0.
    
    for scale_id in range(num_levels):
        grid_spatial = ms_grids_spatial[scale_id]
        grid_time = ms_grids_time[scale_id]
        
        # Output feature dimension is assumed to be the same for space and time grids
        feature_dim = grid_spatial[0].shape[1]

        # --- 1. Spatial Interpolation ---
        # Interpolate from each of the 3 spatial planes and multiply the results.
        interp_spatial_product = 1.
        for ci, coo_comb in enumerate(coo_combs_spatial):
            # grid_spatial[ci] corresponds to one of xy, xz, yz planes
            interp_out_plane = grid_sample_wrapper(
                grid_spatial[ci], spatial_coords[..., coo_comb]
            ).view(-1, feature_dim)
            interp_spatial_product = interp_spatial_product * interp_out_plane

        # breakpoint()

        # --- 2. Temporal Interpolation ---
        # Interpolate from the 1D time axis.
        # grid_time is a list containing a single 1D grid.
        interp_time = grid_sample_wrapper(
            grid_time[0], time_coords
        ).view(-1, feature_dim)

        # breakpoint()

        # --- 3. Combine spatial and temporal features ---
        # Multiply the spatial feature product with the temporal feature.
        interp_at_scale = interp_spatial_product * interp_time

        # --- 4. Aggregate features across different scales ---
        if concat_features:
            multi_scale_interp.append(interp_at_scale)
        else:
            multi_scale_interp = multi_scale_interp + interp_at_scale

    if concat_features:
        multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)
        
    return multi_scale_interp


class HexPlaneField(nn.Module):
    def __init__(
        self,
        
        bounds,
        planeconfig,
        multires
    ) -> None:
        super().__init__()
        aabb = torch.tensor([[bounds,bounds,bounds],
                             [-bounds,-bounds,-bounds]])
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        self.grid_config =  [planeconfig]
        self.multiscale_res_multipliers = multires
        self.concat_features = True

        # MODIFICATION 3: Initialize separate grids for spatial and temporal components.
        # self.grids = nn.ModuleList() # Old grid storage
        self.grids = nn.ModuleList()
        self.grids_time = nn.ModuleList()
        self.feat_dim = 0
        for res in self.multiscale_res_multipliers:
            # initialize coordinate grid
            config = self.grid_config[0].copy()
            # Resolution fix: multi-res only on spatial planes
            config["resolution"] = [
                r * res for r in config["resolution"][:3]
            ] + config["resolution"][3:]
            # Initialize 3 spatial planes (xy, xz, yz) from 3D coordinates (x,y,z)
            gp_spatial = init_grid_param(
                grid_nd=2,  # 2D planes
                in_dim=3,   # Input is 3D (x, y, z)
                out_dim=config["output_coordinate_dim"],
                reso=config["resolution"][:3],  # Use only spatial resolutions
            )
            self.grids.append(gp_spatial)

            # Initialize 1 time axis (t) from 1D coordinate (t)
            gp_time = init_grid_param(   
                grid_nd=2,  # 2D "plane"              1D "plane" (an axis)
                in_dim=2,   # Input is 2D (t, 0.5)               Input is 1D (t)
                out_dim=config["output_coordinate_dim"],
                reso=[config["resolution"][3], 1], # Use only temporal resolution
            )
            # Initialize time features to 1, which is a common practice.
            nn.init.ones_(gp_time[0])
            self.grids_time.append(gp_time)

           # Update feature dimension based on one scale's output dimension
            if self.concat_features:
                # The output dimension of our new interpolation logic for one scale
                # is `out_dim`. We concatenate these across scales.
                self.feat_dim += config["output_coordinate_dim"]
            else:
                self.feat_dim = config["output_coordinate_dim"]

        print(f"Initialized optimized model with {len(self.grids[0])} spatial planes and {len(self.grids_time[0])} time axes per scale.")
        print("Total feature dimension:", self.feat_dim)

        self.time_placeholder =  torch.tensor(0.5).cuda()

    @property
    def get_aabb(self):
        return self.aabb[0], self.aabb[1]
    
    def set_aabb(self,xyz_max, xyz_min):
        aabb = torch.tensor([
            xyz_max,
            xyz_min
        ],dtype=torch.float32)
        self.aabb = nn.Parameter(aabb,requires_grad=False)
        print("Voxel Plane: set aabb=",self.aabb)

    def get_density(self, pts: torch.Tensor, timestamps: Optional[torch.Tensor] = None):
        """Computes and returns the densities."""
        pts = normalize_aabb(pts, self.aabb)
        if timestamps is not None:
            # breakpoint()
            # pts = torch.cat((pts, timestamps), dim=-1)  # [n_rays, n_samples, 4]
            pts = torch.cat((pts, timestamps, self.time_placeholder.unsqueeze(0).repeat(pts.shape[0], 1)), dim=-1)  # [n_rays, n_samples, 4]

        pts = pts.reshape(-1, pts.shape[-1])

        # MODIFICATION 4: Call the new interpolation function with the new grid structures.
        features = interpolate_ms_features(
            pts,
            ms_grids_spatial=self.grids,
            ms_grids_time=self.grids_time,
            concat_features=self.concat_features,
            num_levels=None)
        
        if len(features.shape) == 1 or features.shape[0] == 0:
            # Handle cases where no points are processed, return a tensor with correct device and shape.
            features = torch.zeros((pts.shape[0], self.feat_dim), device=features.device)

        return features

    def forward(self,
                pts: torch.Tensor,
                timestamps: Optional[torch.Tensor] = None):

        features = self.get_density(pts, timestamps)

        return features
