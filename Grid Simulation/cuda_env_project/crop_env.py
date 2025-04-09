import torch
import torch.nn as nn
import torch.nn.functional as F

def crop_env_around_agents(grid, obs_size=5, pop_id=0):
    EPS = 1e-6
    B, C, H, W = grid.shape     # Channel <-> Populations

    # Create a Mask for populated cells Mask size
    mask_populate = grid[:,pop_id,:,:] > EPS    # shape:  [B, H, W]

    # Add padding for the periodic bound condition
    padded_grid = nn.CircularPad2d(obs_size // 2)(grid)

    # Save patch around each cell
    all_patches = F.unfold(padded_grid, kernel_size=obs_size)  # shape: (B, C * obs^2, H * W)
    all_patches = all_patches.view(B, C, obs_size * obs_size, H, W)  # shape: [B, C, obs^2, H, W]
    patches_permuted = all_patches.permute(0, 3, 4, 1, 2)   # shape: [B, H, W, C, obs^2]

    # Select only patches in mask
    selected_patches = patches_permuted[mask_populate]    # shape: [N, C, obs^2] for N: size of selected pop

    # Small tests
    print(mask_populate.sum() == selected_patches.shape[0])     # Check that population size = num of saved patches
    center_index = (obs_size // 2) * obs_size + (obs_size // 2)
    print((selected_patches[:, 0, center_index] > 0).all())     # Check center element is populated

    # Save the coord of the patch s.t. coord[i] <-> selected_patches[i,...]
    coord = torch.nonzero(mask_populate, as_tuple=False)
    print(coord.shape[0] == selected_patches.shape[0])  # Check that the is the same n of coord and patches

    return selected_patches, coord


if __name__ == "__main__":

    #grid_tensor = torch.randn(5, 3, 10, 10).clamp(0) # in reality a cell is never populated by 2 pop at the same time
    grid_tensor = torch.zeros(5, 3, 10, 10)
    grid_tensor[0,0,0,0] = 1
    grid_tensor[0,0,1,1] = 1
    grid_tensor[0,0,2,2] = 1
    grid_tensor[0,0,4,5] = 1
    grid_tensor[0,2,0,9] = 2
    grid_tensor[0,2,8,9] = 3 # invisible si obs_size < 5
    grid_tensor[1,1,0,9] = 9 # insivible car autre batch
    #grid_tensor = torch.randn(1000, 3, 50, 50).clamp(0)

    # Check si on veux obs size = diametre (=> impaire) ou rayon
    patches, coords = crop_env_around_agents(grid_tensor, obs_size=5)
    print(patches)