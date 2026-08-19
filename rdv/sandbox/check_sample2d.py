import torch
import os
os.environ['RDV_DEBUG'] = 'True'
import rdv
import matplotlib.pyplot as plt


grid = torch.rand(3, 3, 3, device='cuda')  # A 2D grid with shape (H, W, 8)

plt.imshow(grid[:, :, 0:3].cpu())
plt.gca().invert_yaxis()
plt.show()


sampler = rdv.Sample2DMap(grid, align_corners=False)
points = torch.cartesian_prod(torch.linspace(-1.0, 1.0, steps=100, device='cuda'),
                                torch.linspace(-1.0, 1.0, steps=100, device='cuda'))
points = points[:,[1,0]]  # Swap x and y for correct orientation
result_map = sampler(points.reshape(-1, 2))
result_map_img = result_map.reshape(100, 100, 3).cpu()
plt.imshow(result_map_img)
plt.gca().invert_yaxis()
plt.show()

def sample2d_torch(grid, points):
    return torch.nn.functional.grid_sample(
        grid.permute(2, 0, 1).unsqueeze(0),  # (1, C, H, W)
        points.unsqueeze(0).unsqueeze(2),      # (1, N, 1, 2)
        mode='bilinear',
        padding_mode='border',
        align_corners=False
    ).squeeze(0).squeeze(2).permute(1, 0)  # (C, N)


result_map = sample2d_torch(grid, points.reshape(-1, 2))
result_map_img = result_map.reshape(100, 100, 3).cpu()
plt.imshow(result_map_img)
plt.gca().invert_yaxis()
plt.show()
