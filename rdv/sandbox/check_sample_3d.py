import torch
import rdv


grid = torch.rand(64, 64, 64, 32, device='cuda')  # A 3D grid with shape (D, H, W, 3)

sampler = rdv.Sample3DMap(rdv.tensor_clone(grid))

# Sample some points
input = torch.rand(100, 3) * 2 - 1  # Points in the range [-1, 1]
# points = torch.ones(1000, 3)   # corner cases points at the center
points = rdv.tensor_clone(input)

result = sampler(points)

# result.sum().backward()

# grad = result.grad.clone()

def sample3d_torch(grid, points):
    o = torch.nn.functional.grid_sample(
        grid.permute(3, 0, 1, 2).unsqueeze(0),  # (1, C, D, H, W
        points.unsqueeze(0).unsqueeze(0).unsqueeze(0),      # (1, N, 1, 3)
        mode='bilinear',
        align_corners=True
    )
    o = o.squeeze(0).squeeze(2).squeeze(1).permute(1, 0)  # (C, N)
    return o

expected = sample3d_torch(grid, points)

print((result.to(expected.device) - expected).abs().max())

points = torch.rand(10000000, 3) * 2 - 1  # Points in the range [-1, 1]
# points = torch.ones(1000, 3)   # corner cases points at the center
points = rdv.tensor_copy(points)
def test_several():
    for _ in range(1):
        result = sampler(points)
        # result = sample3d_torch(grid, points)
        torch.cuda.synchronize()
        print(result.sum())
        # torch.cuda.empty_cache()

import cProfile
cProfile.run('test_several()', sort='cumtime')

