# THIS IS ONLY USABLE ON LINUX AND WITH SOME DEPENDENCIES
# USED TO GENERATE A .nvdb FROM A .pt

#import torch
#import fvdb

#vol = torch.load("synthetic256.pt", map_location="cuda", weights_only=True)
#vol_3d = vol.squeeze(-1)
#vol_3d = vol_3d.permute(2, 1, 0).contiguous()  # (D,H,W) -> (W,H,D), matches nanovdb_grid3d.h's ijk convention
#mask = vol_3d > 1e-4

#grid = fvdb.Grid.from_dense(dense_dims=vol_3d.shape, mask=mask, device="cuda")
#ijk = grid.ijk
#values = vol_3d[ijk[:, 0], ijk[:, 1], ijk[:, 2]].unsqueeze(-1)

#grid.save_nanovdb("synthetic256.nvdb", data=values, name="density")