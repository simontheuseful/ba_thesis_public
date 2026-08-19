import rdv
import grid_implementations as imp
import torch
from vulky import datasets
import matplotlib.pyplot as plt
from utility import create_two_level_grid

block_size = 8

vol = datasets.Volumes.disney_cloud

D, H, W, C = vol.shape
D, H, W = (D // block_size) * block_size, (H // block_size) * block_size, (W // block_size) * block_size
vol = vol[:D, :H, :W, :]

plt.imshow(vol[:, :, vol.shape[2] // 2, 0].T.cpu())
plt.gca().axis('off')
plt.gca().invert_yaxis()
plt.show()


vol = rdv.tensor_copy(vol)
macro_grid, block_pool = create_two_level_grid(vol, block_size=block_size, threshold=1e-4)
bmin, bmax = rdv.grid3d_fit_box(vol.shape[:3])
scale, offset = rdv.unit2box(bmin, bmax)
transform = rdv.mat4x3.trs(offset=offset, scale=scale)

transmittance_map = imp.RaymarchingTransmittanceTwoLevelDDA(
    macro_grid, block_pool, block_size, step_size=0.001, transform=transform, extinction_scale=5.0
)

camera_poses = rdv.tensor_from(
    [
        [
            0.2, 0.3, -2.5, # position
            0.0, 0.0, 0.0,  # target
            0.0, 1.0, 0.0,  # up vector
        ]
    ]
)


sensor = rdv.Sensor(1, 512, 512,
                    samples_location=(rdv.SampleLocation.CORNER, rdv.SampleLocation.CENTER, rdv.SampleLocation.CENTER),
                    probes_map=rdv.CameraProbes(camera_poses=camera_poses)
                    )

transmittance_view = sensor.view(transmittance_map)

with torch.no_grad():
    transmittance_img = transmittance_view.capture()

#view the transmittance image
plt.imshow(transmittance_img[0].cpu(), vmin=0.0, vmax=1.0, cmap='Blues_r')
plt.gca().axis('off')
plt.gca().invert_yaxis()
plt.tight_layout(pad=0.0)
plt.show()
