import os
import numpy as np
import rdv
import grid_implementations as imp
import torch
from vulky import datasets
import matplotlib.pyplot as plt

NVDB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "disney_cloud_fp16.nvdb")

vol = datasets.Volumes.disney_cloud
D, H, W, C = vol.shape

plt.imshow(vol[:, :, vol.shape[2] // 2, 0].T.cpu())
plt.gca().axis('off')
plt.gca().invert_yaxis()
plt.show()

nvdb_bytes = np.fromfile(NVDB_PATH, dtype=np.uint8)
nvdb_tensor = rdv.tensor_copy(torch.from_numpy(nvdb_bytes))
grid = imp.NanoVDBGrid3D(nvdb_tensor, shape=(D, H, W), align_corners=True)

bmin, bmax = rdv.grid3d_fit_box((D, H, W))
scale, offset = rdv.unit2box(bmin, bmax)
transform = rdv.mat4x3.trs(offset=offset, scale=scale)

transmittance_map = rdv.RaymarchingTransmittance(
    extinction=grid * 5,
    step_size=0.001,
    transform=transform
)

camera_poses = rdv.tensor_from(
    [
        [
            0.2, 0.3, -2.5,  # position
            0.0, 0.0, 0.0,   # target
            0.0, 1.0, 0.0,   # up vector
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

# view the transmittance image
plt.imshow(transmittance_img[0].cpu(), vmin=0.0, vmax=1.0, cmap='Blues_r')
plt.gca().axis('off')
plt.gca().invert_yaxis()
plt.tight_layout(pad=0.0)
plt.show()
