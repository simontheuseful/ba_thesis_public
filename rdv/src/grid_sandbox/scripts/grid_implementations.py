import os as _os
import rdv
import torch
from utility import strip_nvdb_header
import math

_SHADERS_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "shaders")

class DenseGrid3D(rdv.Map):
    """
    Maps 3D coordinates x,y,z in range [-1, 1] to a regular grid of values
    at indices vz, vy, vx from a tenor (D, H, W, C).
    If align corner is true, the 0, 0, 0 is exactly the value 0,0,0 in the grid
    """
    __extension_info__ = dict(
        path=_os.path.join(_SHADERS_DIR, "dense_grid3d.h"), # The path to the file with the FORWARD and BACKWARD implementations
        parameters=dict(
            grid=torch.Tensor,  # inside code parameters.grid is a tensor
            shape=[3, int],  # inside code parameters.shape is a int[3]
            align_corners=int,  # inside code parameters.align_corners is an int (0 or 1)
        )
    )

    def __init__(self,
                 grid: rdv.TensorLike,
                 align_corners: bool | int = True,
                 # all maps must receive named arguments input_dim, output_dim, input_requires_grad and bw_uses_output
                 input_dim=3, output_dim=None, input_requires_grad=False, bw_uses_output=False):
        # grid is converted to a tensor in case other types are passed
        grid = rdv.ensure_tensor(grid, map_dim=4)
        # if output_dim is None it can be inferred from the last dimension in the tensor
        if output_dim is None:
            output_dim = grid.shape[-1]
        # output_dim must match last dimension of the tensor
        assert output_dim == grid.shape[-1]
        # input_dim must be 3 (x, y, z)
        assert input_dim == 3
        super().__init__(input_dim=input_dim, output_dim=output_dim, input_requires_grad=input_requires_grad,
                         bw_uses_output=bw_uses_output)
        # bind arguments to parameters
        # from the __extension_info__['parameters'] dict there are attributes
        # handling from grid, shape and align_corners
        self.grid = grid
        # arrays must be set by indexing
        for i in range(3):
            self.shape[i] = grid.shape[i]
        # internally boolean is not supported using int instead
        self.align_corners = int(align_corners)

    def clone(self,
              **kwargs) -> rdv.Map:
        """
        This method is important to make all autocast mechanism to work.
        It just recreates the map with the same bound parameters, changing potentially
        input_dim, output_dim in **kwargs, but that's automatic.
        """
        return DenseGrid3D(self.grid, self.align_corners, **kwargs)


class NullSampler3D(rdv.Map):
    """Diagnostic sampler. Reads the three input coordinates, one multiply-add,
    writes one density. No index math, no fetch. Times the floor set by
    dispatching the threads and streaming the input and output tensors of a point
    query. Same constructor as DenseGrid3D so build_grid can pass the volume
    unchanged; the shader ignores it. See scripts/other/measure_io_floor.py."""
    __extension_info__ = dict(
        path=_os.path.join(_SHADERS_DIR, "null_sampler.h"),
        parameters=dict(grid=torch.Tensor, shape=[3, int], align_corners=int),
    )

    def __init__(self, grid, align_corners=True,
                 input_dim=3, output_dim=None, input_requires_grad=False, bw_uses_output=False):
        grid = rdv.ensure_tensor(grid, map_dim=4)
        if output_dim is None:
            output_dim = grid.shape[-1]
        super().__init__(input_dim=input_dim, output_dim=output_dim,
                         input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output)
        self.grid = grid
        for i in range(3):
            self.shape[i] = grid.shape[i]
        self.align_corners = int(align_corners)

    def clone(self, **kwargs) -> rdv.Map:
        return NullSampler3D(self.grid, self.align_corners, **kwargs)


class TwoLevelGrid3D(rdv.Map):
    __extension_info__ = dict(
        path=_os.path.join(_SHADERS_DIR, "two_level_grid3d.h"),
        parameters=dict(
            macro_grid=torch.Tensor,
            block_pool=torch.Tensor,
            shape=[3, int],
            macro_shape=[3, int],
            block_size=int,
            align_corners=int,
            block_shift=int
        )
    )

    def __init__(self,
                 macro_grid: rdv.TensorLike,
                 block_pool: rdv.TensorLike,
                 block_size: int = 8,
                 align_corners: bool | int = True,
                 input_dim=3, output_dim=None, input_requires_grad=False, bw_uses_output=False):

        assert block_size > 0 and (block_size & (block_size - 1)) == 0, \
            f"block_size must be a power of 2, got {block_size}"

        macro_grid = rdv.ensure_tensor(macro_grid, map_dim=3)
        block_pool = rdv.ensure_tensor(block_pool, map_dim=5)

        if output_dim is None:
            output_dim = block_pool.shape[-1]

        super().__init__(input_dim=input_dim, output_dim=output_dim,
                         input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output)

        self.macro_grid = macro_grid
        self.block_pool = block_pool
        self.block_size = int(block_size)
        self.align_corners = int(align_corners)
        self.block_shift =int(math.log2(block_size))

        for i in range(3):
            self.macro_shape[i] = macro_grid.shape[i]
            self.shape[i] = macro_grid.shape[i] * block_size

    def clone(self, **kwargs) -> 'TwoLevelGrid3D':
        # KORREKTUR: align_corners beim Klonen mit übergeben
        return TwoLevelGrid3D(self.macro_grid, self.block_pool, self.block_size, self.align_corners, **kwargs)

class TwoLevelGrid3DPadded(rdv.Map):
    """
    Same two-level structure as TwoLevelGrid3D, but block_pool blocks are (block_size+1)^3:
    each block carries one extra layer of its +x/+y/+z neighbours' voxels, so every trilinear
    sample resolves from a single macro_grid/block_pool lookup instead of up to eight -- see
    create_two_level_grid_padded in utility.py and shaders/two_level_grid3d_padded.h.
    """
    __extension_info__ = dict(
        path=_os.path.join(_SHADERS_DIR, "two_level_grid3d_padded.h"),
        parameters=dict(
            macro_grid=torch.Tensor,
            block_pool=torch.Tensor,
            shape=[3, int],
            macro_shape=[3, int],
            block_size=int,
            align_corners=int,
            block_shift=int
        )
    )

    def __init__(self,
                 macro_grid: rdv.TensorLike,
                 block_pool: rdv.TensorLike,
                 block_size: int = 8,
                 align_corners: bool | int = True,
                 input_dim=3, output_dim=None, input_requires_grad=False, bw_uses_output=False):

        assert block_size > 0 and (block_size & (block_size - 1)) == 0, \
            f"block_size must be a power of 2, got {block_size}"

        macro_grid = rdv.ensure_tensor(macro_grid, map_dim=3)
        block_pool = rdv.ensure_tensor(block_pool, map_dim=5)

        if output_dim is None:
            output_dim = block_pool.shape[-1]

        super().__init__(input_dim=input_dim, output_dim=output_dim,
                         input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output)

        self.macro_grid = macro_grid
        self.block_pool = block_pool
        self.block_size = int(block_size)
        self.align_corners = int(align_corners)
        self.block_shift = int(math.log2(block_size))

        for i in range(3):
            self.macro_shape[i] = macro_grid.shape[i]
            self.shape[i] = macro_grid.shape[i] * block_size

    def clone(self, **kwargs) -> 'TwoLevelGrid3DPadded':
        return TwoLevelGrid3DPadded(self.macro_grid, self.block_pool, self.block_size, self.align_corners, **kwargs)

class NanoVDBGrid3D(rdv.Map):
    __extension_info__ = dict(
        path=_os.path.join(_SHADERS_DIR, "nanovdb_grid3d.h"),
        parameters=dict(
            nvdb_data=torch.Tensor,  # inside code parameters.nvdb_data is a tensor (raw GridData bytes)
            shape=[3, int],
            align_corners=int,
        )
    )

    def __init__(self,
                 nvdb_data: rdv.TensorLike,
                 shape,
                 # shape: (D, H, W) of the source volume the .nvdb was built from -- NOT inferred
                 # from the tree's active-voxel bounding box, since that can be smaller than the
                 # source volume on any axis (see nanovdb_grid3d.h's file header comment).
                 align_corners: bool | int = True,
                 # NanoVDB "float" grids are single-channel; kept as args for the common
                 # Map(**kwargs) calling convention, but only input_dim=3/output_dim=1 are valid.
                 input_dim=3, output_dim=1, input_requires_grad=False, bw_uses_output=False):

        nvdb_data = rdv.ensure_tensor(nvdb_data, map_dim=1)
        assert nvdb_data.dtype == torch.uint8, "nvdb_data must be the raw bytes of a .nvdb file (torch.uint8)"
        assert output_dim == 1, "NanoVDBGrid3D only supports single-channel (PNANOVDB_GRID_TYPE_FLOAT) grids"
        assert len(shape) == 3, "shape must be (D, H, W)"


        nvdb_data = strip_nvdb_header(nvdb_data)
        assert input_dim == 3
        super().__init__(input_dim=input_dim, output_dim=output_dim, input_requires_grad=input_requires_grad,
                         bw_uses_output=bw_uses_output)
        self.nvdb_data = nvdb_data
        self.align_corners = int(align_corners)
        for i in range(3):
            self.shape[i] = int(shape[i])

    def clone(self, **kwargs) -> 'NanoVDBGrid3D':
        return NanoVDBGrid3D(self.nvdb_data, (self.shape[0], self.shape[1], self.shape[2]), self.align_corners, **kwargs)

class NanoVDBGrid3DOneFetchNoTrilinear(rdv.Map):
    """Diagnostic sampler, same constructor and parameters as NanoVDBGrid3D but a single
    nearest-neighbour NanoVDB lookup instead of the eight-corner trilinear gather. One full
    accessor descent, one memory read, no interpolation at all. Isolates the descent cost
    from the per-corner fetch cost, see nanovdb_grid3d_onefetch_notrilinear.h. Compare
    against NanoVDBGrid3DOneFetch, which restores the trilinear arithmetic on top of the
    same single fetch."""
    __extension_info__ = dict(
        path=_os.path.join(_SHADERS_DIR, "nanovdb_grid3d_onefetch_notrilinear.h"),
        parameters=dict(
            nvdb_data=torch.Tensor,
            shape=[3, int],
            align_corners=int,
        )
    )

    def __init__(self,
                 nvdb_data: rdv.TensorLike,
                 shape,
                 align_corners: bool | int = True,
                 input_dim=3, output_dim=1, input_requires_grad=False, bw_uses_output=False):

        nvdb_data = rdv.ensure_tensor(nvdb_data, map_dim=1)
        assert nvdb_data.dtype == torch.uint8, "nvdb_data must be the raw bytes of a .nvdb file (torch.uint8)"
        assert output_dim == 1, "NanoVDBGrid3DOneFetchNoTrilinear only supports single-channel (PNANOVDB_GRID_TYPE_FLOAT) grids"
        assert len(shape) == 3, "shape must be (D, H, W)"

        nvdb_data = strip_nvdb_header(nvdb_data)
        assert input_dim == 3
        super().__init__(input_dim=input_dim, output_dim=output_dim, input_requires_grad=input_requires_grad,
                         bw_uses_output=bw_uses_output)
        self.nvdb_data = nvdb_data
        self.align_corners = int(align_corners)
        for i in range(3):
            self.shape[i] = int(shape[i])

    def clone(self, **kwargs) -> 'NanoVDBGrid3DOneFetchNoTrilinear':
        return NanoVDBGrid3DOneFetchNoTrilinear(
            self.nvdb_data, (self.shape[0], self.shape[1], self.shape[2]), self.align_corners, **kwargs)

class NanoVDBGrid3DOneFetch(rdv.Map):
    """Diagnostic sampler, same constructor and parameters as NanoVDBGrid3D. Runs the same
    trilinear blend (alpha computation, seven mix() calls) as NanoVDBGrid3D but from a
    single accessor descent and a single memory read, the other seven "corners" are derived
    from that one fetched value with ALU-only offsets, see nanovdb_grid3d_onefetch.h.
    Compared against NanoVDBGrid3D (eight real fetches, identical arithmetic) this isolates
    the cost of the seven extra fetches from the interpolation arithmetic. Compared against
    NanoVDBGrid3DOneFetchNoTrilinear (one fetch, no arithmetic) it isolates the arithmetic
    cost of the blend itself."""
    __extension_info__ = dict(
        path=_os.path.join(_SHADERS_DIR, "nanovdb_grid3d_onefetch.h"),
        parameters=dict(
            nvdb_data=torch.Tensor,
            shape=[3, int],
            align_corners=int,
        )
    )

    def __init__(self,
                 nvdb_data: rdv.TensorLike,
                 shape,
                 align_corners: bool | int = True,
                 input_dim=3, output_dim=1, input_requires_grad=False, bw_uses_output=False):

        nvdb_data = rdv.ensure_tensor(nvdb_data, map_dim=1)
        assert nvdb_data.dtype == torch.uint8, "nvdb_data must be the raw bytes of a .nvdb file (torch.uint8)"
        assert output_dim == 1, "NanoVDBGrid3DOneFetch only supports single-channel (PNANOVDB_GRID_TYPE_FLOAT) grids"
        assert len(shape) == 3, "shape must be (D, H, W)"

        nvdb_data = strip_nvdb_header(nvdb_data)
        assert input_dim == 3
        super().__init__(input_dim=input_dim, output_dim=output_dim, input_requires_grad=input_requires_grad,
                         bw_uses_output=bw_uses_output)
        self.nvdb_data = nvdb_data
        self.align_corners = int(align_corners)
        for i in range(3):
            self.shape[i] = int(shape[i])

    def clone(self, **kwargs) -> 'NanoVDBGrid3DOneFetch':
        return NanoVDBGrid3DOneFetch(
            self.nvdb_data, (self.shape[0], self.shape[1], self.shape[2]), self.align_corners, **kwargs)

class RaymarchingTransmittanceTwoLevelDDA(rdv.Map):
    __extension_info__ = dict(
        path=_os.path.join(_SHADERS_DIR, "transmittance_rm_two_level_dda.h"),
        parameters=dict(
            macro_grid=torch.Tensor,
            block_pool=torch.Tensor,
            shape=[3, int],
            macro_shape=[3, int],
            block_size=int,
            align_corners=int,
            step_size=float,
            transform=torch.Tensor,
            extinction_scale=float,
            block_shift=int
        )
    )

    def __init__(self,
                 macro_grid: rdv.TensorLike,
                 block_pool: rdv.TensorLike,
                 block_size: int = 8,
                 step_size: float = 0.005,
                 transform: rdv.TensorLike = None,
                 align_corners: bool | int = True,
                 extinction_scale: float = 1.0,
                 input_dim=None, output_dim=None, input_requires_grad=False, bw_uses_output=False):
        if transform is None:
            transform = rdv.mat4x3.trs()

        assert block_size > 0 and (block_size & (block_size - 1)) == 0

        macro_grid = rdv.ensure_tensor(macro_grid, map_dim=3)
        block_pool = rdv.ensure_tensor(block_pool, map_dim=5)
        assert block_pool.shape[-1] == 1
        transform = rdv.ensure_tensor(transform, map_dim=2)

        if input_dim is None:
            input_dim = 6
        output_dim = output_dim or 1
        assert output_dim == 1

        super().__init__(input_dim=input_dim, output_dim=output_dim, input_requires_grad=input_requires_grad,
                         bw_uses_output=bw_uses_output)
        self.macro_grid = macro_grid
        self.block_pool = block_pool
        self.block_size = int(block_size)
        self.align_corners = int(align_corners)
        self.step_size = step_size
        self.transform = transform
        self.extinction_scale = extinction_scale
        self.block_shift =int(math.log2(block_size))
        for i in range(3):
            self.macro_shape[i] = macro_grid.shape[i]
            self.shape[i] = macro_grid.shape[i] * block_size

    def clone(self, **kwargs) -> 'RaymarchingTransmittanceTwoLevelDDA':
        return RaymarchingTransmittanceTwoLevelDDA(
            self.macro_grid, self.block_pool, self.block_size,
            self.step_size, self.transform, self.align_corners, self.extinction_scale, **kwargs)

class RaymarchingTransmittanceTwoLevelDDAPadded(rdv.Map):
    """
    Same macro-level 3D-DDA traversal as RaymarchingTransmittanceTwoLevelDDA, but samples
    density from a padded block_pool (block_size+1 per axis, see TwoLevelGrid3DPadded /
    create_two_level_grid_padded) so each density sample resolves from a single macro_grid/
    block_pool lookup instead of up to eight.
    """
    __extension_info__ = dict(
        path=_os.path.join(_SHADERS_DIR, "transmittance_rm_two_level_dda_padded.h"),
        parameters=dict(
            macro_grid=torch.Tensor,
            block_pool=torch.Tensor,
            shape=[3, int],
            macro_shape=[3, int],
            block_size=int,
            align_corners=int,
            step_size=float,
            transform=torch.Tensor,
            extinction_scale=float,
            block_shift=int
        )
    )

    def __init__(self,
                 macro_grid: rdv.TensorLike,
                 block_pool: rdv.TensorLike,
                 block_size: int = 8,
                 step_size: float = 0.005,
                 transform: rdv.TensorLike = None,
                 align_corners: bool | int = True,
                 extinction_scale: float = 1.0,
                 input_dim=None, output_dim=None, input_requires_grad=False, bw_uses_output=False):
        if transform is None:
            transform = rdv.mat4x3.trs()

        assert block_size > 0 and (block_size & (block_size - 1)) == 0

        macro_grid = rdv.ensure_tensor(macro_grid, map_dim=3)
        block_pool = rdv.ensure_tensor(block_pool, map_dim=5)
        assert block_pool.shape[-1] == 1
        transform = rdv.ensure_tensor(transform, map_dim=2)

        if input_dim is None:
            input_dim = 6
        output_dim = output_dim or 1
        assert output_dim == 1

        super().__init__(input_dim=input_dim, output_dim=output_dim, input_requires_grad=input_requires_grad,
                         bw_uses_output=bw_uses_output)
        self.macro_grid = macro_grid
        self.block_pool = block_pool
        self.block_size = int(block_size)
        self.align_corners = int(align_corners)
        self.step_size = step_size
        self.transform = transform
        self.extinction_scale = extinction_scale
        self.block_shift = int(math.log2(block_size))
        for i in range(3):
            self.macro_shape[i] = macro_grid.shape[i]
            self.shape[i] = macro_grid.shape[i] * block_size

    def clone(self, **kwargs) -> 'RaymarchingTransmittanceTwoLevelDDAPadded':
        return RaymarchingTransmittanceTwoLevelDDAPadded(
            self.macro_grid, self.block_pool, self.block_size,
            self.step_size, self.transform, self.align_corners, self.extinction_scale, **kwargs)

class RaymarchingTransmittanceNanoVDBDDA(rdv.Map):
    __extension_info__ = dict(
        path=_os.path.join(_SHADERS_DIR, "transmittance_rm_nanovdb_dda.h"),
        parameters=dict(
            nvdb_data=torch.Tensor,
            shape=[3, int],
            align_corners=int,
            step_size=float,
            transform=torch.Tensor,
            extinction_scale=float,
        )
    )

    def __init__(self,
                 nvdb_data: rdv.TensorLike,
                 shape,  # (D, H, W) of the source volume the .nvdb was built from
                 step_size: float = 0.005,
                 transform: rdv.TensorLike = None,
                 align_corners: bool | int = True,
                 extinction_scale: float = 1.0,
                 input_dim=None, output_dim=None, input_requires_grad=False, bw_uses_output=False):
        if transform is None:
            transform = rdv.mat4x3.trs()

        nvdb_data = rdv.ensure_tensor(nvdb_data, map_dim=1)
        assert nvdb_data.dtype == torch.uint8
        assert len(shape) == 3

        nvdb_data = strip_nvdb_header(nvdb_data)
        transform = rdv.ensure_tensor(transform, map_dim=2)

        if input_dim is None:
            input_dim = 6
        output_dim = output_dim or 1
        assert output_dim == 1

        super().__init__(input_dim=input_dim, output_dim=output_dim, input_requires_grad=input_requires_grad,
                         bw_uses_output=bw_uses_output)
        self.nvdb_data = nvdb_data
        self.align_corners = int(align_corners)
        self.step_size = step_size
        self.transform = transform
        self.extinction_scale = extinction_scale
        for i in range(3):
            self.shape[i] = int(shape[i])

    def clone(self, **kwargs) -> 'RaymarchingTransmittanceNanoVDBDDA':
        return RaymarchingTransmittanceNanoVDBDDA(
            self.nvdb_data, (self.shape[0], self.shape[1], self.shape[2]),
            self.step_size, self.transform, self.align_corners, self.extinction_scale, **kwargs)

