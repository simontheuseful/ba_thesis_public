/* Parameters
macro_grid: tensor of shape (mD, mH, mW), int32. -1 means the block is empty (skipped), otherwise
            it is the index of the block inside block_pool.
block_pool: tensor of shape (K, block_size, block_size, block_size, OUTPUT_DIM), values of the active blocks.
macro_shape: int[3] with mD, mH, mW.
block_size: side length (in voxels) of a block. Must be a power of 2 (enforced in
            TwoLevelGrid3D.__init__), so macro/local block coordinates can be computed with
            a shift/mask instead of a runtime division/modulo.
align_corners: int, whether the tensor grid represent corner values or voxel values
*/

// copied from experimental_grid3d.h
void scanline_interpolator(MAP_DECL, inout float dst[OUTPUT_DIM],
    float_ptr src_left, float_ptr src_right, float x_weight, float yz_weight) {
    for (int i=0; i<OUTPUT_DIM; i++)
        dst[i] += yz_weight * mix(src_left.data[i], src_right.data[i], x_weight);
}

// copied from experimental_grid3d.h but with empty check to support interpolation with empty blocks
void scanline_interpolator_sparse(MAP_DECL, inout float dst[OUTPUT_DIM],
    float_ptr src_left, bool empty_left, float_ptr src_right, bool empty_right,
    float x_weight, float yz_weight) {
    for (int i=0; i<OUTPUT_DIM; i++) {
        float l = empty_left  ? 0.0 : src_left.data[i];
        float r = empty_right ? 0.0 : src_right.data[i];
        dst[i] += yz_weight * mix(l, r, x_weight);
    }
}

GPUPtr resolve_block(MAP_DECL, ivec3 macro_c,
    int macro_stride_x, int macro_stride_y, int macro_stride_z, int bp_stride_block,
    out bool empty) {

    // use strides to get the 1d index of the macro block
    GPUPtr macro_ptr = load_tensor(parameters.macro_grid)
        + macro_c.x * macro_stride_x + macro_c.y * macro_stride_y + macro_c.z * macro_stride_z;

    // get the block index from the macro grid, -1 means empty
    int block_idx = int_ptr(macro_ptr).data[0];
    empty = block_idx < 0;

    // return empty pointer or the pointer to the block in the block pool
    return empty ? GPUPtr(0) : load_tensor(parameters.block_pool) + block_idx * bp_stride_block;
}

// this is used for slow checking if the 8 corners are not aligned in one block
GPUPtr resolve_corner(MAP_DECL, ivec3 macro_c, ivec3 local_c,
    int macro_stride_x, int macro_stride_y, int macro_stride_z,
    int bp_stride_x, int bp_stride_y, int bp_stride_z, int bp_stride_block,
    out bool empty) {

    // uses base logic from above but directly computes the edge by adding the local offset to the base pointer
    GPUPtr base = resolve_block(_this, macro_c, macro_stride_x, macro_stride_y, macro_stride_z, bp_stride_block, empty);
    return empty ? GPUPtr(0) : base + local_c.x * bp_stride_x + local_c.y * bp_stride_y + local_c.z * bp_stride_z;
}

FORWARD {
    // zeros the output array which is garbage until then
    for(int i=0; i<OUTPUT_DIM; i++) _output[i] = 0.0;

    // grid_size is the size of the grid in each dimension, grid_size = (W, H, D)
    // here (W, H, D) needs to be computed by the block count and size
    vec3 grid_size = vec3(
        float(parameters.macro_shape[2] * parameters.block_size),
        float(parameters.macro_shape[1] * parameters.block_size),
        float(parameters.macro_shape[0] * parameters.block_size));

    // algebraic optimization from the two steps before (found in experimental_grid3d.h). For me align_corners is probably always True = 1
    vec3 index_f = (vec3(_input[0], _input[1], _input[2]) * (grid_size - parameters.align_corners) + grid_size) * 0.5 - vec3(0.5);

    // bottom left voxel of the current sampling position
    ivec3 index0 = ivec3(floor(index_f));

    // top right voxel of the current sampling position
    ivec3 index1 = index0 + ivec3(1);

    // relative distance to the bottom left voxel, used for interpolation, alpha = (alpha_x, alpha_y, alpha_z)
    // note: this works for negative values of index_f as well, which is why I need to clamp the indices later
    vec3 alpha = index_f - vec3(index0);

    // clamp into the grid, so that we don't go out of bounds. Important for negative values of index_f
    index0 = clamp(index0, ivec3(0), ivec3(grid_size) - ivec3(1));
    index1 = clamp(index1, ivec3(0), ivec3(grid_size) - ivec3(1));

    int block_shift = findLSB(parameters.block_size);
    int block_mask = parameters.block_size - 1;

    // compute the macro block coordinates and local coordinates within the block for both index0 and index1
    ivec3 macro0 = index0 >> block_shift;
    ivec3 local0 = index0 & block_mask;
    ivec3 macro1 = index1 >> block_shift;
    ivec3 local1 = index1 & block_mask;

    // OUTPUT_DIM is always 1 here because we only store int32 (4 bytes) indices
    int macro_stride_x = 4;
    int macro_stride_y = parameters.macro_shape[2] * macro_stride_x;
    int macro_stride_z = parameters.macro_shape[1] * macro_stride_y;

    // for imagination I think about this backwards:
    // find the block with stride_block then jump to the exact z the exact y and lastly finely to x
    int bp_stride_x = OUTPUT_DIM * 4;
    int bp_stride_y = parameters.block_size * bp_stride_x;
    int bp_stride_z = parameters.block_size * bp_stride_y;
    int bp_stride_block = parameters.block_size * bp_stride_z;

    // check if bottom left and top right are in the same block
    // if so enter the fast implementation where only one lookup is needed
    if (macro0 == macro1) {

        // get the block pointer and check if it is empty, if so return early
        bool empty;
        GPUPtr base = resolve_block(_this, macro0, macro_stride_x, macro_stride_y, macro_stride_z, bp_stride_block, empty);
        if (empty) return; // here return enough _output already empty

        // in the found block look up the position of bottom left corner
        GPUPtr ptr = base + local0.x * bp_stride_x + local0.y * bp_stride_y + local0.z * bp_stride_z;
        int sx = bp_stride_x * (local1.x - local0.x);
        int sy = bp_stride_y * (local1.y - local0.y);
        int sz = bp_stride_z * (local1.z - local0.z);

        scanline_interpolator(_this, _output,
            float_ptr(ptr), float_ptr(ptr + sx),
            alpha.x, (1 - alpha.y) * (1 - alpha.z));
        ptr += sy;
        scanline_interpolator(_this, _output,
            float_ptr(ptr), float_ptr(ptr + sx),
            alpha.x, alpha.y * (1 - alpha.z));
        ptr += sz - sy;
        scanline_interpolator(_this, _output,
            float_ptr(ptr), float_ptr(ptr + sx),
            alpha.x, (1 - alpha.y) * alpha.z);
        ptr += sy;
        scanline_interpolator(_this, _output,
            float_ptr(ptr), float_ptr(ptr + sx),
            alpha.x, alpha.y * alpha.z);
        return;
    }

    bool e000, e100, e010, e110, e001, e101, e011, e111;
    GPUPtr p000 = resolve_corner(_this, ivec3(macro0.x, macro0.y, macro0.z), ivec3(local0.x, local0.y, local0.z),
        macro_stride_x, macro_stride_y, macro_stride_z, bp_stride_x, bp_stride_y, bp_stride_z, bp_stride_block, e000);
    GPUPtr p100 = resolve_corner(_this, ivec3(macro1.x, macro0.y, macro0.z), ivec3(local1.x, local0.y, local0.z),
        macro_stride_x, macro_stride_y, macro_stride_z, bp_stride_x, bp_stride_y, bp_stride_z, bp_stride_block, e100);
    GPUPtr p010 = resolve_corner(_this, ivec3(macro0.x, macro1.y, macro0.z), ivec3(local0.x, local1.y, local0.z),
        macro_stride_x, macro_stride_y, macro_stride_z, bp_stride_x, bp_stride_y, bp_stride_z, bp_stride_block, e010);
    GPUPtr p110 = resolve_corner(_this, ivec3(macro1.x, macro1.y, macro0.z), ivec3(local1.x, local1.y, local0.z),
        macro_stride_x, macro_stride_y, macro_stride_z, bp_stride_x, bp_stride_y, bp_stride_z, bp_stride_block, e110);
    GPUPtr p001 = resolve_corner(_this, ivec3(macro0.x, macro0.y, macro1.z), ivec3(local0.x, local0.y, local1.z),
        macro_stride_x, macro_stride_y, macro_stride_z, bp_stride_x, bp_stride_y, bp_stride_z, bp_stride_block, e001);
    GPUPtr p101 = resolve_corner(_this, ivec3(macro1.x, macro0.y, macro1.z), ivec3(local1.x, local0.y, local1.z),
        macro_stride_x, macro_stride_y, macro_stride_z, bp_stride_x, bp_stride_y, bp_stride_z, bp_stride_block, e101);
    GPUPtr p011 = resolve_corner(_this, ivec3(macro0.x, macro1.y, macro1.z), ivec3(local0.x, local1.y, local1.z),
        macro_stride_x, macro_stride_y, macro_stride_z, bp_stride_x, bp_stride_y, bp_stride_z, bp_stride_block, e011);
    GPUPtr p111 = resolve_corner(_this, ivec3(macro1.x, macro1.y, macro1.z), ivec3(local1.x, local1.y, local1.z),
        macro_stride_x, macro_stride_y, macro_stride_z, bp_stride_x, bp_stride_y, bp_stride_z, bp_stride_block, e111);

    scanline_interpolator_sparse(_this, _output,
        float_ptr(p000), e000, float_ptr(p100), e100,
        alpha.x, (1 - alpha.y) * (1 - alpha.z));
    scanline_interpolator_sparse(_this, _output,
        float_ptr(p010), e010, float_ptr(p110), e110,
        alpha.x, alpha.y * (1 - alpha.z));
    scanline_interpolator_sparse(_this, _output,
        float_ptr(p001), e001, float_ptr(p101), e101,
        alpha.x, (1 - alpha.y) * alpha.z);
    scanline_interpolator_sparse(_this, _output,
        float_ptr(p011), e011, float_ptr(p111), e111,
        alpha.x, alpha.y * alpha.z);
}

BACKWARD {
    NOT_SUPPORTED("future work");
}
