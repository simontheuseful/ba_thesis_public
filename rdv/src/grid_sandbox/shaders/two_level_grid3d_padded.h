/* Parameters
macro_grid: tensor of shape (mD, mH, mW), int32. -1 means the block is empty (skipped), otherwise
            it is the index of the block inside block_pool.
block_pool: tensor of shape (K, block_size+1, block_size+1, block_size+1, OUTPUT_DIM). Each block
            stores its own block_size^3 voxels PLUS one extra layer copied in from its +x/+y/+z
            neighbours, so every trilinear cell inside a block's own region is fully resolvable
            from this one block -- see create_two_level_grid_padded in utility.py.
macro_shape: int[3] with mD, mH, mW.
block_size: side length (in voxels) of a block's own region. Must be a power of 2. (block_pool
            blocks are block_size+1 wide per axis, to hold the padding.)
block_shift: log2(block_size)
align_corners: int, whether the tensor grid represents corner values or voxel values
*/

FORWARD {
    vec3 grid_size = vec3(float(parameters.shape[2]), float(parameters.shape[1]), float(parameters.shape[0]));

    vec3 index_f = (vec3(_input[0], _input[1], _input[2]) * (grid_size - parameters.align_corners) + grid_size) * 0.5 - vec3(0.5);

    ivec3 index0 = ivec3(floor(index_f));
    vec3 alpha = index_f - vec3(index0);

    // clamped exactly like the unpadded grid -- a point on the outer edge of the volume never
    // actually reaches into a block's padding layer, so that layer's true-boundary value (zero,
    // see create_two_level_grid_padded) is never sampled.
    index0 = clamp(index0, ivec3(0), ivec3(grid_size) - ivec3(1));

    int block_shift = parameters.block_shift;
    int block_mask = parameters.block_size - 1;
    int padded_size = parameters.block_size + 1;

    // one macro lookup only: local0 and local0+1 are both guaranteed to fit inside this block's
    // own padded data, so there is no macro1/local1 to resolve separately.
    ivec3 macro0 = index0 >> block_shift;
    ivec3 local0 = index0 & block_mask;

    int macro_stride_x = 4;
    int macro_stride_y = parameters.macro_shape[2] * macro_stride_x;
    int macro_stride_z = parameters.macro_shape[1] * macro_stride_y;

    GPUPtr macro_ptr = load_tensor(parameters.macro_grid)
        + macro0.x * macro_stride_x + macro0.y * macro_stride_y + macro0.z * macro_stride_z;
    int block_idx = int_ptr(macro_ptr).data[0];

    if (block_idx < 0) {
        for (int i = 0; i < OUTPUT_DIM; i++) _output[i] = 0.0;
        return;
    }

    int bp_stride_x = OUTPUT_DIM * 4;
    int bp_stride_y = padded_size * bp_stride_x;
    int bp_stride_z = padded_size * bp_stride_y;
    int bp_stride_block = padded_size * bp_stride_z;

    GPUPtr base = load_tensor(parameters.block_pool)
        + block_idx * bp_stride_block
        + local0.x * bp_stride_x + local0.y * bp_stride_y + local0.z * bp_stride_z;

    GPUPtr v000 = base;
    GPUPtr v100 = base + bp_stride_x;
    GPUPtr v010 = base + bp_stride_y;
    GPUPtr v110 = base + bp_stride_x + bp_stride_y;
    GPUPtr v001 = base + bp_stride_z;
    GPUPtr v101 = base + bp_stride_x + bp_stride_z;
    GPUPtr v011 = base + bp_stride_y + bp_stride_z;
    GPUPtr v111 = base + bp_stride_x + bp_stride_y + bp_stride_z;

    for (int i = 0; i < OUTPUT_DIM; i++) {
        float x00 = mix(float_ptr(v000).data[i], float_ptr(v100).data[i], alpha.x);
        float x10 = mix(float_ptr(v010).data[i], float_ptr(v110).data[i], alpha.x);
        float x01 = mix(float_ptr(v001).data[i], float_ptr(v101).data[i], alpha.x);
        float x11 = mix(float_ptr(v011).data[i], float_ptr(v111).data[i], alpha.x);

        float y0 = mix(x00, x10, alpha.y);
        float y1 = mix(x01, x11, alpha.y);

        _output[i] = mix(y0, y1, alpha.z);
    }
}

BACKWARD {
    NOT_SUPPORTED("future work");
}
