GPUPtr resolve_corner(MAP_DECL, ivec3 macro_c, ivec3 local_c,
    ivec3 macro_strides, ivec4 bp_strides) {

    GPUPtr macro_ptr = load_tensor(parameters.macro_grid)
        + macro_c.x * macro_strides.x + macro_c.y * macro_strides.y + macro_c.z * macro_strides.z;

    int block_idx = int_ptr(macro_ptr).data[0];

    return load_tensor(parameters.block_pool)
        + block_idx * bp_strides.w
        + local_c.x * bp_strides.x
        + local_c.y * bp_strides.y
        + local_c.z * bp_strides.z;
}

FORWARD {
    vec3 grid_size = vec3(float(parameters.shape[2]), float(parameters.shape[1]), float(parameters.shape[0]));

    //vec3 coord = vec3(_input[0], _input[1], _input[2]) * 0.5 + 0.5;
    //vec3 index_f = parameters.align_corners != 0 ? coord * (grid_size - vec3(1.0)) : coord * grid_size - vec3(0.5);
    vec3 index_f = (vec3(_input[0], _input[1], _input[2]) * (grid_size - parameters.align_corners) + grid_size) * 0.5 - vec3(0.5);

    // bottom left and upper right voxels
    ivec3 index0 = ivec3(floor(index_f));
    ivec3 index1 = index0 + ivec3(1);

    // if ray is on (0,0) then alpha here is just 0
    vec3 alpha = index_f - vec3(index0);

    // if ray lands on the edge of the grid, index1 would be out of bounds so clamp it back to the grid
    // if ray actually lands on the edge, then index0 and index1 are equal here
    index0 = clamp(index0, ivec3(0), ivec3(grid_size) - ivec3(1));
    index1 = clamp(index1, ivec3(0), ivec3(grid_size) - ivec3(1));

    int block_shift = parameters.block_shift;
    int block_mask = parameters.block_size - 1;

    // this is just optimized
    // macro0 = index0 / block_size, local0 = index0 % block_size
    ivec3 macro0 = index0 >> block_shift;
    ivec3 local0 = index0 & block_mask;
    ivec3 macro1 = index1 >> block_shift;
    ivec3 local1 = index1 & block_mask;

    // strides to find 1d index in macro array for the block
    int macro_stride_x = 4;
    int macro_stride_y = parameters.macro_shape[2] * macro_stride_x;
    int macro_stride_z = parameters.macro_shape[1] * macro_stride_y;
    ivec3 macro_strides = ivec3(macro_stride_x, macro_stride_y, macro_stride_z);

    // strides to find the 1d index in block_pool for the voxel
    int bp_stride_x = OUTPUT_DIM * 4;
    int bp_stride_y = parameters.block_size * bp_stride_x;
    int bp_stride_z = parameters.block_size * bp_stride_y;
    int bp_stride_block = parameters.block_size * bp_stride_z;
    ivec4 bp_strides = ivec4(bp_stride_x, bp_stride_y, bp_stride_z, bp_stride_block);

    // 3d coordinates of all 8 blocks in which the corners are
    ivec3 m000 = ivec3(macro0.x, macro0.y, macro0.z);
    ivec3 m100 = ivec3(macro1.x, macro0.y, macro0.z);
    ivec3 m010 = ivec3(macro0.x, macro1.y, macro0.z);
    ivec3 m110 = ivec3(macro1.x, macro1.y, macro0.z);
    ivec3 m001 = ivec3(macro0.x, macro0.y, macro1.z);
    ivec3 m101 = ivec3(macro1.x, macro0.y, macro1.z);
    ivec3 m011 = ivec3(macro0.x, macro1.y, macro1.z);
    ivec3 m111 = ivec3(macro1.x, macro1.y, macro1.z);

    // all 8 local corners
    ivec3 l000 = ivec3(local0.x, local0.y, local0.z);
    ivec3 l100 = ivec3(local1.x, local0.y, local0.z);
    ivec3 l010 = ivec3(local0.x, local1.y, local0.z);
    ivec3 l110 = ivec3(local1.x, local1.y, local0.z);
    ivec3 l001 = ivec3(local0.x, local0.y, local1.z);
    ivec3 l101 = ivec3(local1.x, local0.y, local1.z);
    ivec3 l011 = ivec3(local0.x, local1.y, local1.z);
    ivec3 l111 = ivec3(local1.x, local1.y, local1.z);

    // use strides to resole all 8 corner values
    // often they are in the same block so all m's are equal
    GPUPtr v000 = resolve_corner(_this, m000, l000, macro_strides, bp_strides);
    GPUPtr v100 = resolve_corner(_this, m100, l100, macro_strides, bp_strides);
    GPUPtr v010 = resolve_corner(_this, m010, l010, macro_strides, bp_strides);
    GPUPtr v110 = resolve_corner(_this, m110, l110, macro_strides, bp_strides);
    GPUPtr v001 = resolve_corner(_this, m001, l001, macro_strides, bp_strides);
    GPUPtr v101 = resolve_corner(_this, m101, l101, macro_strides, bp_strides);
    GPUPtr v011 = resolve_corner(_this, m011, l011, macro_strides, bp_strides);
    GPUPtr v111 = resolve_corner(_this, m111, l111, macro_strides, bp_strides);

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