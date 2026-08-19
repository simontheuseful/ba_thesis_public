void scanline_interpolator(MAP_DECL, inout float dst[OUTPUT_DIM],
    float_ptr src_left, float_ptr src_right, float x_weight, float yz_weight) {
    for (int i=0; i<OUTPUT_DIM; i++)
        dst[i] += yz_weight * mix(src_left.data[i], src_right.data[i], x_weight);
}

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
    GPUPtr macro_ptr = load_tensor(parameters.macro_grid)
        + macro_c.x * macro_stride_x + macro_c.y * macro_stride_y + macro_c.z * macro_stride_z;
    int block_idx = int_ptr(macro_ptr).data[0];
    empty = block_idx < 0;
    return empty ? GPUPtr(0) : load_tensor(parameters.block_pool) + block_idx * bp_stride_block;
}

GPUPtr resolve_corner(MAP_DECL, ivec3 macro_c, ivec3 local_c,
    int macro_stride_x, int macro_stride_y, int macro_stride_z,
    int bp_stride_x, int bp_stride_y, int bp_stride_z, int bp_stride_block,
    out bool empty) {
    GPUPtr base = resolve_block(_this, macro_c, macro_stride_x, macro_stride_y, macro_stride_z, bp_stride_block, empty);
    return empty ? GPUPtr(0) : base + local_c.x * bp_stride_x + local_c.y * bp_stride_y + local_c.z * bp_stride_z;
}

bool block_is_empty(MAP_DECL, ivec3 macro_c) {
    int macro_stride_x = 4;
    int macro_stride_y = parameters.macro_shape[2] * macro_stride_x;
    int macro_stride_z = parameters.macro_shape[1] * macro_stride_y;
    GPUPtr macro_ptr = load_tensor(parameters.macro_grid)
        + macro_c.x * macro_stride_x + macro_c.y * macro_stride_y + macro_c.z * macro_stride_z;
    return int_ptr(macro_ptr).data[0] < 0;
}

float sample_density(MAP_DECL, vec3 x, vec3 grid_size, int align_corners) {
    vec3 index_f = (x * (grid_size - align_corners) + grid_size) * 0.5 - vec3(0.5);
    ivec3 index0 = ivec3(floor(index_f));
    ivec3 index1 = index0 + ivec3(1);
    vec3 alpha = index_f - vec3(index0);
    index0 = clamp(index0, ivec3(0), ivec3(grid_size) - ivec3(1));
    index1 = clamp(index1, ivec3(0), ivec3(grid_size) - ivec3(1));

    int block_shift = findLSB(parameters.block_size);
    int block_mask = parameters.block_size - 1;

    ivec3 macro0 = index0 >> block_shift;
    ivec3 local0 = index0 & block_mask;
    ivec3 macro1 = index1 >> block_shift;
    ivec3 local1 = index1 & block_mask;

    int macro_stride_x = 4;
    int macro_stride_y = parameters.macro_shape[2] * macro_stride_x;
    int macro_stride_z = parameters.macro_shape[1] * macro_stride_y;

    int bp_stride_x = 4; // OUTPUT_DIM == 1
    int bp_stride_y = parameters.block_size * bp_stride_x;
    int bp_stride_z = parameters.block_size * bp_stride_y;
    int bp_stride_block = parameters.block_size * bp_stride_z;

    float density_out[1];
    density_out[0] = 0.0;

    if (macro0 == macro1) {
        bool empty;
        GPUPtr base = resolve_block(_this, macro0, macro_stride_x, macro_stride_y, macro_stride_z, bp_stride_block, empty);
        if (empty) return 0.0;
        GPUPtr ptr = base + local0.x * bp_stride_x + local0.y * bp_stride_y + local0.z * bp_stride_z;
        int sx = bp_stride_x * (local1.x - local0.x);
        int sy = bp_stride_y * (local1.y - local0.y);
        int sz = bp_stride_z * (local1.z - local0.z);
        scanline_interpolator(_this, density_out, float_ptr(ptr), float_ptr(ptr + sx), alpha.x, (1 - alpha.y) * (1 - alpha.z));
        ptr += sy;
        scanline_interpolator(_this, density_out, float_ptr(ptr), float_ptr(ptr + sx), alpha.x, alpha.y * (1 - alpha.z));
        ptr += sz - sy;
        scanline_interpolator(_this, density_out, float_ptr(ptr), float_ptr(ptr + sx), alpha.x, (1 - alpha.y) * alpha.z);
        ptr += sy;
        scanline_interpolator(_this, density_out, float_ptr(ptr), float_ptr(ptr + sx), alpha.x, alpha.y * alpha.z);
        return density_out[0];
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

    scanline_interpolator_sparse(_this, density_out, float_ptr(p000), e000, float_ptr(p100), e100, alpha.x, (1 - alpha.y) * (1 - alpha.z));
    scanline_interpolator_sparse(_this, density_out, float_ptr(p010), e010, float_ptr(p110), e110, alpha.x, alpha.y * (1 - alpha.z));
    scanline_interpolator_sparse(_this, density_out, float_ptr(p001), e001, float_ptr(p101), e101, alpha.x, (1 - alpha.y) * alpha.z);
    scanline_interpolator_sparse(_this, density_out, float_ptr(p011), e011, float_ptr(p111), e111, alpha.x, alpha.y * alpha.z);
    return density_out[0];
}

FORWARD {

    vec3 x = vec3(_input[0], _input[1], _input[2]); // ray origin in world space
    vec3 w = vec3(_input[3], _input[4], _input[5]); // ray direction in world space

    mat4x3 M = mat4x3_ptr(load_tensor(parameters.transform)).data[0]; // transformation matrix
    mat3 L = inverse(mat3(M[0].xyz, M[1].xyz, M[2].xyz)); // world to object space linear transform
    vec3 O = M[3].xyz; // object world space origin
    x = L * (x - O); // ray origin in object space
    vec3 wo = L * w; // ray direction in object space

    float tMin, tMax;
    ray_box_intersection(x, wo, tMin, tMax); // intersection with [-1, 1]^3 box in object space
    if (tMax <= 0 || tMin > tMax) { // ray miss, early exit
        _output[0] = 1.0;
        return;
    }
    tMin = max(0, tMin); // clamp to 0 to avoid negative tMin
    x += wo * tMin; // move ray origin to the entry point of the box
    float d = tMax - tMin; // distance to exit point of the box

    vec3 grid_size = vec3(
        float(parameters.macro_shape[2] * parameters.block_size),
        float(parameters.macro_shape[1] * parameters.block_size),
        float(parameters.macro_shape[0] * parameters.block_size));

    vec3 idx_scale = (grid_size - vec3(float(parameters.align_corners))) * 0.5;
    vec3 idx_offset = grid_size * 0.5 - vec3(0.5);
    vec3 idx_origin = x * idx_scale + idx_offset;
    vec3 idx_dir = wo * idx_scale;

    int block_size = parameters.block_size;

    float tau = 0.0;
    float t = parameters.step_size * random(); // random jittering to reduce banding artifacts
    while (t < d) {
        vec3 idx_pos = idx_origin + idx_dir * t;
        ivec3 ijk = ivec3(floor(idx_pos));
        ivec3 macro_c = ijk >> findLSB(block_size);

        if (block_is_empty(_this, macro_c)) {
            ivec3 voxel_min = ijk & ivec3(~(block_size - 1));
            vec3 node_min = vec3(voxel_min);
            vec3 node_max = node_min + vec3(float(block_size));
            vec3 t0 = (node_min - idx_origin) / idx_dir;
            vec3 t1 = (node_max - idx_origin) / idx_dir;
            vec3 t_exit3 = max(t0, t1);
            float t_exit = min(t_exit3.x, min(t_exit3.y, t_exit3.z));
            t = max(t_exit, t + 1.0e-4);
            continue;
        }

        float density = sample_density(_this, x + wo * t, grid_size, parameters.align_corners);
        tau += density * parameters.extinction_scale * parameters.step_size;
        if (tau > 20) { _output[0] = 0.0; return; }
        t += parameters.step_size;
    }
    _output[0] = exp(-tau);
}

BACKWARD {
    NOT_SUPPORTED("Backward not implemented for transmittance_rm_twolevel_dda");
}
