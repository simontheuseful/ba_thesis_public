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

    // strides, OUTPUT_DIM is always 1 so stride_x in this case is just 4 for 4 bytes of float
    int stride_x = OUTPUT_DIM * 4;
    int stride_y = int(parameters.shape[2]) * stride_x;
    int stride_z = int(parameters.shape[1]) * stride_y;

    // this is the actual 1d index to the bottom left voxel
    GPUPtr base_ptr = load_tensor(parameters.grid) + index0.x * stride_x + index0.y * stride_y + index0.z * stride_z;

    // this is either just the strides or 0 if index0 and index1 are equal
    // if this is 0 we are actually just interpolating with the same voxel
    int dx = (index1.x - index0.x) * stride_x;
    int dy = (index1.y - index0.y) * stride_y;
    int dz = (index1.z - index0.z) * stride_z;

    float_ptr v000 = float_ptr(base_ptr);
    float_ptr v100 = float_ptr(base_ptr + dx);
    float_ptr v010 = float_ptr(base_ptr + dy);
    float_ptr v110 = float_ptr(base_ptr + dy + dx);
    float_ptr v001 = float_ptr(base_ptr + dz);
    float_ptr v101 = float_ptr(base_ptr + dz + dx);
    float_ptr v011 = float_ptr(base_ptr + dz + dy);
    float_ptr v111 = float_ptr(base_ptr + dz + dy + dx);

    for (int i = 0; i < OUTPUT_DIM; i++) {
        float x00 = mix(v000.data[i], v100.data[i], alpha.x);
        float x10 = mix(v010.data[i], v110.data[i], alpha.x);
        float x01 = mix(v001.data[i], v101.data[i], alpha.x);
        float x11 = mix(v011.data[i], v111.data[i], alpha.x);

        float y0 = mix(x00, x10, alpha.y);
        float y1 = mix(x01, x11, alpha.y);

        _output[i] = mix(y0, y1, alpha.z);
    }
}

BACKWARD {
    NOT_SUPPORTED("future work");
}