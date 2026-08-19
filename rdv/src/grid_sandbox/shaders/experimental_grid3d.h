/* Parameters
grid: tensor of shape (D, H, W, OUTPUT_DIM)
shape: int[3] with D, H, W
align_corners: int, whether the tensor grid represent corner values or voxel values
*/


// inout float dst is the output array that will be updated with the interpolated values
// src_left, src_right are pointers to the voxels on the left and right sides of the interpolation
// this calculates yz_weight * (1 - x_weight) * src_left + yz_weight * x_weight * src_right and adds it to dst
void scanline_interpolator(MAP_DECL, inout float dst[OUTPUT_DIM],
    float_ptr src_left, float_ptr src_right, float x_weight, float yz_weight) {
    for (int i=0; i<OUTPUT_DIM; i++)
        dst[i] += yz_weight * mix(src_left.data[i], src_right.data[i], x_weight);
}

FORWARD {
    // zeros the output array which is garbage until then
    for(int i=0; i<OUTPUT_DIM; i++) _output[i] = 0.0;

    // grid_size is the size of the grid in each dimension, grid_size = (W, H, D)
    vec3 grid_size = vec3(float(parameters.shape[2]), float(parameters.shape[1]), float(parameters.shape[0]));

    // takes the current sampling coordinate and maps it from [-1, 1] to [0, 1]
    //vec3 coord = vec3(_input[0], _input[1], _input[2]) * 0.5 + 0.5;

    // takes the current sampling coordinate and map it from [0, 1] to [0, grid_size] or [0, grid_size - 1] depending on align_corners
    //vec3 index_f = parameters.align_corners != 0 ? coord * (grid_size - vec3(1.0)) : coord * grid_size - vec3(0.5);

    // algebraic optimization from the two steps before. For me align_corners is probably always True = 1
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

    // each OUTPUT_DIM float is 4 bytes in size
    int stride_x = OUTPUT_DIM * 4;

    // we need the 1d index of the (x,y,z)
    int stride_y = int(parameters.shape[2]) * stride_x;
    int stride_z = int(parameters.shape[1]) * stride_y;

    // base address + stride to correct index
    GPUPtr ptr = load_tensor(parameters.grid) + index0.x * stride_x + index0.y * stride_y + index0.z * stride_z;

    // interpolate along the x axis, then along the y axis, then along the z axis
    stride_x *= (index1.x - index0.x);
    scanline_interpolator(_this, _output,
        float_ptr(ptr),
        float_ptr(ptr + stride_x),
        alpha.x, (1 - alpha.y)*(1 - alpha.z));

    stride_y *= (index1.y - index0.y);
    ptr += stride_y;
    scanline_interpolator(_this, _output,
        float_ptr(ptr),
        float_ptr(ptr + stride_x),
        alpha.x, alpha.y * (1 - alpha.z));

    stride_z *= (index1.z - index0.z);
    ptr += stride_z - stride_y;
    scanline_interpolator(_this, _output,
        float_ptr(ptr),
        float_ptr(ptr + stride_x),
        alpha.x, (1 - alpha.y)*alpha.z);

    ptr += stride_y;
    scanline_interpolator(_this, _output,
        float_ptr(ptr),
        float_ptr(ptr + stride_x),
        alpha.x, alpha.y*alpha.z);

    // _output now holds the interpolated value at the current sampling coordinate
}

BACKWARD {
    NOT_SUPPORTED("future work");
}