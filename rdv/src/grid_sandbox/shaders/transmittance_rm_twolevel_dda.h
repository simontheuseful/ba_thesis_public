GPUPtr resolve_corner(MAP_DECL, ivec3 macro_c, ivec3 local_c,
    ivec3 macro_strides, ivec4 bp_strides) {

    GPUPtr macro_ptr = load_tensor(parameters.macro_grid)
        + macro_c.x * macro_strides.x + macro_c.y * macro_strides.y + macro_c.z * macro_strides.z;

    int block_idx = int_ptr(macro_ptr).data[0];
    if (block_idx < 0) return GPUPtr(0);

    return load_tensor(parameters.block_pool)
        + block_idx * bp_strides.w
        + local_c.x * bp_strides.x + local_c.y * bp_strides.y + local_c.z * bp_strides.z;
}

// Safely reads a channel value or returns 0.0 for empty blocks
float read_val(GPUPtr ptr, int channel) {
    return (ptr != GPUPtr(0)) ? float_ptr(ptr).data[channel] : 0.0;
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
    ivec3 macro_strides = ivec3(macro_stride_x, macro_stride_y, macro_stride_z);

    int bp_stride_x = 4; // OUTPUT_DIM == 1
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

    // use strides to resolve all 8 corner values
    // often they are in the same block so all m's are equal
    // no fast pass for macro0 == macro1: that branch made the code slower under warp (see two_level_grid3d.h)
    GPUPtr v000 = resolve_corner(_this, m000, l000, macro_strides, bp_strides);
    GPUPtr v100 = resolve_corner(_this, m100, l100, macro_strides, bp_strides);
    GPUPtr v010 = resolve_corner(_this, m010, l010, macro_strides, bp_strides);
    GPUPtr v110 = resolve_corner(_this, m110, l110, macro_strides, bp_strides);
    GPUPtr v001 = resolve_corner(_this, m001, l001, macro_strides, bp_strides);
    GPUPtr v101 = resolve_corner(_this, m101, l101, macro_strides, bp_strides);
    GPUPtr v011 = resolve_corner(_this, m011, l011, macro_strides, bp_strides);
    GPUPtr v111 = resolve_corner(_this, m111, l111, macro_strides, bp_strides);

    float x00 = mix(read_val(v000, 0), read_val(v100, 0), alpha.x);
    float x10 = mix(read_val(v010, 0), read_val(v110, 0), alpha.x);
    float x01 = mix(read_val(v001, 0), read_val(v101, 0), alpha.x);
    float x11 = mix(read_val(v011, 0), read_val(v111, 0), alpha.x);

    float y0 = mix(x00, x10, alpha.y);
    float y1 = mix(x01, x11, alpha.y);

    return mix(y0, y1, alpha.z);
}

/*
float transmittance_rm_twolevel_dda(MAP_DECL, vec3 x, vec3 w, float d, float step_size, float scale)
{
    vec3 grid_size = vec3(float(parameters.shape[2]), float(parameters.shape[1]), float(parameters.shape[0]));
    vec3 grid_scale = (grid_size - float(parameters.align_corners)) * 0.5;

    vec3 idx_origin = (x * (grid_size - float(parameters.align_corners)) + grid_size) * 0.5 - vec3(0.5);
    vec3 idx_dir = w * grid_scale;

    int block_shift = parameters.block_shift;
    int block_mask = parameters.block_size - 1;

    float tau = 0.0;
    float t = step_size * random(); // random jittering to reduce banding artifacts

    while (t < d) {
        vec3 index_f = idx_origin + idx_dir * t;
        ivec3 index0 = ivec3(floor(index_f));
        ivec3 macro0 = index0 >> block_shift;

        if (block_is_empty(_this, macro0)) {
            ivec3 voxel_min = index0 & ~block_mask; // find the minimum voxel index of the block
            vec3 node_min = vec3(voxel_min);
            vec3 node_max = node_min + vec3(float(parameters.block_size));

            // t_exit == the tMax this returns; we're already inside the block,
            // so we don't need its tMin (entry) side
            float block_tMin, t_exit;
            ray_box_intersection(idx_origin, idx_dir, node_min, node_max, block_tMin, t_exit);

            t = max(t_exit, t + 1.0e-4);
            continue;
        }

        float density = sample_density(_this, x + w * t, grid_size, parameters.align_corners);
        tau += density * step_size;

        if (tau * scale > 20.0) return 0.0;

        t += step_size;
    }
    return exp(-tau * scale);
}
*/

// this is 3dda from this github repository: https://github.com/DeadlockCode/voxel_ray_traversal
// it is a implementation of Amanatides and Woo (1987) "A Fast Voxel Traversal Algorithm for Ray Tracing"
float transmittance_rm_twolevel_dda(MAP_DECL, vec3 x, vec3 w, float d, float step_size, float scale)
{
    vec3 grid_size = vec3(float(parameters.shape[2]), float(parameters.shape[1]), float(parameters.shape[0]));
    vec3 grid_scale = (grid_size - float(parameters.align_corners)) * 0.5;

    ivec3 macro_dims = ivec3(parameters.macro_shape[2], parameters.macro_shape[1], parameters.macro_shape[0]);
    float block_size_f = float(parameters.block_size);

    // ray in voxel space
    vec3 idx_origin = (x * (grid_size - float(parameters.align_corners)) + grid_size) * 0.5 - vec3(0.5);

    // ray in macro block space
    vec3 ray_org = idx_origin / block_size_f;
    vec3 ray_dir = (w * grid_scale) / block_size_f;

    vec3 ray_inv = vec3(1.0) / ray_dir; // optimization: precompute the inverse of the ray direction to avoid division in the loop
    ivec3 coord = clamp(ivec3(floor(ray_org)), ivec3(0), macro_dims - ivec3(1)); // this is the macro block coordinate
    ivec3 step = ivec3(sign(ray_inv)); // step direction for each axis
    vec3 delta = abs(ray_inv); // distance to the next voxel boundary for each axis
    vec3 select = vec3(0.5) + 0.5 * sign(ray_inv); // backwards is just the actual coordinate and forwards is +1
    vec3 planes = vec3(coord) + select; // this is the next voxel boundary for each axis
    vec3 t_next = (planes - ray_org) * ray_inv; // solves ray_org + t*ray_dir = plane for t

    float tau = 0.0;
    float current_t = step_size * random(); // jittering

    while (current_t < d) {
        if (coord.x < 0 || coord.x >= macro_dims.x ||
            coord.y < 0 || coord.y >= macro_dims.y ||
            coord.z < 0 || coord.z >= macro_dims.z) {
            break;
        }

        float t_block_exit = min(t_next.x, min(t_next.y, t_next.z));

        if (!block_is_empty(_this, coord)) {
            float march_limit = min(d, t_block_exit);
            while (current_t < march_limit) {
                float density = sample_density(_this, x + w * current_t, grid_size, parameters.align_corners);
                tau += density * step_size;

                if (tau * scale > 20.0) return 0.0;

                current_t += step_size;
            }
        } else {
            current_t = max(t_block_exit, current_t);
        }

        if (t_next.x < t_next.y) {
            if (t_next.x < t_next.z) {
                coord.x += step.x;
                t_next.x += delta.x;
            } else {
                coord.z += step.z;
                t_next.z += delta.z;
            }
        } else {
            if (t_next.y < t_next.z) {
                coord.y += step.y;
                t_next.y += delta.y;
            } else {
                coord.z += step.z;
                t_next.z += delta.z;
            }
        }
    }

    return exp(-tau * scale);
}

FORWARD {
    // copy from transmittance_rm.h
    vec3 x = vec3(_input[0], _input[1], _input[2]); // ray origin in world space (camera pos)
    vec3 w = vec3(_input[3], _input[4], _input[5]); // ray direction in world space

    // Transform ray to local space
    mat4x3 M = mat4x3_ptr(load_tensor(parameters.transform)).data[0]; // from object to world space
    mat3 L = inverse(mat3(M[0].xyz, M[1].xyz, M[2].xyz));
    vec3 O = M[3].xyz;
    x = L * (x - O); // convert world position to object space position (use same x)
    vec3 wo = L * w; // convert world direction to object space direction
    // notice, in this point wo is not normalized, but is not a problem since we will use it for traversing the unnormalized density field, and we will normalize it later for sampling the phase function

    float tMin, tMax;
    ray_box_intersection(x, wo, tMin, tMax);
    if (tMax <= 0 || tMin > tMax) // ray points away from the volume
    {
        _output[0] = 1.0;
        return;
    }

    tMin = max(0, tMin); // clamp to 0 to avoid negative tMin

    x += wo * tMin; // initial position in object space
    float d = tMax - tMin; // max_t wrt wo

    _output[0] = transmittance_rm_twolevel_dda(_this, x, wo, d, parameters.step_size, parameters.extinction_scale);
}

BACKWARD {
    NOT_SUPPORTED("Backward not implemented for transmittance_rm_twolevel_dda");
}
