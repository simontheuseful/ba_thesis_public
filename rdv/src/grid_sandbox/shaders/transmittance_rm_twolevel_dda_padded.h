// Same macro-level 3D-DDA traversal as transmittance_rm_twolevel_dda.h (Amanatides & Woo 1987,
// https://github.com/DeadlockCode/voxel_ray_traversal), but sample_density() uses the padded
// block layout from two_level_grid3d_padded.h: one macro_grid/block_pool lookup per sample
// instead of up to eight -- see create_two_level_grid_padded in utility.py.

// block_pool[0] is the reserved all-zero block that empty macro cells point at (see
// create_two_level_grid_padded in utility.py); used here to skip marching through empty macro blocks.
bool block_is_empty(MAP_DECL, ivec3 macro_c) {
    int macro_stride_x = 4;
    int macro_stride_y = parameters.macro_shape[2] * macro_stride_x;
    int macro_stride_z = parameters.macro_shape[1] * macro_stride_y;
    GPUPtr macro_ptr = load_tensor(parameters.macro_grid)
        + macro_c.x * macro_stride_x + macro_c.y * macro_stride_y + macro_c.z * macro_stride_z;
    return int_ptr(macro_ptr).data[0] == 0;
}

float sample_density(MAP_DECL, vec3 x, vec3 grid_size, int align_corners) {
    vec3 index_f = (x * (grid_size - align_corners) + grid_size) * 0.5 - vec3(0.5);
    ivec3 index0 = ivec3(floor(index_f));
    vec3 alpha = index_f - vec3(index0);
    index0 = clamp(index0, ivec3(0), ivec3(grid_size) - ivec3(1));

    int block_shift = parameters.block_shift;
    int block_mask = parameters.block_size - 1;
    int padded_size = parameters.block_size + 1;

    // single macro lookup -- local0 and local0+1 both fit inside this one padded block
    ivec3 macro0 = index0 >> block_shift;
    ivec3 local0 = index0 & block_mask;

    int macro_stride_x = 4;
    int macro_stride_y = parameters.macro_shape[2] * macro_stride_x;
    int macro_stride_z = parameters.macro_shape[1] * macro_stride_y;

    GPUPtr macro_ptr = load_tensor(parameters.macro_grid)
        + macro0.x * macro_stride_x + macro0.y * macro_stride_y + macro0.z * macro_stride_z;
    int block_idx = int_ptr(macro_ptr).data[0];

    int bp_stride_x = 4; // OUTPUT_DIM == 1
    int bp_stride_y = padded_size * bp_stride_x;
    int bp_stride_z = padded_size * bp_stride_y;
    int bp_stride_block = padded_size * bp_stride_z;

    GPUPtr base = load_tensor(parameters.block_pool)
        + block_idx * bp_stride_block
        + local0.x * bp_stride_x + local0.y * bp_stride_y + local0.z * bp_stride_z;

    float v000 = float_ptr(base).data[0];
    float v100 = float_ptr(base + bp_stride_x).data[0];
    float v010 = float_ptr(base + bp_stride_y).data[0];
    float v110 = float_ptr(base + bp_stride_x + bp_stride_y).data[0];
    float v001 = float_ptr(base + bp_stride_z).data[0];
    float v101 = float_ptr(base + bp_stride_x + bp_stride_z).data[0];
    float v011 = float_ptr(base + bp_stride_y + bp_stride_z).data[0];
    float v111 = float_ptr(base + bp_stride_x + bp_stride_y + bp_stride_z).data[0];

    float x00 = mix(v000, v100, alpha.x);
    float x10 = mix(v010, v110, alpha.x);
    float x01 = mix(v001, v101, alpha.x);
    float x11 = mix(v011, v111, alpha.x);

    float y0 = mix(x00, x10, alpha.y);
    float y1 = mix(x01, x11, alpha.y);

    return mix(y0, y1, alpha.z);
}

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
    NOT_SUPPORTED("Backward not implemented for transmittance_rm_twolevel_dda_padded");
}
