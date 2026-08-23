#define PNANOVDB_GLSL
#define PNANOVDB_BUF_CUSTOM

struct pnanovdb_buf_t { GPUPtr base; };

pnanovdb_buf_t pnanovdb_make_buf(GPUPtr base) {
    pnanovdb_buf_t buf;
    buf.base = base;
    return buf;
}

uint pnanovdb_buf_read_uint32(pnanovdb_buf_t buf, uint byte_offset) {
    return uint_ptr(buf.base + byte_offset).data[0];
}

uvec2 pnanovdb_buf_read_uint64(pnanovdb_buf_t buf, uint byte_offset) {
    uvec2 ret;
    ret.x = pnanovdb_buf_read_uint32(buf, byte_offset + 0u);
    ret.y = pnanovdb_buf_read_uint32(buf, byte_offset + 4u);
    return ret;
}

void pnanovdb_buf_write_uint32(pnanovdb_buf_t buf, uint byte_offset, uint value) {}
void pnanovdb_buf_write_uint64(pnanovdb_buf_t buf, uint byte_offset, uvec2 value) {}

#define pnanovdb_grid_type_t uint
#define PNANOVDB_GRID_TYPE_GET(grid_typeIn, nameIn) pnanovdb_grid_type_constants[grid_typeIn].nameIn

#include "external/nanovdb/PNanoVDB.h"  // PNANOVDB_HDDA is on by default for GLSL

float nanovdb_read(pnanovdb_buf_t buf, pnanovdb_grid_type_t grid_type,
    inout pnanovdb_readaccessor_t acc, ivec3 ijk) {
    pnanovdb_address_t address = pnanovdb_readaccessor_get_value_address(grid_type, buf, acc, ijk);
    return pnanovdb_read_float(buf, address);
}

float sample_density(pnanovdb_buf_t buf, pnanovdb_grid_type_t grid_type,
    inout pnanovdb_readaccessor_t acc, vec3 x, vec3 grid_size, int align_corners) {

    vec3 index_f = (x * (grid_size - align_corners) + grid_size) * 0.5 - vec3(0.5);
    ivec3 index0 = ivec3(floor(index_f));
    ivec3 index1 = index0 + ivec3(1);
    vec3 alpha = index_f - vec3(index0);
    index0 = clamp(index0, ivec3(0), ivec3(grid_size) - ivec3(1));
    index1 = clamp(index1, ivec3(0), ivec3(grid_size) - ivec3(1));

    float v000 = nanovdb_read(buf, grid_type, acc, ivec3(index0.x, index0.y, index0.z));
    float v100 = nanovdb_read(buf, grid_type, acc, ivec3(index1.x, index0.y, index0.z));
    float v010 = nanovdb_read(buf, grid_type, acc, ivec3(index0.x, index1.y, index0.z));
    float v110 = nanovdb_read(buf, grid_type, acc, ivec3(index1.x, index1.y, index0.z));
    float v001 = nanovdb_read(buf, grid_type, acc, ivec3(index0.x, index0.y, index1.z));
    float v101 = nanovdb_read(buf, grid_type, acc, ivec3(index1.x, index0.y, index1.z));
    float v011 = nanovdb_read(buf, grid_type, acc, ivec3(index0.x, index1.y, index1.z));
    float v111 = nanovdb_read(buf, grid_type, acc, ivec3(index1.x, index1.y, index1.z));

    float x00 = mix(v000, v100, alpha.x);
    float x10 = mix(v010, v110, alpha.x);
    float x01 = mix(v001, v101, alpha.x);
    float x11 = mix(v011, v111, alpha.x);
    float y0 = mix(x00, x10, alpha.y);
    float y1 = mix(x01, x11, alpha.y);
    return mix(y0, y1, alpha.z);
}

/*
float transmittance_rm_nanovdb_dda(MAP_DECL,
    pnanovdb_buf_t buf,
    pnanovdb_grid_type_t grid_type,
    inout pnanovdb_readaccessor_t acc,
    vec3 x, vec3 w, float d, float step_size, float scale)
{

    vec3 grid_size = vec3(float(parameters.shape[2]), float(parameters.shape[1]), float(parameters.shape[0]));

    vec3 idx_scale = (grid_size - vec3(float(parameters.align_corners))) * 0.5;
    vec3 idx_offset = grid_size * 0.5 - vec3(0.5);
    vec3 idx_origin = x * idx_scale + idx_offset;
    vec3 idx_dir = w * idx_scale;

    float tau = 0.0;
    float t = step_size * random(); // random jittering to reduce banding artifacts

    while (t < d) {
        vec3 idx_pos = idx_origin + idx_dir * t;
        ivec3 ijk = ivec3(floor(idx_pos));
        int dim = int(pnanovdb_readaccessor_get_dim(grid_type, buf, acc, ijk));

        if (dim > 1) {
            ivec3 voxel_min = ijk & ivec3(~(dim - 1));
            vec3 node_min = vec3(voxel_min);
            vec3 node_max = node_min + vec3(float(dim));

            // t_exit == the tMax this returns; we're already inside the node,
            // so we don't need its tMin (entry) side
            float node_tMin, t_exit;
            ray_box_intersection(idx_origin, idx_dir, node_min, node_max, node_tMin, t_exit);

            t = max(t_exit, t + 1.0e-4);
            continue;
        }

        if (!bool(pnanovdb_readaccessor_is_active(grid_type, buf, acc, ijk))) {
            vec3 node_min = vec3(ijk);
            vec3 node_max = node_min + vec3(1.0);

            float voxel_tMin, t_exit;
            ray_box_intersection(idx_origin, idx_dir, node_min, node_max, voxel_tMin, t_exit);

            t = max(t_exit, t + 1.0e-4);
            continue;
        }

        float density = sample_density(buf, grid_type, acc, x + w * t, grid_size, parameters.align_corners);
        tau += density * scale * step_size;

        if (tau > 20.0) return 0.0;

        t += step_size;
    }
    return exp(-tau);
}
*/

float transmittance_rm_nanovdb_dda(MAP_DECL,
    pnanovdb_buf_t buf,
    pnanovdb_grid_type_t grid_type,
    inout pnanovdb_readaccessor_t acc,
    vec3 x, vec3 w, float d, float step_size, float scale)
{
    vec3 grid_size = vec3(float(parameters.shape[2]), float(parameters.shape[1]), float(parameters.shape[0]));

    vec3 idx_scale = (grid_size - vec3(float(parameters.align_corners))) * 0.5;
    vec3 idx_offset = grid_size * 0.5 - vec3(0.5);
    vec3 idx_origin = x * idx_scale + idx_offset;
    vec3 idx_dir = w * idx_scale;

    float tau = 0.0;
    float current_t = step_size * random(); // random jittering to reduce banding artifacts

    ivec3 ijk = ivec3(floor(idx_origin + idx_dir * current_t));
    int dim = int(pnanovdb_readaccessor_get_dim(grid_type, buf, acc, ijk));

    pnanovdb_hdda_t hdda;
    pnanovdb_hdda_init(hdda, idx_origin, current_t, idx_dir, d, dim);

    while (true) {
        float cell_exit = min(d, min(hdda.next.x, min(hdda.next.y, hdda.next.z)));

        if (hdda.dim == 1 && bool(pnanovdb_readaccessor_is_active(grid_type, buf, acc, hdda.voxel))) {
            while (current_t < cell_exit) {
                float density = sample_density(buf, grid_type, acc, x + w * current_t, grid_size, parameters.align_corners);
                tau += density * scale * step_size;

                if (tau > 20.0) return 0.0;

                current_t += step_size;
            }
        } else {
            current_t = max(cell_exit, current_t);
        }

        if (current_t >= d || !bool(pnanovdb_hdda_step(hdda))) {
            break;
        }

        vec3 pos = idx_origin + idx_dir * (hdda.tmin + 1.0e-4);
        ijk = ivec3(floor(pos));
        dim = int(pnanovdb_readaccessor_get_dim(grid_type, buf, acc, ijk));
        pnanovdb_hdda_update(hdda, idx_origin, idx_dir, dim);
    }

    return exp(-tau);
}

FORWARD {
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);

    mat4x3 M = mat4x3_ptr(load_tensor(parameters.transform)).data[0];
    mat3 L = inverse(mat3(M[0].xyz, M[1].xyz, M[2].xyz));
    vec3 O = M[3].xyz;
    x = L * (x - O);
    vec3 wo = L * w;

    float tMin, tMax;
    ray_box_intersection(x, wo, tMin, tMax);
    if (tMax <= 0 || tMin > tMax) {
        _output[0] = 1.0;
        return;
    }

    tMin = max(0.0, tMin);
    x += wo * tMin;
    float d = tMax - tMin;

    // NanoVDB Setup
    pnanovdb_buf_t buf = pnanovdb_make_buf(load_tensor(parameters.nvdb_data));
    pnanovdb_grid_handle_t grid_handle = pnanovdb_grid_handle_t(pnanovdb_address_null());
    pnanovdb_grid_type_t grid_type = PNANOVDB_GRID_TYPE_FLOAT;
    pnanovdb_tree_handle_t tree = pnanovdb_grid_get_tree(buf, grid_handle);
    pnanovdb_root_handle_t root = pnanovdb_tree_get_root(buf, tree);

    pnanovdb_readaccessor_t acc;
    pnanovdb_readaccessor_init(acc, root);

    _output[0] = transmittance_rm_nanovdb_dda(_this,
        buf, grid_type, acc,
        x, wo, d,
        parameters.step_size,
        parameters.extinction_scale
    );
}

BACKWARD {
    NOT_SUPPORTED("Backward not implemented for transmittance_rm_nanovdb_dda");
}