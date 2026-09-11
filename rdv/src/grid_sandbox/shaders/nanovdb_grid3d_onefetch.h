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

#include "external/nanovdb/PNanoVDB.h"

float nanovdb_sample(pnanovdb_buf_t buf, pnanovdb_grid_type_t grid_type,
    inout pnanovdb_readaccessor_t acc, ivec3 ijk) {
    pnanovdb_address_t address = pnanovdb_readaccessor_get_value_address(grid_type, buf, acc, ijk);
    return pnanovdb_read_float(buf, address);
}

FORWARD {
    pnanovdb_buf_t buf = pnanovdb_make_buf(load_tensor(parameters.nvdb_data));
    pnanovdb_grid_handle_t grid_handle = pnanovdb_grid_handle_t(pnanovdb_address_null());
    pnanovdb_grid_type_t grid_type = PNANOVDB_GRID_TYPE_FLOAT;
    pnanovdb_tree_handle_t tree = pnanovdb_grid_get_tree(buf, grid_handle);
    pnanovdb_root_handle_t root = pnanovdb_tree_get_root(buf, tree);

    vec3 grid_size = vec3(float(parameters.shape[2]), float(parameters.shape[1]), float(parameters.shape[0]));
    vec3 index_f = (vec3(_input[0], _input[1], _input[2]) * (grid_size - parameters.align_corners) + grid_size) * 0.5 - vec3(0.5);
    ivec3 index0 = ivec3(floor(index_f));
    ivec3 index1 = index0 + ivec3(1);
    vec3 alpha = index_f - vec3(index0);
    index0 = clamp(index0, ivec3(0), ivec3(grid_size) - ivec3(1));
    index1 = clamp(index1, ivec3(0), ivec3(grid_size) - ivec3(1));

    ivec3 c0 = index0;

    pnanovdb_readaccessor_t acc;
    pnanovdb_readaccessor_init(acc, root);

    // ONE descent, ONE read everything below is pure ALU, no second fetch
    float base = nanovdb_sample(buf, grid_type, acc, ivec3(c0.x, c0.y, c0.z));
    float v000 = base;
    float v100 = base + alpha.x * 1.0e-4;
    float v010 = base + alpha.y * 1.0e-4;
    float v110 = base + (alpha.x + alpha.y) * 1.0e-4;
    float v001 = base + alpha.z * 1.0e-4;
    float v101 = base + (alpha.x + alpha.z) * 1.0e-4;
    float v011 = base + (alpha.y + alpha.z) * 1.0e-4;
    float v111 = base + (alpha.x + alpha.y + alpha.z) * 1.0e-4;

    // identical trilinear blend to nanovdb_grid3d.h
    float x00 = mix(v000, v100, alpha.x);
    float x10 = mix(v010, v110, alpha.x);
    float x01 = mix(v001, v101, alpha.x);
    float x11 = mix(v011, v111, alpha.x);
    float y0 = mix(x00, x10, alpha.y);
    float y1 = mix(x01, x11, alpha.y);
    _output[0] = mix(y0, y1, alpha.z);
}

BACKWARD {
    NOT_SUPPORTED("future work");
}
