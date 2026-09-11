/* Parameters
nvdb_data: tensor holding the raw bytes of a single-grid .nvdb file, grid type PNANOVDB_GRID_TYPE_FLOAT
           (one float value per active voxel)
shape: int[3] with the FULL logical (D, H, W) of the source volume the .nvdb was built from
align_corners: int, same convention as dense_grid3d.h/two_level_grid3d.h.
*/

// DISCLAIMER: some parts here are AI generated as they were too specific with rdvs pointer missmatch problem

// define PNANOVDB_BUF_CUSTOM so we use GPUPtr
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

// One corner lookup: walks the tree (cached in acc) for ijk and reads its float value.
float nanovdb_sample(pnanovdb_buf_t buf, pnanovdb_grid_type_t grid_type,
    inout pnanovdb_readaccessor_t acc, ivec3 ijk) {
    pnanovdb_address_t address = pnanovdb_readaccessor_get_value_address(grid_type, buf, acc, ijk);
    return pnanovdb_read_float(buf, address);
}

FORWARD {
    // load the .nvdb data into a pnanovdb_buf_t
    pnanovdb_buf_t buf = pnanovdb_make_buf(load_tensor(parameters.nvdb_data));

    // data starts at offset 0 because we already stripped away the header
    pnanovdb_grid_handle_t grid_handle = pnanovdb_grid_handle_t(pnanovdb_address_null());

    // single float per voxel
    pnanovdb_grid_type_t grid_type = PNANOVDB_GRID_TYPE_FLOAT;

    pnanovdb_tree_handle_t tree = pnanovdb_grid_get_tree(buf, grid_handle);
    pnanovdb_root_handle_t root = pnanovdb_tree_get_root(buf, tree);


    // copy from other grid representations
    vec3 grid_size = vec3(float(parameters.shape[2]), float(parameters.shape[1]), float(parameters.shape[0]));

    vec3 index_f = (vec3(_input[0], _input[1], _input[2]) * (grid_size - parameters.align_corners) + grid_size) * 0.5 - vec3(0.5);
    ivec3 index0 = ivec3(floor(index_f));
    ivec3 index1 = index0 + ivec3(1);
    vec3 alpha = index_f - vec3(index0);
    index0 = clamp(index0, ivec3(0), ivec3(grid_size) - ivec3(1));
    index1 = clamp(index1, ivec3(0), ivec3(grid_size) - ivec3(1));

    ivec3 c0 = index0;
    ivec3 c1 = index1;

    pnanovdb_readaccessor_t acc;
    pnanovdb_readaccessor_init(acc, root);

    // sample all 8 corners with the given coordinates
    float v000 = nanovdb_sample(buf, grid_type, acc, ivec3(c0.x, c0.y, c0.z));
    float v100 = nanovdb_sample(buf, grid_type, acc, ivec3(c1.x, c0.y, c0.z));
    float v010 = nanovdb_sample(buf, grid_type, acc, ivec3(c0.x, c1.y, c0.z));
    float v110 = nanovdb_sample(buf, grid_type, acc, ivec3(c1.x, c1.y, c0.z));
    float v001 = nanovdb_sample(buf, grid_type, acc, ivec3(c0.x, c0.y, c1.z));
    float v101 = nanovdb_sample(buf, grid_type, acc, ivec3(c1.x, c0.y, c1.z));
    float v011 = nanovdb_sample(buf, grid_type, acc, ivec3(c0.x, c1.y, c1.z));
    float v111 = nanovdb_sample(buf, grid_type, acc, ivec3(c1.x, c1.y, c1.z));

    // trilinear interpolation of the 8 corner values
    float x00 = mix(v000, v100, alpha.x);
    float x10 = mix(v010, v110, alpha.x);
    float x01 = mix(v001, v101, alpha.x);
    float x11 = mix(v011, v111, alpha.x);
    float y0 = mix(x00, x10, alpha.y);
    float y1 = mix(x01, x11, alpha.y);
    _output[0] = mix(y0, y1, alpha.z);

    // output now holds the interpolated value at the current sampling coordinate
}

BACKWARD {
    NOT_SUPPORTED("future work");
}