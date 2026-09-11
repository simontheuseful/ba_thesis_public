/* Diagnostic: nearest-neighbour NanoVDB sampler, one accessor descent and one
   memory read per sample, no trilinear interpolation at all (no alpha, no
   mix()). Compared against nanovdb_grid3d.h (one descent, eight reads, full
   trilinear blend) this isolates the descent cost from the per-corner fetch
   cost. See also nanovdb_grid3d_onefetch.h, which restores the full
   interpolation arithmetic on top of a single fetch, to further separate the
   arithmetic cost of the blend from the memory cost of the extra fetches. */
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

FORWARD {
    pnanovdb_buf_t buf = pnanovdb_make_buf(load_tensor(parameters.nvdb_data));
    pnanovdb_grid_handle_t grid_handle = pnanovdb_grid_handle_t(pnanovdb_address_null());
    pnanovdb_grid_type_t grid_type = PNANOVDB_GRID_TYPE_FLOAT;
    pnanovdb_tree_handle_t tree = pnanovdb_grid_get_tree(buf, grid_handle);
    pnanovdb_root_handle_t root = pnanovdb_tree_get_root(buf, tree);

    vec3 grid_size = vec3(float(parameters.shape[2]), float(parameters.shape[1]), float(parameters.shape[0]));
    vec3 index_f = (vec3(_input[0], _input[1], _input[2]) * (grid_size - parameters.align_corners) + grid_size) * 0.5 - vec3(0.5);
    ivec3 index0 = clamp(ivec3(floor(index_f)), ivec3(0), ivec3(grid_size) - ivec3(1));

    pnanovdb_readaccessor_t acc;
    pnanovdb_readaccessor_init(acc, root);
    pnanovdb_address_t a = pnanovdb_readaccessor_get_value_address(grid_type, buf, acc, index0);
    _output[0] = pnanovdb_read_float(buf, a);
}

BACKWARD {
    NOT_SUPPORTED("future work");
}
