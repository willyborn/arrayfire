/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

typedef struct {
    int dims[4];
} dims_t;

kernel void memCopy(global T *d_out, const dims_t ostrides, const uint ooffset,
                    global const T *d_in, const dims_t idims,
                    const dims_t istrides, const uint ioffset) {
    const int g0 = get_global_id(0);  // dim[0]
    const int g1 = get_global_id(1);  // dim[1]
    const int g2 = get_global_id(2);  // dim[2]
                                      // dim[3] is through loop

    const bool valid =
        (g0 < idims.dims[0]) && (g1 < idims.dims[1]) && (g2 < idims.dims[2]);
    if (valid) {
        int idx_in = ioffset + g0 * istrides.dims[0] + g1 * istrides.dims[1] +
                     g2 * istrides.dims[2];
        T val       = d_in[idx_in];
        int idx_out = ooffset + g0 * ostrides.dims[0] + g1 * ostrides.dims[1] +
                      g2 * ostrides.dims[2];
        d_out[idx_out]      = val;
        const int istrides3 = istrides.dims[3];
        const int ostrides3 = ostrides.dims[3];
        const int idims3    = idims.dims[3];
        // g3==0 is performed above
        int g3 = 1;
        while (g3 < idims3) {
            idx_in += istrides3;
            val = d_in[idx_in];
            idx_out += ostrides3;
            d_out[idx_out] = val;
            ++g3;
        }
    }
}
