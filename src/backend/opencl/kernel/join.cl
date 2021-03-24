/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

kernel void join2(global T *d_out, const KParam out, const int jdim,
                  const global T *d_in1, const KParam in1,
                  const global T *d_in2, const KParam in2) {
    const int g0 = get_global_id(0);
    const int g1 = get_global_id(1);
    const int g2 = get_global_id(2);

    int idx_out = out.offset + g0 * out.strides[0] + g1 * out.strides[1] +
                  g2 * out.strides[2];

    const bool inside_in1 =
        (g0 < in1.dims[0]) && (g1 < in1.dims[1]) && (g2 < in1.dims[2]);
    if (inside_in1) {
        int idx_in = in1.offset + g0 * in1.strides[0] + g1 * in1.strides[1] +
                     g2 * in1.strides[2];
        d_out[idx_out]      = d_in1[idx_in];
        int g3              = 1;
        const int istrides3 = in1.strides[3];
        const int ostrides3 = out.strides[3];
        const int idims3    = in1.dims[3];
        int idx_out3        = idx_out;
        while (g3 < idims3) {
            idx_in += istrides3;
            T val = d_in1[idx_in];
            idx_out3 += ostrides3;
            d_out[idx_out3] = val;
            ++g3;
        }
    }
    idx_out += in1.dims[jdim] * out.strides[jdim];

    const bool inside_in2 =
        (g0 < in2.dims[0]) && (g1 < in2.dims[1]) && (g2 < in2.dims[2]);
    if (inside_in2) {
        int idx_in = in2.offset + g0 * in2.strides[0] + g1 * in2.strides[1] +
                     g2 * in2.strides[2];
        d_out[idx_out]      = d_in2[idx_in];
        int g3              = 1;
        const int istrides3 = in2.strides[3];
        const int ostrides3 = out.strides[3];
        const int idims3    = in2.dims[3];
        while (g3 < idims3) {
            idx_in += istrides3;
            T val = d_in2[idx_in];
            idx_out += ostrides3;
            d_out[idx_out] = val;
            ++g3;
        }
    }
}

kernel void joinN(global T *d_out, const KParam out, const int jdim,
                  const int ins, global const T *d_in0, const KParam in0,
                  global const T *d_in1, const KParam in1,
                  global const T *d_in2, const KParam in2,
                  global const T *d_in3, const KParam in3,
                  global const T *d_in4, const KParam in4) {
    const int g0 = get_global_id(0);
    const int g1 = get_global_id(1);
    const int g2 = get_global_id(2);

    global const T *d_in[5] = {d_in0, d_in1, d_in2, d_in3, d_in4};
    const KParam *in[5]     = {&in0, &in1, &in2, &in3, &in4};

    int idx_out = out.offset + g0 * out.strides[0] + g1 * out.strides[1] +
                  g2 * out.strides[2];

    const int ostrides3 = out.strides[3];
    const int ostridesj = out.strides[jdim];
    const int n         = ins > 5 ? 5 : ins;
    for (int i = 0; i < n; ++i) {
        const KParam *src     = in[i];
        const global T *d_src = d_in[i];
        const bool inside_in =
            (g0 < src->dims[0]) && (g1 < src->dims[1]) && (g2 < src->dims[2]);
        if (inside_in) {
            int idx_in = src->offset + g0 * src->strides[0] +
                         g1 * src->strides[1] + g2 * src->strides[2];
            d_out[idx_out]      = d_src[idx_in];
            int g3              = 1;
            const int istrides3 = src->strides[3];
            const int idims3    = src->dims[3];
            int idx_out3        = idx_out;
            while (g3 < idims3) {
                idx_in += istrides3;
                T val = d_src[idx_in];
                idx_out3 += ostrides3;
                d_out[idx_out3] = val;
                ++g3;
            }
        }
        idx_out += src->dims[jdim] * ostridesj;
    }
}
