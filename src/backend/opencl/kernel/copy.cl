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

inType scale(inType value, factorType factor) {
#if defined(inType_float2) || defined(inType_double2)
    return (inType)(value.s0 * factor, value.s1 * factor);
#else
    return (inType)(value * factor);
#endif
}

#if defined(outType_double2)

// complex double output begin
#if defined(inType_float2) || defined(inType_double2)
#define CONVERT(value) convert_double2(value)
#else
#define CONVERT(value) (double2)((value), 0.0)
#endif
// complex double output macro ends

#elif defined(outType_float2)

// complex float output begins
#if defined(inType_float2) || defined(inType_double2)
#define CONVERT(value) convert_float2(value)
#else
#define CONVERT(value) (float2)((value), 0.0f)
#endif
// complex float output macro ends

#else

// scalar output, hence no complex input
// just enforce regular casting
#define CONVERT(value) ((outType)(value))

#endif

kernel void reshapeCopy(global outType *out, const dims_t odims,
                        const dims_t ostrides, const uint ooffset,
                        global const inType *in, const dims_t idims,
                        const dims_t istrides, const uint ioffset,
                        const outType default_value, const factorType factor) {
    const int g0 = get_global_id(0);  // dim0 of OUT buffer, not in buffer!!
    const int g1 = get_global_id(1);  // dim1 of OUT buffer, not in buffer!!
    const int g2 = get_global_id(2);  // dim2 of OUT buffer, not in buffer!!

    const bool inside_out =
        (g0 < odims.dims[0]) && (g1 < odims.dims[1]) && (g2 < odims.dims[2]);
    if (inside_out) {
        int idx_in = ioffset + g0 * istrides.dims[0] + g1 * istrides.dims[1] +
                     g2 * istrides.dims[2];
#if SAME_DIMS
        outType val = CONVERT(scale(in[idx_in], factor));
        int idx_out = ooffset + g0 * ostrides.dims[0] + g1 * ostrides.dims[1] +
                      g2 * ostrides.dims[2];
        out[idx_out]        = val;
        const int istrides3 = istrides.dims[3];
        const int ostrides3 = ostrides.dims[3];
        const int odims3    = odims.dims[3];
        // g3=0 is performed above
        int g3 = 1;
        while (g3 < odims3) {
            idx_in += istrides3;
            val = CONVERT(scale(in[idx_in], factor));
            idx_out += ostrides3;
            out[idx_out] = val;
            ++g3;
        }
#else
        const bool inside_in = (g0 < idims.dims[0]) && (g1 < idims.dims[1]) &&
                               (g2 < idims.dims[2]);
        outType val =
            inside_in ? CONVERT(scale(in[idx_in], factor)) : default_value;
        int idx_out = ooffset + g0 * ostrides.dims[0] + g1 * ostrides.dims[1] +
                      g2 * ostrides.dims[2];
        out[idx_out] = val;
        const int istrides3 = istrides.dims[3];
        const int ostrides3 = ostrides.dims[3];
        const int odims3 = odims.dims[3];
        const int idims3 = inside_in ? min(idims.dims[3], odims3) : 0;
        // g3=0 is performed above
        int g3 = 1;
        while (g3 < idims3) {
            idx_in += istrides3;
            val = CONVERT(scale(in[idx_in], factor));
            idx_out += ostrides3;
            out[idx_out] = val;
            ++g3;
        }
        while (g3 < odims3) {
            // Here we are certain that we are outside inside_in
            idx_out += ostrides3;
            out[idx_out] = default_value;
            ++g3;
        }
#endif
    }
}