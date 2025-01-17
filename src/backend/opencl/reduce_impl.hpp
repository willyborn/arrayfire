/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <err_opencl.hpp>
#include <kernel/reduce.hpp>
#include <kernel/reduce_by_key.hpp>
#include <reduce.hpp>
#include <af/dim4.hpp>

#include <complex>

using af::dim4;
using std::swap;
namespace arrayfire {
namespace opencl {
template<af_op_t op, typename Ti, typename To>
Array<To> reduce(const Array<Ti> &in, const int dim, bool change_nan,
                 double nanval) {
    // Calling point for Array<T> backend
    ARG_ASSERT(1, dim >= 0 && dim < 4);
    DIM_ASSERT(0, in.ndims() > 0);

    dim4 odims    = in.dims();
    odims[dim]    = 1;
    Array<To> out = createEmptyArray<To>(odims);
    kernel::reduce<Ti, To, op>(out, in, dim, change_nan, nanval);
    return out;
}

template<af_op_t op, typename Ti, typename Tk, typename To>
void reduce_by_key(Array<Tk> &keys_out, Array<To> &vals_out,
                   const Array<Tk> &keys, const Array<Ti> &vals, const int dim,
                   bool change_nan, double nanval) {
    // Calling point for Array<T> backend
    ARG_ASSERT(4, dim >= 0 && dim < 4);
    DIM_ASSERT(3, vals.ndims() > 0);
    TYPE_ASSERT(keys.getType() == u32 || keys.getType() == s32);
    DIM_ASSERT(2, keys.isVector());
    DIM_ASSERT(2, keys.dims()[0] == vals.dims()[dim]);
    // keys_out will be replaced, so no assert
    // vals_out will be replaced, so no assert

    kernel::reduceByKey<op, Ti, Tk, To>(keys_out, vals_out, keys, vals, dim,
                                        change_nan, nanval);
}

template<af_op_t op, typename Ti, typename To>
Array<To> reduce_all(const Array<Ti> &in, bool change_nan, double nanval) {
    // Calling point for Array<T> backend
    DIM_ASSERT(0, in.ndims() > 0);

    Array<To> out = createEmptyArray<To>(1);
    kernel::reduceAll<Ti, To, op>(out, in, change_nan, nanval);
    return out;
}

}  // namespace opencl
}  // namespace arrayfire

#define INSTANTIATE(Op, Ti, To)                                                \
    template Array<To> reduce<Op, Ti, To>(const Array<Ti> &in, const int dim,  \
                                          bool change_nan, double nanval);     \
    template void reduce_by_key<Op, Ti, int, To>(                              \
        Array<int> & keys_out, Array<To> & vals_out, const Array<int> &keys,   \
        const Array<Ti> &vals, const int dim, bool change_nan, double nanval); \
    template void reduce_by_key<Op, Ti, uint, To>(                             \
        Array<uint> & keys_out, Array<To> & vals_out, const Array<uint> &keys, \
        const Array<Ti> &vals, const int dim, bool change_nan, double nanval); \
    template Array<To> reduce_all<Op, Ti, To>(const Array<Ti> &in,             \
                                              bool change_nan, double nanval);
