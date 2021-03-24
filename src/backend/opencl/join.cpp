/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <common/half.hpp>
#include <err_opencl.hpp>
#include <join.hpp>
#include <kernel/join.hpp>

#include <algorithm>
#include <stdexcept>
#include <vector>

using af::dim4;
using common::half;
using std::transform;
using std::vector;

namespace opencl {
template<typename T>
Array<T> join(const int jdim, const Array<T> &first, const Array<T> &second) {
    // All dimensions except join dimension must be equal
    const dim4 &sdims = second.dims();
    // Compute output dims
    dim4 odims(first.dims());
    odims.dims[jdim] += sdims.dims[jdim];

    Array<T> out = createEmptyArray<T>(odims);
    kernel::join<T>(out, jdim, first, second);

    return out;
}

template<typename T>
Array<T> join(const int jdim, const vector<Array<T>> &inputs) {
    // All dimensions except join dimension must be equal
    // Compute output dims
    dim4 odims(inputs[0].dims());
    for (size_t i = 1, size = inputs.size(); i < size; ++i)
        odims.dims[jdim] += inputs[i].dims().dims[jdim];

    // Combine all evals into 1 preparation call
    vector<Array<T> *> input_ptrs(inputs.size());
    transform(
        begin(inputs), end(inputs), begin(input_ptrs),
        [](const Array<T> &input) { return const_cast<Array<T> *>(&input); });
    evalMultiple(input_ptrs);

    Array<T> out = createEmptyArray<T>(odims);
    vector<Param> ins(begin(inputs), end(inputs));
    kernel::join<T>(out, jdim, ins);
    return out;
}

#define INSTANTIATE(T)                                               \
    template Array<T> join<T>(const int jdim, const Array<T> &first, \
                              const Array<T> &second);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(cfloat)
INSTANTIATE(cdouble)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(intl)
INSTANTIATE(uintl)
INSTANTIATE(short)
INSTANTIATE(ushort)
INSTANTIATE(uchar)
INSTANTIATE(char)
INSTANTIATE(half)

#undef INSTANTIATE

#define INSTANTIATE(T) \
    template Array<T> join<T>(const int jdim, const vector<Array<T>> &inputs);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(cfloat)
INSTANTIATE(cdouble)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(intl)
INSTANTIATE(uintl)
INSTANTIATE(short)
INSTANTIATE(ushort)
INSTANTIATE(uchar)
INSTANTIATE(char)
INSTANTIATE(half)

#undef INSTANTIATE
}  // namespace opencl
