/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <Param.hpp>
#include <common/dispatch.hpp>
#include <common/kernel_cache.hpp>
#include <common/traits.hpp>
#include <debug_opencl.hpp>
#include <kernel_headers/copy.hpp>
#include <kernel_headers/memcopy.hpp>
#include <traits.hpp>

#include <algorithm>
#include <string>
#include <vector>

using std::string;
using std::vector;

namespace opencl {
namespace kernel {
typedef struct {
    int dims[4];
} dims_t;

template<typename T>
void memcopy(const cl::Buffer &b_out, const dim4 &ostrides,
             const cl::Buffer &b_in, const dim4 &idims, const dim4 &istrides,
             const dim_t ioffset, const dim_t indims, const dim_t ooffset = 0) {
    dims_t idims_{
        static_cast<int>(idims.dims[0]), static_cast<int>(idims.dims[1]),
        static_cast<int>(idims.dims[2]), static_cast<int>(idims.dims[3])};
    dims_t istrides_{
        static_cast<int>(istrides.dims[0]), static_cast<int>(istrides.dims[1]),
        static_cast<int>(istrides.dims[2]), static_cast<int>(istrides.dims[3])};
    dims_t ostrides_{
        static_cast<int>(ostrides.dims[0]), static_cast<int>(ostrides.dims[1]),
        static_cast<int>(ostrides.dims[2]), static_cast<int>(ostrides.dims[3])};
    int indims_ = static_cast<int>(indims);

    bool isLinear = true;
    int elements  = (indims_ == 0) ? 0 : 1;
    for (int dim = 0; dim < indims_; ++dim) {
        isLinear &= (elements == istrides_.dims[dim]) &
                    (elements == ostrides_.dims[dim]);
        elements *= idims_.dims[dim];
    }
    if (elements > 0) {
        if (isLinear) {
            // Both input and output arrays are linear
            getQueue().enqueueCopyBuffer(
                b_in, b_out, ioffset * sizeof(T), ooffset * sizeof(T),
                elements * sizeof(T), nullptr, nullptr);
        } else {
            const vector<TemplateArg> targs = {
                TemplateTypename<T>(),
            };
            const vector<string> options = {
                DefineKeyValue(T, dtype_traits<T>::getName()),
                {getTypeBuildDefinition<T>()},
            };
            auto memCopy =
                common::getKernel("memCopy", {memcopy_cl_src}, targs, options);

            for (int c = 0; c < indims_; ++c) {
                // Eliminate the columns with 1, so that we have more
                // appropriate dimensions in the local & global parameters
                if (idims_.dims[c] == 1) {
                    for (int i = c; i < indims_ - 1; ++i) {
                        idims_.dims[i]    = idims_.dims[i + 1];
                        istrides_.dims[i] = istrides_.dims[i + 1];
                        ostrides_.dims[i] = ostrides_.dims[i + 1];
                    }
                    --indims_;
                    idims_.dims[indims_] = 1;
                    --c;  // Redo this column, since it is eliminated now!!
                }
            }
            // Increase work inside each thread (if last dim is free and a
            // minimum threads remain)
            if (elements >= 2048 * 2 && indims_ != AF_MAX_DIMS &&
                indims_ != 0) {
                for (int i : {3, 4, 5, 7, 11, 2}) {
                    if (elements >= 2048 * i &&
                        (idims_.dims[indims_ - 1] % i) == 0) {
                        idims_.dims[indims_ - 1] /= i;
                        idims_.dims[AF_MAX_DIMS - 1] = i;
                        for (int c = indims_; c < AF_MAX_DIMS; ++c) {
                            istrides_.dims[c] =
                                idims_.dims[c - 1] * istrides_.dims[c - 1];
                            ostrides_.dims[c] =
                                idims_.dims[c - 1] * ostrides_.dims[c - 1];
                        }
                        indims_ = AF_MAX_DIMS;
                        // Once is sufficient
                        break;
                    }
                }
            }

            // This kernel is memory bound, so focus on caching (dim0+++)
            const int dim0 = (idims_.dims[0] <= 128) ? 128 : 256;
            const int dim1 = (dim0 == 256) || (idims_.dims[1] & 0x1) ? 1 : 2;
            const int dim2 =
                (dim0 * dim1 == 256) || (idims_.dims[2] & 0x1) ? 1 : 2;
            const cl::NDRange local(dim0, dim1, dim2);
            const cl::NDRange global(dim0 * divup(idims_.dims[0], dim0),
                                     dim1 * divup(idims_.dims[1], dim1),
                                     dim2 * divup(idims_.dims[2], dim2));
            memCopy(cl::EnqueueArgs(getQueue(), global, local), b_out,
                    ostrides_, static_cast<unsigned>(ooffset), b_in, idims_,
                    istrides_, static_cast<unsigned>(ioffset));
            CL_DEBUG_FINISH(getQueue());
        }
    }
}

template<typename inType, typename outType>
void copy(const Param out, const Param in, const dim_t ondims,
          const outType default_value, const double factor,
          const bool same_dims) {
    dims_t idims_{
        static_cast<int>(in.info.dims[0]), static_cast<int>(in.info.dims[1]),
        static_cast<int>(in.info.dims[2]), static_cast<int>(in.info.dims[3])};
    dims_t istrides_{static_cast<int>(in.info.strides[0]),
                     static_cast<int>(in.info.strides[1]),
                     static_cast<int>(in.info.strides[2]),
                     static_cast<int>(in.info.strides[3])};
    dims_t odims_{
        static_cast<int>(out.info.dims[0]), static_cast<int>(out.info.dims[1]),
        static_cast<int>(out.info.dims[2]), static_cast<int>(out.info.dims[3])};
    dims_t ostrides_{static_cast<int>(out.info.strides[0]),
                     static_cast<int>(out.info.strides[1]),
                     static_cast<int>(out.info.strides[2]),
                     static_cast<int>(out.info.strides[3])};
    int ondims_  = static_cast<int>(ondims);
    int elements = (ondims_ == 0) ? 0 : 1;
    for (int dim = 0; dim < ondims_; ++dim) elements *= odims_.dims[dim];
    if (elements > 0) {
        for (int c = 0; c < ondims_; ++c) {
            // Eliminate the columns with 1, so that we have more
            // appropriate dimensions in the local & global parameters
            if (odims_.dims[c] == 1) {
                for (int i = c; i < ondims_ - 1; ++i) {
                    odims_.dims[i]    = odims_.dims[i + 1];
                    ostrides_.dims[i] = ostrides_.dims[i + 1];
                    idims_.dims[i]    = idims_.dims[i + 1];
                    istrides_.dims[i] = istrides_.dims[i + 1];
                }
                --ondims_;
                odims_.dims[ondims_] = 1;
                idims_.dims[ondims_] = 1;
                --c;  // Redo this column, since it is eliminated now!!
            }
        }
        // This kernel is memory bound, so focus on caching (dim0+++)
        const int dim0 = (odims_.dims[0] <= 128) ? 128 : 256;
        const int dim1 = (dim0 == 256) || (odims_.dims[1] & 0x1) ? 1 : 2;
        const int dim2 = (dim0 * dim1 == 256) || (odims_.dims[2] & 0x1) ? 1 : 2;
        const cl::NDRange local(dim0, dim1, dim2);
        const cl::NDRange global(dim0 * divup(odims_.dims[0], dim0),
                                 dim1 * divup(odims_.dims[1], dim1),
                                 dim2 * divup(odims_.dims[2], dim2));

        if (std::is_same<inType, double>::value ||
            std::is_same<inType, cdouble>::value) {
            // Only scale in double precision when the input array is also
            // in double or cdouble, otherwise it is a waste of GPU time
            const vector<TemplateArg> targs = {
                TemplateTypename<inType>(),
                TemplateTypename<outType>(),
                TemplateArg(same_dims),
                TemplateTypename<double>(),
            };
            const vector<string> options = {
                DefineKeyValue(inType, dtype_traits<inType>::getName()),
                DefineKeyValue(outType, dtype_traits<outType>::getName()),
                string(" -D inType_" + string(dtype_traits<inType>::getName())),
                string(" -D outType_" +
                       string(dtype_traits<outType>::getName())),
                DefineKeyValue(SAME_DIMS, static_cast<int>(same_dims)),
                string(" -D factorType=double"),
                {getTypeBuildDefinition<inType, outType>()},
            };
            auto copy =
                common::getKernel("reshapeCopy", {copy_cl_src}, targs, options);
            copy(cl::EnqueueArgs(getQueue(), global, local), *out.data, odims_,
                 ostrides_, static_cast<uint>(out.info.offset), *in.data,
                 idims_, istrides_, static_cast<uint>(in.info.offset),
                 default_value, static_cast<double>(factor));
        } else {
            const vector<TemplateArg> targs = {
                TemplateTypename<inType>(),
                TemplateTypename<outType>(),
                TemplateArg(same_dims),
                TemplateTypename<float>(),
            };
            const vector<string> options = {
                DefineKeyValue(inType, dtype_traits<inType>::getName()),
                DefineKeyValue(outType, dtype_traits<outType>::getName()),
                string(" -D inType_" + string(dtype_traits<inType>::getName())),
                string(" -D outType_" +
                       string(dtype_traits<outType>::getName())),
                DefineKeyValue(SAME_DIMS, static_cast<int>(same_dims)),
                string(" -D factorType=float"),
                {getTypeBuildDefinition<inType, outType>()},
            };
            auto copy =
                common::getKernel("reshapeCopy", {copy_cl_src}, targs, options);
            copy(cl::EnqueueArgs(getQueue(), global, local), *out.data, odims_,
                 ostrides_, static_cast<uint>(out.info.offset), *in.data,
                 idims_, istrides_, static_cast<uint>(in.info.offset),
                 default_value, static_cast<float>(factor));
        }
        CL_DEBUG_FINISH(getQueue());
    }
}
}  // namespace kernel
}  // namespace opencl
