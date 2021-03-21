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
#include <debug_opencl.hpp>
#include <kernel/memcopy.hpp>
#include <kernel_headers/join.hpp>
#include <traits.hpp>

#include <string>
#include <vector>

namespace opencl {
namespace kernel {

template<typename T>
void join(Param out, const int jdim, const Param in1, const Param in2) {
    join(out, jdim, {in1, in2});
}

template<typename T>
void join(Param out, const int jdim, const std::vector<Param> &in) {
    switch (size) {
        case 0: return;
        case 1:
            return kernel::memcopy<T>(*out.data, dim4(4, out.info.strides),
                                      *in[0].data, dim4(4, in[0].info.dims),
                                      dim4(4, in[0].info.strides),
                                      in[0].info.offset, 4, out.info.offset);
    }

    const KParam nullInfo{};
    const cl::Buffer nullData;

    const std::vector<TemplateArg> targs = {
        TemplateTypename<T>(),
    };
    const std::vector<std::string> options = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
        {getTypeBuildDefinition<T>()},
    };

    auto joinN = common::getKernel("joinN", {join_cl_src}, targs, options);
    int ondims = out.info.dims[3] > 1   ? 4
                 : out.info.dims[2] > 1 ? 3
                 : out.info.dims[1] > 1 ? 2
                 : out.info.dims[0] > 0 ? 1
                                        : 0;
    for (int c = 0; c < ondims; ++c) {
        // Eliminate the columns with 1, so that we have more
        // appropriate dimensions in the local & global parameters
        if (out.info.dims[c] == 1) {
            for (int i = c; i < ondims - 1; ++i) {
                out.info.dims[i]    = out.info.dims[i + 1];
                out.info.strides[i] = out.info.strides[i + 1];
                for (auto &in_ : in) {
                    in_.info.dims[i]    = in_.info.dims[i + 1];
                    in_.info.strides[i] = in_.info.strides[i + 1];
                }
            }
            --ondims;
            if (c <= jdim) --jdim;
            out.info.dims[ondims] = 1;
            for (auto &in_ : in) in_.info.dims[ondims] = 1;
            --c;  // Redo this column, since it is eliminated now!!
        }
    }

    for (int start = 0, end = std::min(static_cast<int>(in.size()), start + 5);
         start < end;
         start += 5, end = std::min(static_cast<int>(in.size()), start + 5)) {
        // Handle max 5 inputs each cycle
        dim4 info(in[start].info.dims);
        for (int n = start + 1; n < end; ++n) {
            // Other dims are per definition equal
            info.dims[jdim] = std::max(info.dims[jdim], in[n].info.dims[jdim]);
        };
        // This kernel is memory bound, so focus on caching (dim0+++)
        const int dim0 = (info.dims[0] <= 128) ? 128 : 256;
        const int dim1 = (dim0 == 256) || (info.dims[1] & 0x1) ? 1 : 2;
        const int dim2 = (dim0 * dim1 == 256) || (info.dims[2] & 0x1) ? 1 : 2;
        const cl::NDRange local(dim0, dim1, dim2);
        const cl::NDRange global(dim0 * divup(info.dims[0], dim0),
                                 dim1 * divup(info.dims[1], dim1),
                                 dim2 * divup(info.dims[2], dim2));

        const int size = end - start;
        switch (size) {
            case 1: {
                const auto &i = in[start];
                return kernel::memcopy<T>(
                    *out.data, dim4(ondims, out.info.strides), *i.data,
                    dim4(ondims, i.info.dims), dim4(ondims, i.info.strides),
                    i.info.offset, ondims, out.info.offset);
            }
            case 2:
                joinN(cl::EnqueueArgs(getQueue(), global, local), *out.data,
                      out.info, jdim, size, *in[start].data, in[start].info,
                      *in[start + 1].data, in[start + 1].info, nullData,
                      nullInfo, nullData, nullInfo, nullData, nullInfo);
                CL_DEBUG_FINISH(getQueue());
                return;
            case 3:
                joinN(cl::EnqueueArgs(getQueue(), global, local), *out.data,
                      out.info, jdim, size, *in[start].data, in[start].info,
                      *in[start + 1].data, in[start + 1].info,
                      *in[start + 2].data, in[start + 2].info, nullData,
                      nullInfo, nullData, nullInfo);
                CL_DEBUG_FINISH(getQueue());
                return;
            case 4:
                joinN(cl::EnqueueArgs(getQueue(), global, local), *out.data,
                      out.info, jdim, size, *in[start].data, in[start].info,
                      *in[start + 1].data, in[start + 1].info,
                      *in[start + 2].data, in[start + 2].info,
                      *in[start + 3].data, in[start + 3].info, nullData,
                      nullInfo);
                CL_DEBUG_FINISH(getQueue());
                return;
            case 5:
                joinN(cl::EnqueueArgs(getQueue(), global, local), *out.data,
                      out.info, jdim, size, *in[start].data, in[start].info,
                      *in[start + 1].data, in[start + 1].info,
                      *in[start + 2].data, in[start + 2].info,
                      *in[start + 3].data, in[start + 3].info,
                      *in[start + 4].data, in[start + 4].info);
                CL_DEBUG_FINISH(getQueue());
                for (int d = start; d < end; ++d) {
                    out.info.offset +=
                        in[d].info.dims[jdim] * in[d].info.strides[jdim];
                }
        }
    }
}
}  // namespace kernel
}  // namespace opencl
