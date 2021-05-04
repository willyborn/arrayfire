/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <af/defines.h>
#include <array>
#include <cmath>

#define divup(a, b) (((a) + (b)-1) / (b))

unsigned nextpow2(unsigned x);

// isPrime & greatestPrimeFactor are tailored after
// itk::Math::{IsPrimt, GreatestPrimeFactor}
template<typename T>
inline bool isPrime(T n) {
    if (n <= 1) return false;

    const T last = (T)std::sqrt((double)n);
    for (T x = 2; x <= last; ++x) {
        if (n % x == 0) return false;
    }

    return true;
}

template<typename T>
inline T greatestPrimeFactor(T n) {
    T v = 2;

    while (v <= n) {
        if (n % v == 0 && isPrime(v))
            n /= v;
        else
            v += 1;
    }

    return v;
}

// For OPENCL, the dimensions of local are returned
// usage: cl::NDRange local = bestBlockSize<cl::NDRange>(dims)
// For CUDA, the dimensions of the warp are returned
// usage: dim3 block = bestBlockSize<dim3>(dims);
// The bestBlockSize is only best for independent element operations, as are:
//   copy, scaling, math on independent elements, ...
// So NO BLOCK OPERATIONS as matmul, etc!!
template<typename Tout, int NDIMS = 0, typename Tin>
Tout bestBlockSize(Tin dims[4]) {
    // considerations:
    //      1. Typically we have pre-caching, meaning that the next memory
    //          access are guessed and already loaded.  More threads for dim0 is
    //          typically faster (up to 64; WGs are scheduled in blocks of
    //          32/64)
    //      2. Occupation should ALWAYS be higher than 60% to have enough
    //      threads waiting on memory to become available (latency)
    //      3. Larger WGs mean less overhead, since the subblocks are scheduled
    //          together.  (limited impact)
    //      4. For very small element arrays, distribution among CU's is more
    //          important than occupation so that all cores are occupied.
    //          Cores typically work on 8 threads in parallel (SMID) in the same
    //          WG, so having one core doing all the work will not help
    //
    // occupation calculation:
    //      100% ex for 128: elements divisable by 128 (bin: 1000 0000)
    //                    => or no zero's in last 7 bits
    //                    => 128-1 = bin: 0111 1111
    //                    => elements AND (128-1) == 0
    //
    //      90% ex for 128:   (128(n-1)+1)/128n > 90%;
    //                     => n > (128-1) / 128(1-0.90)
    //                     => elements > 128(n-1)+1 = 9*(128-1)
    //
    //      Having this applied simultaneously on dims0 & dims1
    //                  >95% * >95% = >>90% total occup
    //                  >90% * >90% = >>81% total occup
    //                  >85% * >85% = >>72% total occup
    //                  >83% * >83% = >>69% total occup
    //                  >80% * >80% = >>64% total occup  ==> Seems fastest
    //                  >75% * >75% = >>56% total occup
    //                  >66% * >66% = >>43% total occup  ==> dramatic
    //                  >50% * >50% = >>25% total occup
    //
    //      >95% occup  -> elements > 19*(128-1)    > 19*(64-1) >19*(32-1)
    //      >90% occup  -> elements >  9*(128-1)    >  9*(64-1) > 9*(32-1)
    //      >85% occup  -> elements >  5.7*(128-1)
    //      >83% occup  -> elements >  5*(128-1)    >  5*(64-1) > 5*(32-1)
    //      >80% occup  -> elements >  4*(128-1)    >  4*(64-1) > 4*(32-1)
    //      >75% occup  -> elements >  3*(128-1)    >  3*(64-1) > 3*(32-1)
    //      >66% occup  -> elements >  2*(128-1)    >  2*(64-1) > 2*(32-1)
    //      >50% occup  -> elements >  1*(128-1)    >  1*(64-1) > 1*(32-1)

    // nrdims count the number of dimensions contributing to occupation
    // while NDIMS corresponds with total number of dimensions

    /* JIT3N-v3  --> OPENCL GOOD / CUDA??
    constexpr int OCC  = 3;
    const Tin threads0 = (dims[0] == 1)                ? 1    // hope on dims1
                         : (dims[0] <= 32)             ? 32   // min for caching
                         : !(dims[0] & (128 - 1))      ? 128  // 100% occupancy
                         : !(dims[0] & (64 - 1))       ? 64   // 100% occupancy
                         : !(dims[0] & (32 - 1))       ? 32   // 100% occupancy
                         : (dims[0] > OCC * (256 - 1)) ? 256  // >NN% occupancy
                         : (dims[0] > OCC * (128 - 1))
                             ? 128  // >NN% occupancy
                             : 64;  // best alternative
    */
    /* JTI3N-v4  --> OPENCL >v3 / CUDA <v3
    constexpr int OCC  = 3;
    const Tin threads0 = (dims[0] <= 16)          ? dims[0]   // hope on dims1
                         : (dims[0] <= 32)        ? 32        // min for caching
                         : !(dims[0] & (128 - 1)) ? 128       // 100% occupancy
                         : !(dims[0] & (64 - 1))  ? 64        // 100% occupancy
                         : !(dims[0] & (32 - 1))  ? 32        // 100% occupancy
                         : (dims[0] > OCC * (256 - 1)) ? 256  // >NN% occupancy
                         : (dims[0] > OCC * (128 - 1))
                             ? 128  // >NN% occupancy
                             : 64;  // best alternative
    */
    /* JIT3N-v5  --> OPENCL <v4 (>v3) / CUDA >v3 (>v4)
    constexpr int OCC  = 3;
    const Tin threads0 = (dims[0] <= 32)               ? 32   // min for caching
                         : !(dims[0] & (128 - 1))      ? 128  // 100% occupancy
                         : !(dims[0] & (64 - 1))       ? 64   // 100% occupancy
                         : !(dims[0] & (32 - 1))       ? 32   // 100% occupancy
                         : (dims[0] > OCC * (256 - 1)) ? 256  // >NN% occupancy
                         : (dims[0] > OCC * (128 - 1))
                             ? 128  // >NN% occupancy
                             : 64;  // best alternative
    */

    // TEMP if (NDIMS == 1) return Tout((int)threads0);

    /* v3, v4, v5
    const Tin threads1 =
        (threads0 <= 256 / 128) &&
                (!(dims[1] & (128 - 1)) || (dims[1] > OCC * (128 - 1)))
            ? 128
        : (threads0 <= 256 / 64) &&
                (!(dims[1] & (64 - 1)) || (dims[1] > OCC * (64 - 1)))
            ? 64
        : (threads0 <= 256 / 32) &&
                (!(dims[1] & (32 - 1)) || (dims[1] > OCC * (32 - 1)))
            ? 32
        : (threads0 <= 256 / 16) &&
                (!(dims[1] & (16 - 1)) || (dims[1] > OCC * (16 - 1)))
            ? 16
        : (threads0 <= 256 / 8) &&
                (!(dims[1] & (8 - 1)) || (dims[1] > OCC * (8 - 1)))
            ? 8
        : (threads0 <= 256 / 4) &&
                (!(dims[1] & (4 - 1)) || (dims[1] > OCC * (4 - 1)))
            ? 4
        : (threads0 <= 256 / 2) &&
                (!(dims[1] & (2 - 1)) || (dims[1] > OCC * (2 - 1)))
            ? 2
            : 1;
            */

    /* JIT3N-v6  --> OPENCL <v4 (<v3) / CUDA >v5 (>v3)
    constexpr int OCC  = 3;
    const Tin threads0 = (dims[0] <= 16)          ? dims[0]   // hope on dims1
                         : (dims[0] <= 32)        ? 32        // min for caching
                         : !(dims[0] & (128 - 1)) ? 128       // 100% occupancy
                         : !(dims[0] & (64 - 1))  ? 64        // 100% occupancy
                         : !(dims[0] & (32 - 1))  ? 32        // 100% occupancy
                         : (dims[0] > OCC * (256 - 1)) ? 256  // >NN% occupancy
                         : (dims[0] > OCC * (128 - 1))
                             ? 128  // >NN% occupancy
                             : 64;  // best alternative
    if (NDIMS == 1) return Tout((int)threads0);
    const Tin threads1 =
        (threads0 <= 256 / 128) &&
                (!(dims[1] & (128 - 1)) || (dims[1] > OCC * (128 - 1)))
            ? 128
        : (threads0 <= 256 / 64) &&
                (!(dims[1] & (64 - 1)) || (dims[1] > OCC * (64 - 1)))
            ? 64
        : (threads0 <= 256 / 32) &&
                (!(dims[1] & (32 - 1)) || (dims[1] > OCC * (32 - 1)))
            ? 32
        : (threads0 <= 256 / 16)
            ? (!(dims[1] & (16 - 1)) || (dims[1] > OCC * (16 - 1)))
                  ? 16
                  : std::min(128 / threads0, dims[1])
        : (threads0 <= 256 / 8) &&
                (!(dims[1] & (8 - 1)) || (dims[1] > OCC * (8 - 1)))
            ? 8
        : (threads0 <= 256 / 4) &&
                (!(dims[1] & (4 - 1)) || (dims[1] > OCC * (4 - 1)))
            ? 4
        : (threads0 <= 256 / 2) &&
                (!(dims[1] & (2 - 1)) || (dims[1] > OCC * (2 - 1)))
            ? 2
            : 1;
    */

    /* JIT3N-v7  --> OPENCL <v4 (<v3);  CUDA =v6 (>v5)
    constexpr int OCC  = 3;
    const Tin threads0 = (dims[0] <= 32)               ? 32   // min for caching
                         : !(dims[0] & (128 - 1))      ? 128  // 100% occupancy
                         : !(dims[0] & (64 - 1))       ? 64   // 100% occupancy
                         : !(dims[0] & (32 - 1))       ? 32   // 100% occupancy
                         : (dims[0] > OCC * (256 - 1)) ? 256  // >NN% occupancy
                         : (dims[0] > OCC * (128 - 1))
                             ? 128  // >NN% occupancy
                             : 64;  // best alternative
    if (NDIMS == 1) return Tout((int)threads0);
    const Tin threads1 =
        (threads0 <= 256 / 128) &&
                (!(dims[1] & (128 - 1)) || (dims[1] > OCC * (128 - 1)))
            ? 128
        : (threads0 <= 256 / 64) &&
                (!(dims[1] & (64 - 1)) || (dims[1] > OCC * (64 - 1)))
            ? 64
        : (threads0 <= 256 / 32) &&
                (!(dims[1] & (32 - 1)) || (dims[1] > OCC * (32 - 1)))
            ? 32
        : (threads0 <= 256 / 16) &&
                (!(dims[1] & (16 - 1)) || (dims[1] > OCC * (16 - 1)))
            ? 16
        : (threads0 <= 256 / 8)
            ? (!(dims[1] & (8 - 1)) || (dims[1] > OCC * (8 - 1)))
                  ? 8
                  : std::min(128 / threads0, dims[1])
        : (threads0 <= 256 / 4) &&
                (!(dims[1] & (4 - 1)) || (dims[1] > OCC * (4 - 1)))
            ? 4
        : (threads0 <= 256 / 2) &&
                (!(dims[1] & (2 - 1)) || (dims[1] > OCC * (2 - 1)))
            ? 2
            : 1;
    */

    /* JIT3N-v8  --> OPENCL <v4 (>v3) / CUDA <v7 <v6 (>v5)
    constexpr int OCC  = 3;
    const Tin threads0 = (dims[0] <= 32)               ? 32   // min for caching
                         : !(dims[0] & (128 - 1))      ? 128  // 100% occupancy
                         : !(dims[0] & (64 - 1))       ? 64   // 100% occupancy
                         : !(dims[0] & (32 - 1))       ? 32   // 100% occupancy
                         : (dims[0] > OCC * (256 - 1)) ? 256  // >NN% occupancy
                         : (dims[0] > OCC * (128 - 1))
                             ? 128  // >NN% occupancy
                             : 64;  // best alternative
    if (NDIMS == 1) return Tout((int)threads0);
    const Tin threads1 =
        (threads0 <= 256 / 128) &&
                (!(dims[1] & (128 - 1)) || (dims[1] > OCC * (128 - 1)))
            ? 128
        : (threads0 <= 256 / 64) &&
                (!(dims[1] & (64 - 1)) || (dims[1] > OCC * (64 - 1)))
            ? 64
        : (threads0 <= 256 / 32) &&
                (!(dims[1] & (32 - 1)) || (dims[1] > OCC * (32 - 1)))
            ? 32
        : (threads0 <= 256 / 16) &&
                (!(dims[1] & (16 - 1)) || (dims[1] > OCC * (16 - 1)))
            ? 16
        : (threads0 <= 256 / 8) &&
                (!(dims[1] & (8 - 1)) || (dims[1] > OCC * (8 - 1)))
            ? 8
        : (threads0 <= 256 / 4)
            ? (!(dims[1] & (4 - 1)) || (dims[1] > OCC * (4 - 1)))
                  ? 4
                  : std::min(128 / threads0, dims[1])
        : (threads0 <= 256 / 2) &&
                (!(dims[1] & (2 - 1)) || (dims[1] > OCC * (2 - 1)))
            ? 2
            : 1;
    */

    /* JIT3N-v9  --> OPENCL <v4 (>v3) / CUDA >v7 >v6 (>v5)
    constexpr int OCC  = 3;
    const Tin threads0 = (dims[0] <= 32)               ? 32   // min for caching
                         : !(dims[0] & (128 - 1))      ? 128  // 100% occupancy
                         : !(dims[0] & (64 - 1))       ? 64   // 100% occupancy
                         : !(dims[0] & (32 - 1))       ? 32   // 100% occupancy
                         : (dims[0] > OCC * (256 - 1)) ? 256  // >NN% occupancy
                         : (dims[0] > OCC * (128 - 1)) ? 128  // >NN% occupancy
                         : (dims[0] > OCC * (64 - 1))  ? 64
                                                      : 32;  // best alternative
    if (NDIMS == 1) return Tout((int)threads0);
    const Tin threads1 =
        (threads0 <= 256 / 128) &&
                (!(dims[1] & (128 - 1)) || (dims[1] > OCC * (128 - 1)))
            ? 128
        : (threads0 <= 256 / 64) &&
                (!(dims[1] & (64 - 1)) || (dims[1] > OCC * (64 - 1)))
            ? 64
        : (threads0 <= 256 / 32) &&
                (!(dims[1] & (32 - 1)) || (dims[1] > OCC * (32 - 1)))
            ? 32
        : (threads0 <= 256 / 16) &&
                (!(dims[1] & (16 - 1)) || (dims[1] > OCC * (16 - 1)))
            ? 16
        : (threads0 <= 256 / 8) &&
                (!(dims[1] & (8 - 1)) || (dims[1] > OCC * (8 - 1)))
            ? 8
        : (threads0 <= 256 / 4)
            ? (!(dims[1] & (4 - 1)) || (dims[1] > OCC * (4 - 1)))
                  ? 4
                  : std::min(128 / threads0, dims[1])    // <-- ?? BAD
        : (threads0 <= 256 / 2) &&
                (!(dims[1] & (2 - 1)) || (dims[1] > OCC * (2 - 1)))
            ? 2
            : 1;
    */

    /* JIT3N-v10  --> OPENCL <v4 (=v3) / CUDA <v7 <v6 (<v5)
    constexpr int OCC  = 3;
    const Tin threads0 = (dims[0] <= 32)               ? 32   // min for caching
                         : !(dims[0] & (256 - 1))      ? 256  // 100% occupancy
                         : !(dims[0] & (128 - 1))      ? 128  // 100% occupancy
                         : !(dims[0] & (64 - 1))       ? 64   // 100% occupancy
                         : !(dims[0] & (32 - 1))       ? 32   // 100% occupancy
                         : (dims[0] > OCC * (256 - 1)) ? 256  // >NN% occupancy
                         : (dims[0] > OCC * (128 - 1)) ? 128  // >NN% occupancy
                         : (dims[0] > OCC * (64 - 1))  ? 64
                                                      : 32;  // best alternative
    if (NDIMS == 1) return Tout((int)threads0);
    const Tin threads1 =
        (threads0 <= 256 / 128) &&
                (!(dims[1] & (128 - 1)) || (dims[1] > OCC * (128 - 1)))
            ? 128
        : (threads0 <= 256 / 64) &&
                (!(dims[1] & (64 - 1)) || (dims[1] > OCC * (64 - 1)))
            ? 64
        : (threads0 <= 256 / 32) &&
                (!(dims[1] & (32 - 1)) || (dims[1] > OCC * (32 - 1)))
            ? 32
        : (threads0 <= 256 / 16)
            ? (!(dims[1] & (16 - 1)) || (dims[1] > OCC * (16 - 1)))
                  ? 16
                  : std::min(128 / dims[0], dims[1])
        : (threads0 <= 256 / 8) &&
                (!(dims[1] & (8 - 1)) || (dims[1] > OCC * (8 - 1)))
            ? 8
        : (threads0 <= 256 / 4) &&
                (!(dims[1] & (4 - 1)) || (dims[1] > OCC * (4 - 1)))
            ? 4
        : (threads0 <= 256 / 2) &&
                (!(dims[1] & (2 - 1)) || (dims[1] > OCC * (2 - 1)))
            ? 2
            : 1;
    */

    /* JIT3N-v11  --> OPENCL <<v4 (<v3) / CUDA <v7 <v6 (<v5)
    constexpr int OCC  = 3;
    const Tin threads0 = (dims[0] == 1)           ? 1
                         : !(dims[0] & (256 - 1)) ? 256
                         : !(dims[0] & (128 - 1)) ? 128  // 100% occupancy
                         : !(dims[0] & (64 - 1))  ? 64   // 100% occupancy
                         : !(dims[0] & (32 - 1))  ? 32   // 100% occupancy
                         : (dims[0] > OCC * (256 - 1))
                             ? 256   // >NN% occupancy
                             : 128;  // best alternative
    if (NDIMS == 1) return Tout((int)threads0);
    const Tin threads1 =
        (threads0 <= 256 / 128) &&
                (!(dims[1] & (128 - 1)) || (dims[1] > OCC * (128 - 1)))
            ? 128
        : (threads0 <= 256 / 64) &&
                (!(dims[1] & (64 - 1)) || (dims[1] > OCC * (64 - 1)))
            ? 64
        : (threads0 <= 256 / 32) &&
                (!(dims[1] & (32 - 1)) || (dims[1] > OCC * (32 - 1)))
            ? 32
        : (threads0 <= 256 / 16) &&
                (!(dims[1] & (16 - 1)) || (dims[1] > OCC * (16 - 1)))
            ? 16
        : (threads0 <= 256 / 8) &&
                (!(dims[1] & (8 - 1)) || (dims[1] > OCC * (8 - 1)))
            ? 8
        : (threads0 <= 256 / 4) &&
                (!(dims[1] & (4 - 1)) || (dims[1] > OCC * (4 - 1)))
            ? 4
        : (threads0 <= 256 / 2) &&
                (!(dims[1] & (2 - 1)) || (dims[1] > OCC * (2 - 1)))
            ? 2
            : 1;
    */
    /* JIT3N-v12  --> OPENCL  v4 (v3) / CUDA v7 v6 (v5)
    constexpr int OCC = 3;
    const unsigned threads0 =
        (dims[0] <= 16)   ? dims[0]  // hope on dims1
        : (dims[0] <= 32) ? 32       // min for caching
                                     // JIT3N-v4Rep3
        : !(dims[0] & (256 - 1))      ? 256
        : !(dims[0] & (128 - 1))      ? 128  // 100% occupancy
        : !(dims[0] & (64 - 1))       ? 64   // 100% occupancy
        : !(dims[0] & (32 - 1))       ? 32   // 100% occupancy
        : (dims[0] > OCC * (256 - 1)) ? 256  // >NN% occupancy
        : (dims[0] > OCC * (128 - 1)) ? 128  // >NN% occupancy
                                      : 64;  // best alternative
    if (NDIMS == 1) return Tout(threads0);
    const unsigned threads1 =
        // JIT3N-v4Rep3  != NVIDIA
        //(threads0 <= 256 / 256) &&
        //        (!(dims[1] & (256 - 1)) || (dims[1] > OCC * (256 - 1)))
        //    ? 256
        //:
        // JIT3N-v4Rep3
        (threads0 <= 256 / 128) &&
                (!(dims[1] & (128 - 1)) || (dims[1] > OCC * (128 - 1)))
            ? 128
        : (threads0 <= 256 / 64) &&
                (!(dims[1] & (64 - 1)) || (dims[1] > OCC * (64 - 1)))
            ? 64
#ifdef AF_CUDA
        : (threads0 <= 256 / 32)
            ? (!(dims[1] & (32 - 1)) || (dims[1] > OCC * (32 - 1)))
                  ? 32
                  : std::min(128 / threads0, (unsigned)dims[1])
        : (threads0 <= 256 / 16) &&
                (!(dims[1] & (16 - 1)) || (dims[1] > OCC * (16 - 1)))
            ? 16
#else

        : (threads0 <= 256 / 32) &&
                (!(dims[1] & (32 - 1)) || (dims[1] > OCC * (32 - 1)))
            ? 32
        : (threads0 <= 256 / 16)
            ? (!(dims[1] & (16 - 1)) || (dims[1] > OCC * (16 - 1)))
                  ? 16
                  : std::min(128 / threads0, (unsigned)dims[1])
#endif
        : (threads0 <= 256 / 8) &&
                (!(dims[1] & (8 - 1)) || (dims[1] > OCC * (8 - 1)))
            ? 8
        : (threads0 <= 256 / 4) &&
                (!(dims[1] & (4 - 1)) || (dims[1] > OCC * (4 - 1)))
            ? 4
        : (threads0 <= 256 / 2) &&
                (!(dims[1] & (2 - 1)) || (dims[1] > OCC * (2 - 1)))
            ? 2
            : 1;
    */

    /*
    // JIT3N-v4Rep64, JIT3N-v4Rep128, JIT3N-v4Rep256
    constexpr int OCC             = 3;
    constexpr unsigned maxThreads = 32;

    const unsigned threads0 =
        (dims[0] <= 16)                                      ? dims[0]
        : (dims[0] <= 32)                                    ? 32
        : (maxThreads >= 256) && !(dims[0] & (256 - 1))      ? 256
        : (maxThreads >= 128) && !(dims[0] & (128 - 1))      ? 128
        : (maxThreads >= 64) && !(dims[0] & (64 - 1))        ? 64
        : !(dims[0] & (32 - 1))                              ? 32
        : (maxThreads >= 256) && (dims[0] > OCC * (256 - 1)) ? 256
        : (maxThreads >= 128) && (dims[0] > OCC * (128 - 1)) ? 128
                                                             : 64;
    if (NDIMS == 1) return Tout(threads0);
    const unsigned threads1 =
        (threads0 <= maxThreads / 256) &&
                (!(dims[1] & (256 - 1)) || (dims[1] > OCC * (256 - 1)))
            ? 256
        : (threads0 <= maxThreads / 128) &&
                (!(dims[1] & (128 - 1)) || (dims[1] > OCC * (128 - 1)))
            ? 128
        : (threads0 <= maxThreads / 64) &&
                (!(dims[1] & (64 - 1)) || (dims[1] > OCC * (64 - 1)))
            ? 64
#ifdef AF_CUDA
        : (threads0 <= maxThreads / 32)
            ? (!(dims[1] & (32 - 1)) || (dims[1] > OCC * (32 - 1)))
                  ? 32
                  : std::min(128 / threads0, (unsigned)dims[1])
        : (threads0 <= maxThreads / 16) &&
                (!(dims[1] & (16 - 1)) || (dims[1] > OCC * (16 - 1)))
            ? 16
#else

        : (threads0 <= maxThreads / 32) &&
                (!(dims[1] & (32 - 1)) || (dims[1] > OCC * (32 - 1)))
            ? 32
        : (threads0 <= maxThreads / 16)
            ? (!(dims[1] & (16 - 1)) || (dims[1] > OCC * (16 - 1)))
                  ? 16
                  : std::min(128 / threads0, (unsigned)dims[1])
#endif
        : (threads0 <= maxThreads / 8) &&
                (!(dims[1] & (8 - 1)) || (dims[1] > OCC * (8 - 1)))
            ? 8
        : (threads0 <= maxThreads / 4) &&
                (!(dims[1] & (4 - 1)) || (dims[1] > OCC * (4 - 1)))
            ? 4
        : (threads0 <= maxThreads / 2) &&
                (!(dims[1] & (2 - 1)) || (dims[1] > OCC * (2 - 1)))
            ? 2
            : 1;
    */
    /*
    // JIT3N-v4Rep32.32
    constexpr int OCC             = 3;
    constexpr unsigned maxThreads = 32;

    const unsigned threads0 =
        (dims[0] <= 16)                                      ? dims[0]
        : (dims[0] <= 32)                                    ? 32
        : (maxThreads >= 256) && !(dims[0] & (256 - 1))      ? 256
        : (maxThreads >= 128) && !(dims[0] & (128 - 1))      ? 128
        : (maxThreads >= 64) && !(dims[0] & (64 - 1))        ? 64
        : !(dims[0] & (32 - 1))                              ? 32
        : (maxThreads >= 256) && (dims[0] > OCC * (256 - 1)) ? 256
        : (maxThreads >= 128) && (dims[0] > OCC * (128 - 1)) ? 128
                                                             : 32;
    if (NDIMS == 1) return Tout(threads0);
    const unsigned threads1 =
        (threads0 <= maxThreads / 256) &&
                (!(dims[1] & (256 - 1)) || (dims[1] > OCC * (256 - 1)))
            ? 256
        : (threads0 <= maxThreads / 128) &&
                (!(dims[1] & (128 - 1)) || (dims[1] > OCC * (128 - 1)))
            ? 128
        : (threads0 <= maxThreads / 64) &&
                (!(dims[1] & (64 - 1)) || (dims[1] > OCC * (64 - 1)))
            ? 64
#ifdef AF_CUDA
        : (threads0 <= maxThreads / 32)
            ? (!(dims[1] & (32 - 1)) || (dims[1] > OCC * (32 - 1)))
                  ? 32
                  : std::min(128 / threads0, (unsigned)dims[1])
        : (threads0 <= maxThreads / 16) &&
                (!(dims[1] & (16 - 1)) || (dims[1] > OCC * (16 - 1)))
            ? 16
#else

        : (threads0 <= maxThreads / 32) &&
                (!(dims[1] & (32 - 1)) || (dims[1] > OCC * (32 - 1)))
            ? 32
        : (threads0 <= maxThreads / 16)
            ? (!(dims[1] & (16 - 1)) || (dims[1] > OCC * (16 - 1)))
                  ? 16
                  : std::min(128 / threads0, (unsigned)dims[1])
#endif
        : (threads0 <= maxThreads / 8) &&
                (!(dims[1] & (8 - 1)) || (dims[1] > OCC * (8 - 1)))
            ? 8
        : (threads0 <= maxThreads / 4) &&
                (!(dims[1] & (4 - 1)) || (dims[1] > OCC * (4 - 1)))
            ? 4
        : (threads0 <= maxThreads / 2) &&
                (!(dims[1] & (2 - 1)) || (dims[1] > OCC * (2 - 1)))
            ? 2
            : 1;
    */

    /*
        // JIT3N-v4Rep64-32
        constexpr int OCC = 3;
    #ifdef AF_CUDA
        unsigned maxThreads = 64;
    #else
        unsigned maxThreads = 128;
    #endif

        const unsigned threads0 =
            (dims[0] <= 16)                                      ? dims[0]
            : (dims[0] <= 32)                                    ? 32
            : (maxThreads >= 128) && !(dims[0] & (128 - 1))      ? 128
            : (maxThreads >= 64) && !(dims[0] & (64 - 1))        ? 64
            : (maxThreads >= 32) && !(dims[0] & (32 - 1))        ? 32
            : (maxThreads >= 128) && (dims[0] > OCC * (128 - 1)) ? 128
                                                                 : 64;
        if (NDIMS == 1) return Tout(threads0);

        maxThreads /= 2;
        const unsigned threads1 =
            (threads0 <= maxThreads / 64) &&
                    (!(dims[1] & (64 - 1)) || (dims[1] > OCC * (64 - 1)))
                ? 64
    #ifdef AF_CUDA
            : (threads0 <= maxThreads / 32)
                ? (!(dims[1] & (32 - 1)) || (dims[1] > OCC * (32 - 1)))
                      ? 32
                      : std::min(maxThreads / threads0, (unsigned)dims[1])
            : (threads0 <= maxThreads / 16) &&
                    (!(dims[1] & (16 - 1)) || (dims[1] > OCC * (16 - 1)))
                ? 16
    #else

            : (threads0 <= maxThreads / 32) &&
                    (!(dims[1] & (32 - 1)) || (dims[1] > OCC * (32 - 1)))
                ? 32
            : (threads0 <= maxThreads / 16)
                ? (!(dims[1] & (16 - 1)) || (dims[1] > OCC * (16 - 1)))
                      ? 16
                      : std::min(maxThreads / threads0, (unsigned)dims[1])
    #endif
            : (threads0 <= maxThreads / 8) &&
                    (!(dims[1] & (8 - 1)) || (dims[1] > OCC * (8 - 1)))
                ? 8
            : (threads0 <= maxThreads / 4) &&
                    (!(dims[1] & (4 - 1)) || (dims[1] > OCC * (4 - 1)))
                ? 4
            : (threads0 <= maxThreads / 2) &&
                    (!(dims[1] & (2 - 1)) || (dims[1] > OCC * (2 - 1)))
                ? 2
                : 1;
        */
    /**/
    // JIT3N64
    constexpr int OCC = 3;
#ifdef AF_CUDA
    constexpr unsigned maxThreads = 64;
#else
    constexpr unsigned maxThreads = 128;
#endif

    const unsigned threads0 =
        (dims[0] <= 16)                                      ? dims[0]
        : (dims[0] <= 32)                                    ? 32
        : (maxThreads >= 128) && !(dims[0] & (128 - 1))      ? 128
        : (maxThreads >= 64) && !(dims[0] & (64 - 1))        ? 64
        : (maxThreads >= 32) && !(dims[0] & (32 - 1))        ? 32
        : (maxThreads >= 128) && (dims[0] > OCC * (128 - 1)) ? 128
        : (maxThreads >= 64) && (dims[0] > OCC * (64 - 1))   ? 64
                                                             : maxThreads / 2;
    if (NDIMS == 1) return Tout(threads0);

    const unsigned threads1 =
        (threads0 <= maxThreads / 128) &&
                (!(dims[1] & (128 - 1)) || (dims[1] > OCC * (128 - 1)))
            ? 128
        : (threads0 <= maxThreads / 64) &&
                (!(dims[1] & (64 - 1)) || (dims[1] > OCC * (64 - 1)))
            ? 64
#ifdef AF_CUDA
        : (threads0 <= maxThreads / 32)
            ? (!(dims[1] & (32 - 1)) || (dims[1] > OCC * (32 - 1)))
                  ? 32
                  : std::min(maxThreads / threads0, (unsigned)dims[1])
        : (threads0 <= maxThreads / 16) &&
                (!(dims[1] & (16 - 1)) || (dims[1] > OCC * (16 - 1)))
            ? 16
#else
        : (threads0 <= maxThreads / 32) &&
                (!(dims[1] & (32 - 1)) || (dims[1] > OCC * (32 - 1)))
            ? 32
        : (threads0 <= maxThreads / 16)
            ? (!(dims[1] & (16 - 1)) || (dims[1] > OCC * (16 - 1)))
                  ? 16
                  : std::min(maxThreads / threads0, (unsigned)dims[1])
#endif
        : (threads0 <= maxThreads / 8) &&
                (!(dims[1] & (8 - 1)) || (dims[1] > OCC * (8 - 1)))
            ? 8
        : (threads0 <= maxThreads / 4) &&
                (!(dims[1] & (4 - 1)) || (dims[1] > OCC * (4 - 1)))
            ? 4
        : (threads0 <= maxThreads / 2) &&
                (!(dims[1] & (2 - 1)) || (dims[1] > OCC * (2 - 1)))
            ? 2
            : 1;
    /**/
    /**/
    if (NDIMS == 2) return Tout(threads0, threads1);

    const unsigned threads01 = threads0 * threads1;
    // max 64 for CUDA
    const unsigned threads2 =
        (threads01 <= maxThreads / 64) && !(dims[2] & (64 - 1))   ? 64
        : (threads01 <= maxThreads / 32) && !(dims[2] & (32 - 1)) ? 32
        : (threads01 <= maxThreads / 16) && !(dims[2] & (16 - 1)) ? 16
        : (threads01 <= maxThreads / 8) && !(dims[2] & (8 - 1))   ? 8
        : (threads01 <= maxThreads / 4) && !(dims[2] & (4 - 1))   ? 4
        : (threads01 <= maxThreads / 2) && !(dims[2] & (2 - 1))   ? 2
                                                                  : 1;
    return Tout(threads0, threads1, threads2);
    /**/

    /*
    // JIT-32-64
    // Idea
    // INPUT is WAITING on multiple cacheline before to proceed
    // - reading from mem is SNCHRONISE
    // OUTPUT is writing in cache and IMMEDIATLY returning
    // - writing to mem is in aSYNCHRONISE
    // CUDA is randomly scheduling warps (32 elements)
    // - multiple warps form 1 block and share local memory (overhead)
    //
    // 1. Linear input as much as possible, to optimize precaching
    //  - group all linear dimensions  --> dim0 is largest linear block
    // - always schedule per warp
    // 2. Output is using internal cache
    // 3. Occupation rate is neglected
    //
#ifdef AF_CUDA
    constexpr unsigned minThreads = 32;
#else
    constexpr unsigned minThreads = 64;
#endif

    const unsigned threads0 = (dims[0] <= minThreads) ? dims[0] : minThreads;
    if (NDIMS == 1) return Tout(threads0);

    const unsigned threads1 =
        std::min(minThreads / threads0, (unsigned)dims[1]);
    if (NDIMS == 2) return Tout(threads0, threads1);

    const unsigned threads2 =
        std::min(minThreads / (threads0 * threads1), (unsigned)dims[2]);
    return Tout(threads0, threads1, threads2);
    */

    /*
    // JIT-32-1
    #ifdef AF_CUDA
        constexpr unsigned minThreads = 32;
    #else
        constexpr unsigned minThreads = 64;
    #endif

        const unsigned threads0 = (dims[0] <= minThreads) ? dims[0] :
    minThreads; if (NDIMS == 1) return Tout(threads0);

        const unsigned threads1 =
            (threads0 == 1) ? std::min(minThreads / threads0, (unsigned)dims[1])
                            : 1;
        return Tout(threads0, threads1);
        */
}