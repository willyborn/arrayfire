/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#pragma once
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wparentheses"
#endif
#include <half.hpp>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
#include <af/array.h>
#include <af/defines.h>
#include <af/dim4.hpp>
#include <af/traits.hpp>

#include <gtest/gtest.h>

#include <cfloat>
#include <string>
#include <vector>

#if defined(USE_MTX)
#include <mmio.h>
#include <cstdlib>
#endif

/// GTest deprecated the INSTANTIATED_TEST_CASE_P macro in favor of the
/// INSTANTIATE_TEST_SUITE_P macro which has the same syntax but the older
/// versions of gtest do not support this new macro adds the
/// INSTANTIATE_TEST_SUITE_P macro and maps it to the old macro
#ifndef INSTANTIATE_TEST_SUITE_P
#define INSTANTIATE_TEST_SUITE_P INSTANTIATE_TEST_CASE_P
#endif
#ifndef TYPED_TEST_SUITE
#define TYPED_TEST_SUITE TYPED_TEST_CASE
#endif

bool operator==(const af_half &lhs, const af_half &rhs);

std::ostream &operator<<(std::ostream &os, const af_half &val);

#define UNUSED(expr) \
    do { (void)(expr); } while (0)

namespace aft {
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
typedef intl intl;
typedef uintl uintl;
#ifdef __GNUC__
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif
}  // namespace aft

using aft::intl;
using aft::uintl;

std::ostream &operator<<(std::ostream &os, af::Backend bk);

std::ostream &operator<<(std::ostream &os, af_err e);

std::ostream &operator<<(std::ostream &os, af::dtype type);

namespace af {
template<>
struct dtype_traits<half_float::half> {
    enum { af_type = f16, ctype = f16 };
    typedef half_float::half base_type;
    static const char *getName() { return "half"; }
};

}  // namespace af

typedef unsigned char uchar;
typedef unsigned int uint;
typedef unsigned short ushort;

std::string getBackendName();
std::string getTestName();

std::string readNextNonEmptyLine(std::ifstream &file);

namespace half_float {
std::ostream &operator<<(std::ostream &os, half_float::half val);
}  // namespace half_float

template<typename To, typename Ti>
To convert(Ti in) {
    return static_cast<To>(in);
}

#ifndef EXTERN_TEMPLATE
extern template float convert(af::half in);
extern template af_half convert(int in);
#endif

template<typename inType, typename outType, typename FileElementType>
void readTests(const std::string &FileName, std::vector<af::dim4> &inputDims,
               std::vector<std::vector<inType>> &testInputs,
               std::vector<std::vector<outType>> &testOutputs);

template<typename inType, typename outType>
void readTestsFromFile(const std::string &FileName,
                       std::vector<af::dim4> &inputDims,
                       std::vector<std::vector<inType>> &testInputs,
                       std::vector<std::vector<outType>> &testOutputs);

void readImageTests(const std::string &pFileName,
                    std::vector<af::dim4> &pInputDims,
                    std::vector<std::string> &pTestInputs,
                    std::vector<dim_t> &pTestOutSizes,
                    std::vector<std::string> &pTestOutputs);

template<typename outType>
void readImageTests(const std::string &pFileName,
                    std::vector<af::dim4> &pInputDims,
                    std::vector<std::string> &pTestInputs,
                    std::vector<std::vector<outType>> &pTestOutputs);

template<typename descType>
void readImageFeaturesDescriptors(
    const std::string &pFileName, std::vector<af::dim4> &pInputDims,
    std::vector<std::string> &pTestInputs,
    std::vector<std::vector<float>> &pTestFeats,
    std::vector<std::vector<descType>> &pTestDescs);

/**
 * Below is not a pair wise comparition method, rather
 * it computes the accumulated error of the computed
 * output and gold output.
 *
 * The cut off is decided based on root mean square
 * deviation from cpu result
 *
 * For images, the maximum possible error will happen if all
 * the observed values are zeros and all the predicted values
 * are 255's. In such case, the value of NRMSD will be 1.0
 * Similarly, we can deduce that 0.0 will be the minimum
 * value of NRMSD. Hence, the range of RMSD is [0,255] for image inputs.
 */
template<typename T>
bool compareArraysRMSD(dim_t data_size, T *gold, T *data, double tolerance);

template<typename T>
double computeArraysRMSD(dim_t data_size, T *gold, T *data);

template<typename T, typename Other>
struct is_same_type {
    static const bool value = false;
};

template<typename T>
struct is_same_type<T, T> {
    static const bool value = true;
};

template<bool, typename T, typename O>
struct cond_type;

template<typename T, typename Other>
struct cond_type<true, T, Other> {
    typedef T type;
};

template<typename T, typename Other>
struct cond_type<false, T, Other> {
    typedef Other type;
};

template<bool B, class T = void>
struct enable_if {};

template<class T>
struct enable_if<true, T> {
    typedef T type;
};

template<typename T>
inline double real(T val) {
    return (double)val;
}
template<>
inline double real<af::cdouble>(af::cdouble val) {
    return real(val);
}
template<>
inline double real<af::cfloat>(af::cfloat val) {
    return real(val);
}

template<typename T>
inline double imag(T val) {
    return (double)val;
}
template<>
inline double imag<af::cdouble>(af::cdouble val) {
    return imag(val);
}
template<>
inline double imag<af::cfloat>(af::cfloat val) {
    return imag(val);
}

template<class T>
struct IsComplex {
    static const bool value = is_same_type<af::cfloat, T>::value ||
                              is_same_type<af::cdouble, T>::value;
};

template<class T>
struct IsFloatingPoint {
    static const bool value = is_same_type<half_float::half, T>::value ||
                              is_same_type<float, T>::value ||
                              is_same_type<double, T>::value ||
                              is_same_type<long double, T>::value;
};

bool noDoubleTests(af::dtype ty);

bool noHalfTests(af::dtype ty);

#define SUPPORTED_TYPE_CHECK(type)                                \
    if (noDoubleTests((af_dtype)af::dtype_traits<type>::af_type)) \
        GTEST_SKIP() << "Device doesn't support Doubles";         \
    if (noHalfTests((af_dtype)af::dtype_traits<type>::af_type))   \
    GTEST_SKIP() << "Device doesn't support Half"

#define LAPACK_ENABLED_CHECK() \
    if (!af::isLAPACKAvailable()) GTEST_SKIP() << "LAPACK Not Configured."

#define IMAGEIO_ENABLED_CHECK() \
    if (!af::isImageIOAvailable()) GTEST_SKIP() << "Image IO Not Configured"

#ifdef AF_WITH_FAST_MATH
#define SKIP_IF_FAST_MATH_ENABLED() \
    GTEST_SKIP() << "ArrayFire compiled with AF_WITH_FAST_MATH"
#else
#define SKIP_IF_FAST_MATH_ENABLED()
#endif

template<typename TO, typename FROM>
TO convert_to(FROM in) {
    return TO(in);
}

// TODO: perform conversion on device for CUDA and OpenCL
template<typename T>
af_err conv_image(af_array *out, af_array in);

template<typename T>
af::array cpu_randu(const af::dim4 &dims);

void cleanSlate();

//********** arrayfire custom test asserts ***********

// Overloading unary + op is needed to make unsigned char values printable
//  as numbers
af_half abs(af_half in);

af_half operator-(af_half lhs, af_half rhs);

const af::cfloat &operator+(const af::cfloat &val);

const af::cdouble &operator+(const af::cdouble &val);

const af_half &operator+(const af_half &val);

// Calculate a multi-dimensional coordinates' linearized index
dim_t ravelIdx(const af::dim4 &coords, const af::dim4 &strides);

// Calculate a linearized index's multi-dimensonal coordinates in an af::array,
//  given its dimension sizes and strides
af::dim4 unravelIdx(dim_t idx, const af::dim4 &dims, const af::dim4 &strides);

af::dim4 unravelIdx(dim_t idx, af::array arr);

af::dim4 calcStrides(const af::dim4 &parentDim);

std::string minimalDim4(const af::dim4 &coords, const af::dim4 &dims);

template<typename T>
std::string printContext(const std::vector<T> &hGold,
                         const std::string &goldName,
                         const std::vector<T> &hOut, const std::string &outName,
                         const af::dim4 &arrDims, const af::dim4 &arrStrides,
                         dim_t idx);

struct FloatTag {};
struct IntegerTag {};

template<typename T>
::testing::AssertionResult elemWiseEq(
    const std::string &aName, const std::string &bName, const std::vector<T> &a,
    const af::dim4 &aDims, const std::vector<T> &b, const af::dim4 &bDims,
    float maxAbsDiff, IntegerTag);

template<typename T>
::testing::AssertionResult elemWiseEq(
    const std::string &aName, const std::string &bName, const std::vector<T> &a,
    const af::dim4 &aDims, const std::vector<T> &b, const af::dim4 &bDims,
    float maxAbsDiff, FloatTag);

template<typename T>
::testing::AssertionResult elemWiseEq(const std::string &aName,
                                      const std::string &bName,
                                      const af::array &a, const af::array &b,
                                      float maxAbsDiff);

::testing::AssertionResult assertArrayEq(const std::string &aName,
                                         const std::string &bName,
                                         const af::array &a, const af::array &b,
                                         float maxAbsDiff = 0.f);

// Called by ASSERT_VEC_ARRAY_EQ
template<typename T>
::testing::AssertionResult assertArrayEq(
    const std::string &aName, const std::string &aDimsName,
    const std::string &bName, const std::vector<T> &hA, const af::dim4 &aDims,
    const af::array &b, float maxAbsDiff = 0.0f);

// To support C API
::testing::AssertionResult assertArrayEq(const std::string &aName,
                                         const std::string &bName,
                                         const af_array a, const af_array b);

// To support C API
template<typename T>
::testing::AssertionResult assertArrayEq(const std::string &hA_name,
                                         const std::string &aDimsName,
                                         const std::string &bName,
                                         const std::vector<T> &hA,
                                         const af::dim4 &aDims,
                                         const af_array b);

// Called by ASSERT_ARRAYS_NEAR
::testing::AssertionResult assertArrayNear(const std::string &aName,
                                           const std::string &bName,
                                           const std::string &maxAbsDiffName,
                                           const af::array &a,
                                           const af::array &b,
                                           float maxAbsDiff);

::testing::AssertionResult assertImageNear(const std::string &aName,
                                           const std::string &bName,
                                           const std::string &maxAbsDiffName,
                                           const af_array &a, const af_array &b,
                                           float maxAbsDiff);

::testing::AssertionResult assertImageNear(const std::string &aName,
                                           const std::string &bName,
                                           const std::string &maxAbsDiffName,
                                           const af::array &a,
                                           const af::array &b,
                                           float maxAbsDiff);

// Called by ASSERT_VEC_ARRAY_NEAR
template<typename T>
::testing::AssertionResult assertArrayNear(
    const std::string &hA_name, const std::string &aDimsName,
    const std::string &bName, const std::string &maxAbsDiffName,
    const std::vector<T> &hA, const af::dim4 &aDims, const af::array &b,
    float maxAbsDiff);

// To support C API
::testing::AssertionResult assertArrayNear(const std::string &aName,
                                           const std::string &bName,
                                           const std::string &maxAbsDiffName,
                                           const af_array a, const af_array b,
                                           float maxAbsDiff);

// To support C API
template<typename T>
::testing::AssertionResult assertArrayNear(const std::string &hA_name,
                                           const std::string &aDimsName,
                                           const std::string &bName,
                                           const std::string &maxAbsDiffName,
                                           const std::vector<T> &hA,
                                           const af::dim4 &aDims,
                                           const af_array b, float maxAbsDiff);

::testing::AssertionResult assertRefEq(const std::string &hA_name,
                                       const std::string &expected_name,
                                       const af::array &a, int expected);

/// Checks if the C-API arrayfire function returns successfully
///
/// \param[in] CALL This is the arrayfire C function
#define ASSERT_SUCCESS(CALL) ASSERT_EQ(AF_SUCCESS, CALL)

/// Compares two af::array or af_arrays for their types, dims, and values
/// (strict equality).
///
/// \param[in] EXPECTED The expected array of the assertion
/// \param[in] ACTUAL The actual resulting array from the calculation
#define ASSERT_ARRAYS_EQ(EXPECTED, ACTUAL) \
    ASSERT_PRED_FORMAT2(assertArrayEq, EXPECTED, ACTUAL)

/// Same as ASSERT_ARRAYS_EQ, but for cases when a "special" output array is
/// given to the function.
/// The special array can be null, a full-sized array, a subarray, or reordered
/// Can only be used for testing C-API functions currently
///
/// \param[in] EXPECTED The expected array of the assertion
/// \param[in] ACTUAL The actual resulting array from the calculation
#define ASSERT_SPECIAL_ARRAYS_EQ(EXPECTED, ACTUAL, META) \
    ASSERT_PRED_FORMAT3(assertArrayEq, EXPECTED, ACTUAL, META)

/// Compares a std::vector with an af::/af_array for their types, dims, and
/// values (strict equality).
///
/// \param[in] EXPECTED_VEC The vector that represents the expected array
/// \param[in] EXPECTED_ARR_DIMS The dimensions of the expected array
/// \param[in] ACTUAL_ARR The actual resulting array from the calculation
#define ASSERT_VEC_ARRAY_EQ(EXPECTED_VEC, EXPECTED_ARR_DIMS, ACTUAL_ARR) \
    ASSERT_PRED_FORMAT3(assertArrayEq, EXPECTED_VEC, EXPECTED_ARR_DIMS,  \
                        ACTUAL_ARR)

/// Compares two af::array or af_arrays for their types, dims, and values
/// (strict equality).
///
/// \param[in] EXPECTED The expected array of the assertion
/// \param[in] ACTUAL The actual resulting array from the calculation
#define EXPECT_ARRAYS_EQ(EXPECTED, ACTUAL) \
    EXPECT_PRED_FORMAT2(assertArrayEq, EXPECTED, ACTUAL)

/// Same as EXPECT_ARRAYS_EQ, but for cases when a "special" output array is
/// given to the function.
/// The special array can be null, a full-sized array, a subarray, or reordered
/// Can only be used for testing C-API functions currently
///
/// \param[in] EXPECTED The expected array of the assertion
/// \param[in] ACTUAL The actual resulting array from the calculation
#define EXPECT_SPECIAL_ARRAYS_EQ(EXPECTED, ACTUAL, META) \
    EXPECT_PRED_FORMAT3(assertArrayEq, EXPECTED, ACTUAL, META)

/// Compares a std::vector with an af::/af_array for their types, dims, and
/// values (strict equality).
///
/// \param[in] EXPECTED_VEC The vector that represents the expected array
/// \param[in] EXPECTED_ARR_DIMS The dimensions of the expected array
/// \param[in] ACTUAL_ARR The actual resulting array from the calculation
#define EXPECT_VEC_ARRAY_EQ(EXPECTED_VEC, EXPECTED_ARR_DIMS, ACTUAL_ARR) \
    EXPECT_PRED_FORMAT3(assertArrayEq, EXPECTED_VEC, EXPECTED_ARR_DIMS,  \
                        ACTUAL_ARR)

/// Compares two af::array or af_arrays for their type, dims, and values (with a
/// given tolerance).
///
/// \param[in] EXPECTED Expected value of the assertion
/// \param[in] ACTUAL Actual value of the calculation
/// \param[in] MAX_ABSDIFF Expected maximum absolute difference between
///            elements of EXPECTED and ACTUAL
///
/// \NOTE: This macro will deallocate the af_arrays after the call
#define ASSERT_ARRAYS_NEAR(EXPECTED, ACTUAL, MAX_ABSDIFF) \
    ASSERT_PRED_FORMAT3(assertArrayNear, EXPECTED, ACTUAL, MAX_ABSDIFF)

/// Compares two af::array or af_arrays for their type, dims, and values (with a
/// given tolerance).
///
/// \param[in] EXPECTED Expected value of the assertion
/// \param[in] ACTUAL Actual value of the calculation
/// \param[in] MAX_ABSDIFF Expected maximum absolute difference between
///            elements of EXPECTED and ACTUAL
///
/// \NOTE: This macro will deallocate the af_arrays after the call
#define ASSERT_IMAGES_NEAR(EXPECTED, ACTUAL, MAX_ABSDIFF) \
    ASSERT_PRED_FORMAT3(assertImageNear, EXPECTED, ACTUAL, MAX_ABSDIFF)

/// Compares a std::vector with an af::array for their dims and values (with a
/// given tolerance).
///
/// \param[in] EXPECTED_VEC The vector that represents the expected array
/// \param[in] EXPECTED_ARR_DIMS The dimensions of the expected array
/// \param[in] ACTUAL_ARR The actual array from the calculation
/// \param[in] MAX_ABSDIFF Expected maximum absolute difference between
///            elements of EXPECTED and ACTUAL
#define ASSERT_VEC_ARRAY_NEAR(EXPECTED_VEC, EXPECTED_ARR_DIMS, ACTUAL_ARR, \
                              MAX_ABSDIFF)                                 \
    ASSERT_PRED_FORMAT4(assertArrayNear, EXPECTED_VEC, EXPECTED_ARR_DIMS,  \
                        ACTUAL_ARR, MAX_ABSDIFF)

/// Compares two af::array or af_arrays for their type, dims, and values (with a
/// given tolerance).
///
/// \param[in] EXPECTED Expected value of the assertion
/// \param[in] ACTUAL Actual value of the calculation
/// \param[in] MAX_ABSDIFF Expected maximum absolute difference between
///            elements of EXPECTED and ACTUAL
///
/// \NOTE: This macro will deallocate the af_arrays after the call
#define EXPECT_ARRAYS_NEAR(EXPECTED, ACTUAL, MAX_ABSDIFF) \
    EXPECT_PRED_FORMAT3(assertArrayNear, EXPECTED, ACTUAL, MAX_ABSDIFF)

/// Compares two af::array or af_arrays for their type, dims, and values (with a
/// given tolerance).
///
/// \param[in] EXPECTED Expected value of the assertion
/// \param[in] ACTUAL Actual value of the calculation
/// \param[in] MAX_ABSDIFF Expected maximum absolute difference between
///            elements of EXPECTED and ACTUAL
///
/// \NOTE: This macro will deallocate the af_arrays after the call
#define EXPECT_IMAGES_NEAR(EXPECTED, ACTUAL, MAX_ABSDIFF) \
    EXPECT_PRED_FORMAT3(assertImageNear, EXPECTED, ACTUAL, MAX_ABSDIFF)

/// Compares a std::vector with an af::array for their dims and values (with a
/// given tolerance).
///
/// \param[in] EXPECTED_VEC The vector that represents the expected array
/// \param[in] EXPECTED_ARR_DIMS The dimensions of the expected array
/// \param[in] ACTUAL_ARR The actual array from the calculation
/// \param[in] MAX_ABSDIFF Expected maximum absolute difference between
///            elements of EXPECTED and ACTUAL
#define EXPECT_VEC_ARRAY_NEAR(EXPECTED_VEC, EXPECTED_ARR_DIMS, ACTUAL_ARR, \
                              MAX_ABSDIFF)                                 \
    EXPECT_PRED_FORMAT4(assertArrayNear, EXPECTED_VEC, EXPECTED_ARR_DIMS,  \
                        ACTUAL_ARR, MAX_ABSDIFF)

#define ASSERT_REF(arr, expected) \
    ASSERT_PRED_FORMAT2(assertRefEq, arr, expected)

#if defined(USE_MTX)
::testing::AssertionResult mtxReadSparseMatrix(af::array &out,
                                               const char *fileName);
#endif  // USE_MTX

enum TestArrayType {
    FULL_ARRAY = 0,
    JIT_ARRAY,
    SUB_ARRAY,
    REORDERED_ARRAY,
    NULL_ARRAY,
    EMPTY_ARRAY,
};

// NULL_ARRAY: Test af_* function when given a null array as its output
// EMPTY_ARRAY: Test af_* function when given an empty array as its output
// FULL_ARRAY: Test af_* function when given an output array that is the same
// size as the expected output SUB_ARRAY: Test af_* function when given an
// output array that is a sub-array of a larger array (the sub-array size is
// still the same size as the expected output). Only the sub-array must be
// modified by the af_* function REORDERED_ARRAY: Test af_* function when given
// an output array that was previously reordered (but after the reorder, has
// still the same shape as the expected output). This specifically uses the
// reorder behavior when dim0 is kept, and thus no data movement is done - only
// the dims and strides are modified
#define TestOutputArrayType TestArrayType
#define TestOutputArrayTypeALL \
    { FULL_ARRAY, JIT_ARRAY, SUB_ARRAY, REORDERED_ARRAY, NULL_ARRAY, EMPTY_ARRAY }

class TestOutputArrayInfo {
   protected:
    af_array out_arr;
    af_array out_arr_cpy;
    af_array out_subarr;
    dim_t out_subarr_ndims;
    af_seq out_subarr_idxs[4];
    TestOutputArrayType out_arr_type;
    const char *TestArrayTypeName[6]{"FULL_ARRAY", "JIT_ARRAY","SUB_ARRAY",
                                     "REORDERED_ARRAY", "NULL_ARRAY",
                                     "EMPTY_ARRAY"};

   public:
    TestOutputArrayInfo();

    TestOutputArrayInfo(const TestOutputArrayType arr_type);

    ~TestOutputArrayInfo();

    void init(const unsigned ndims, const dim_t *const dims, const af_dtype ty);

    void init(const unsigned ndims, const dim_t *const dims, const af_dtype ty,
              const af_seq *const subarr_idxs);

    void init(double val, const unsigned ndims, const dim_t *const dims,
              const af_dtype ty);

    void init(double val, const unsigned ndims, const dim_t *const dims,
              const af_dtype ty, const af_seq *const subarr_idxs);

    af_array getOutput() const;

    void setOutput(const af_array array);

    af_array getFullOutput() const;
    af_array getFullOutputCopy() const;
    const af_seq *getSubArrayIdxs() const;
    dim_t getSubArrayNumDims() const;
    TestOutputArrayType getOutputArrayType() const;
    const char *getOutputArrayTypeName() const {
        return TestArrayTypeName[out_arr_type];
    };
};

// Generates a random array. testWriteToOutputArray expects that it will receive
// the same af_array that this generates after the af_* function is called
void genRegularArray(TestOutputArrayInfo *metadata, const unsigned ndims,
                     const dim_t *const dims, const af_dtype ty);

void genRegularArray(TestOutputArrayInfo *metadata, const double val,
                     const unsigned ndims, const dim_t *const dims,
                     const af_dtype ty);

// Generates an empty array.
void genEmptyArray(TestOutputArrayInfo *metadata, const af_dtype ty);

// Generates a large, random array, and extracts a subarray for the af_*
// function to use. testWriteToOutputArray expects that the large array that it
// receives is equal to the same large array with the gold array injected on the
// same subarray location
void genSubArray(TestOutputArrayInfo *metadata, const unsigned ndims,
                 const dim_t *const dims, const af_dtype ty);

void genSubArray(TestOutputArrayInfo *metadata, const double val,
                 const unsigned ndims, const dim_t *const dims,
                 const af_dtype ty);

// Generates a reordered array. testWriteToOutputArray expects that this array
// will still have the correct output values from the af_* function, even though
// the array was initially reordered.
void genReorderedArray(TestOutputArrayInfo *metadata, const unsigned ndims,
                       const dim_t *const dims, const af_dtype ty);

void genReorderedArray(TestOutputArrayInfo *metadata, const double val,
                       const unsigned ndims, const dim_t *const dims,
                       const af_dtype ty);

// Partner function of testWriteToOutputArray. This generates the "special"
// array that testWriteToOutputArray will use to check if the af_* function
// correctly uses an existing array as its output
void genTestOutputArray(af_array *out_ptr, const unsigned ndims,
                        const dim_t *const dims, const af_dtype ty,
                        TestOutputArrayInfo *metadata);

void genTestOutputArray(af_array *out_ptr, const double val,
                        const unsigned ndims, const dim_t *const dims,
                        const af_dtype ty, TestOutputArrayInfo *metadata);

// Partner function of genTestOutputArray. This uses the same "special"
// array that genTestOutputArray generates, and checks whether the
// af_* function wrote to that array correctly
::testing::AssertionResult testWriteToOutputArray(
    const std::string &gold_name, const std::string &result_name,
    const af_array gold, const af_array out, TestOutputArrayInfo *metadata);

// Called by ASSERT_SPECIAL_ARRAYS_EQ
::testing::AssertionResult assertArrayEq(const std::string &aName,
                                         const std::string &bName,
                                         const std::string &metadataName,
                                         const af_array a, const af_array b,
                                         TestOutputArrayInfo *metadata);

// NULL_ARRAY: Test af_* function when given a null array as its output
// EMPTY_ARRAY: Test af_* function for array with no elements
// FULL_ARRAY: Test af_* function with a normal array
// SUB_ARRAY: Test af_* function with a array that is a sub-array of a
// larger parent array.
// REORDERED_ARRAY: Test af_* function with an array that was previously
// reordered (but after the reorder, has still the same shape as the
// expected input). This specifically uses the reorder behavior when dim0 is
// kept, and thus no data movement is done - only the dims and strides are
// modified
#define TestInputArrayType TestArrayType
#define TestInputArrayTypeALL \
    { FULL_ARRAY, JIT_ARRAY, SUB_ARRAY, REORDERED_ARRAY, NULL_ARRAY, EMPTY_ARRAY }

class TestInputArrayInfo {
   protected:
    af_array in_arr_zero;
    af_array in_arr;
    af_array in_subarr;
    dim_t in_subarr_ndims;
    af_seq in_subarr_idxs[4];
    TestInputArrayType in_arr_type;
    const char *TestArrayTypeName[6]{"FULL_ARRAY", "JIT_ARRAY", "SUB_ARRAY",
                                     "REORDERED_ARRAY", "NULL_ARRAY",
                                     "EMPTY_ARRAY"};

   public:
    // Creates a NULL_ARRAY instance
    TestInputArrayInfo();

    // Creates a NULL_ARRAY instance with specified arr_type
    // The array content is added by genTestInputArray() or gen<TYPE>Array()
    // functions
    TestInputArrayInfo(const TestInputArrayType arr_type);

    ~TestInputArrayInfo();

    // Assigns inArray to the instance
    void init(const af_array inArray);
    // ndims, dims are from the parent array
    // dimensions of subarr_idxs from parent == dimensions inArray
    void init(const unsigned ndims, const dim_t *const dims,
              const af_seq *const subarr_idxs, const af_array inArray);

    af_array getInput() const;
    af_array getFullInput() const;
    const af_seq *getSubArrayIdxs() const;
    dim_t getSubArrayNumDims() const;
    TestInputArrayType getInputArrayType() const;
    const char *getInputArrayTypeName() const {
        return TestArrayTypeName[in_arr_type];
    };
};

void genEmptyArray(TestInputArrayInfo *metadata, const af_dtype ty);
void genRegularArray(TestInputArrayInfo *metadata, const af_array inArray);
void genSubArray(TestInputArrayInfo *metadata, const af_array inArray);
void genReorderedArray(TestInputArrayInfo *metadata, const af_array inArray);

// Generates an array corresponding to type of metadata.  Data from inArray
// will be copied into the generated array.
void genTestInputArray(af_array *out_ptr, const af_array inArray,
                       TestInputArrayInfo *metadata);

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
