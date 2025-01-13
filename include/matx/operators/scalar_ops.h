////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2021, NVIDIA Corporation
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
/////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <type_traits>
#include <cuda/std/cmath>

namespace matx {
namespace detail {

// This file defines operators on a scalar


// Utility macro for generating functions that have half precision intrinsics as
// an option. Lots of verbose code in here because of compiler bugs with
// constexpr if
#define MATX_UNARY_OP_GEN(FUNC, OPNAME)                                        \
  template <typename T> \
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto internal_##FUNC(T v1) { \
    if constexpr (is_matx_type_v<T>) {    \
      return FUNC(v1); \
    } \
    else { \
      return cuda::std::FUNC(v1);    \
    } \
  } \
  template <typename T> struct OPNAME##Op {                                     \
    static __MATX_INLINE__ std::string str(const std::string &in) { return std::string(#FUNC) + "(" + in + ")"; }                 \
    template <matx::detail::VecWidth InWidth, typename T1V> \
    __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(const T1V &v1) const { \
      return UnaryVecFunc<InWidth>(internal_##FUNC<T>, v1); \
    } \
    using value_type = std::invoke_result_t<decltype(internal_##FUNC<T>), T>; \
  };

// Unary operator with a custom function
#define MATX_UNARY_OP_GEN_NOFUNC(FUNC, OPNAME)                                        \
  template <typename T> struct OPNAME##Op {                                     \
    static __MATX_INLINE__ std::string str(const std::string &in) { return std::string(#FUNC) + "(" + in + ")"; }                 \
    template <matx::detail::VecWidth InWidth, typename T1V> \
    __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(const T1V &v1) const { \
      return UnaryVecFunc<InWidth>(internal_##FUNC<T>, v1); \
    } \
    using value_type = std::invoke_result_t<decltype(internal_##FUNC<T>), T>; \
  };  

#define MATX_BINARY_OP_GEN(FUNC, OPNAME)                                       \
  template <typename T1, typename T2> \
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto internal_##FUNC(T1 v1, T2 v2) { \
    if constexpr (is_matx_type_v<T1> || is_matx_type_v<T2>) {    \
      return FUNC(v1, v2); \
    } \
    else { \
      cuda::std::FUNC(v1, v2);    \
    } \
  } \
  template <typename T1, typename T2> struct OPNAME##Op {                                     \
    static __MATX_INLINE__ std::string str(const std::string &in1, const std::string &in2) { return std::string(#FUNC) + "(" + in1 + "," + in2 + ")"; } \
    template <matx::detail::VecWidth InWidth, typename T1V, typename T2V> \
    __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(const T1V &v1, const T2V &v2) const { \
      return BinVecFunc<InWidth>(internal_##FUNC<T1, T2>, v1, v2); \
    } \
    using value_type = std::invoke_result_t<decltype(internal_##FUNC<T1, T2>), T1, T2>; \
  };
  
#define MATX_BINARY_OP_GEN_OPERATOR(FUNC, OPNAME, OPSYM)                                       \
  template <typename T1, typename T2> \
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto internal_##FUNC(T1 v1, T2 v2) { \
    return v1 OPSYM v2; \
  } \
  template <typename T1, typename T2> struct OPNAME##Op {                   \
    static __MATX_INLINE__ std::string str(const std::string &in1, const std::string &in2) { return std::string(#FUNC) + "(" + in1 + "," + in2 + ")"; } \
    template <matx::detail::VecWidth InWidth, typename T1V, typename T2V> \
    __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(const T1V &v1, const T2V &v2) const { \
      return BinVecFunc<InWidth>(internal_##FUNC<T1, T2>, v1, v2); \
    } \
    using value_type = std::invoke_result_t<decltype(internal_##FUNC<T1, T2>), T1, T2>; \
  };

// Binary operator with a custom function
#define MATX_BINARY_OP_NOFUNC(FUNC, OPNAME)                                       \
  template <typename T1, typename T2> struct OPNAME##Op {                                     \
    static __MATX_INLINE__ std::string str(const std::string &in1, const std::string &in2) { return std::string(#FUNC) + "(" + in1 + "," + in2 + ")"; } \
    template <matx::detail::VecWidth InWidth, typename T1V, typename T2V> \
    __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(const T1V &v1, const T2V &v2) const { \
      return BinVecFunc<InWidth>(internal_##FUNC<T1, T2>, v1, v2); \
    } \
    using value_type = std::invoke_result_t<decltype(internal_##FUNC<T1, T2>), T1, T2>; \
  };  



// Helper function to apply a callable binary operator onto two inputs. There are many compile-time
// branches in here because we need to handle both scalar and vector inputs on both sides.
template <matx::detail::VecWidth InWidth, typename BinOpFunc, typename T1, typename T2>
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ auto BinVecFunc(const BinOpFunc &func, const T1 &v1, const T2 &v2) {
  if constexpr (InWidth == VecWidth::SCALAR) {
    return func(v1, v2);
  }
  else if constexpr (InWidth == VecWidth::ONE) {
    if constexpr (is_vector_v<T1>) {
      if constexpr (is_vector_v<T2>) {
        using res_type = matx::detail::Vector<decltype(func(v1.data[0], v2.data[0])), 1>;
        return res_type{func(v1.data[0], v2.data[0])};
      }
      else {
        using res_type = matx::detail::Vector<decltype(func(v1.data[0], v2)), 1>;
        return res_type{func(v1.data[0], v2)};
      }
    }
    else {
      if constexpr (is_vector_v<T2>) {
        using res_type = matx::detail::Vector<decltype(func(v1, v2.data[0])), 1>;
        return res_type{func(v1, v2.data[0])};
      }
      else {
        using res_type = matx::detail::Vector<decltype(func(v1, v2)), 1>;
        return res_type{func(v1, v2)};
      }
    }
  }
  else if constexpr (InWidth == VecWidth::TWO) {
    if constexpr (is_vector_v<T1>) {
      if constexpr (is_vector_v<T2>) {
        using res_type = matx::detail::Vector<decltype(func(v1.data[0], v2.data[0])), 2>;
        return res_type{  func(v1.data[0], v2.data[0]),
                          func(v1.data[1], v2.data[1])};
      }
      else {
        using res_type = matx::detail::Vector<decltype(func(v1.data[0], v2)), 2>;
        return res_type{  func(v1.data[0], v2),
                          func(v1.data[1], v2)};
      }
    }
    else {
      if constexpr (is_vector_v<T2>) {
        using res_type = matx::detail::Vector<decltype(func(v1, v2.data[0])), 2>;
        return res_type{  func(v1, v2.data[0]),
                          func(v1, v2.data[1])};
      }
      else {
        using res_type = matx::detail::Vector<decltype(func(v1, v2)), 2>;
        const auto val = func(v1, v2);
        return res_type{val, val};
      }
    }
  }
  else if constexpr (InWidth == VecWidth::FOUR) {
    if constexpr (is_vector_v<T1>) {
      if constexpr (is_vector_v<T2>) {
        using res_type = matx::detail::Vector<decltype(func(v1.data[0], v2.data[0])), 4>;
        return res_type{  func(v1.data[0], v2.data[0]),
                          func(v1.data[1], v2.data[1]),
                          func(v1.data[2], v2.data[2]),
                          func(v1.data[3], v2.data[3])};
      }
      else {
        using res_type = matx::detail::Vector<decltype(func(v1.data[0], v2)), 4>;
        return res_type{  func(v1.data[0], v2),
                          func(v1.data[1], v2),
                          func(v1.data[2], v2),
                          func(v1.data[3], v2)};
      }
    }
    else {
      if constexpr (is_vector_v<T2>) {
        using res_type = matx::detail::Vector<decltype(func(v1, v2.data[0])), 4>;
        return res_type{  func(v1, v2.data[0]),
                          func(v1, v2.data[1]),
                          func(v1, v2.data[2]),
                          func(v1, v2.data[3])};
      }
      else {
        using res_type = matx::detail::Vector<decltype(func(v1, v2)), 4>;
        const auto val = func(v1, v2);
        return res_type{val, val, val, val};
      }
    }
  }
}

template <matx::detail::VecWidth InWidth, typename UnaryOpFunc, typename T1>
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ auto UnaryVecFunc(const UnaryOpFunc &func, const T1 &v1) {
  if constexpr (InWidth == VecWidth::SCALAR) {
    return func(v1);
  }
  else if constexpr (InWidth == VecWidth::ONE) {
    if constexpr (is_vector_v<T1>) {
      using res_type = matx::detail::Vector<decltype(func(v1.data[0])), 1>;
      return res_type{func(v1.data[0])};
    }
    else {
      using res_type = matx::detail::Vector<decltype(func(v1)), 1>;
      return res_type{func(v1)};
    }
  }
  else if constexpr (InWidth == VecWidth::TWO) {
    if constexpr (is_vector_v<T1>) {
      using res_type = matx::detail::Vector<decltype(func(v1.data[0])), 2>;
      return res_type{  func(v1.data[0]),
                        func(v1.data[1])};
    }
    else {
      using res_type = matx::detail::Vector<decltype(func(v1)), 2>;
      const auto ret = func(v1);
      return res_type{  ret,
                        ret};
    }
  }
  else if constexpr (InWidth == VecWidth::FOUR) {
    if constexpr (is_vector_v<T1>) {
      using res_type = matx::detail::Vector<decltype(func(v1.data[0])), 4>;
      return res_type{  func(v1.data[0]),
                        func(v1.data[1]),
                        func(v1.data[2]),
                        func(v1.data[3])};
    }
    else {
      using res_type = matx::detail::Vector<decltype(func(v1)), 4>;
      const auto ret = func(v1);
      return res_type{ret, ret, ret, ret};
    }
  }
}  



MATX_UNARY_OP_GEN(ceil, Ceil);
MATX_UNARY_OP_GEN(floor, Floor);
MATX_UNARY_OP_GEN(round, Round);
MATX_UNARY_OP_GEN(exp, Exp);
MATX_UNARY_OP_GEN(sqrt, Sqrt);
MATX_UNARY_OP_GEN(log10, Log10);
MATX_UNARY_OP_GEN(log2, Log2);
MATX_UNARY_OP_GEN(log, Log);
MATX_UNARY_OP_GEN(abs, Abs);
MATX_UNARY_OP_GEN(tan, Tan);
MATX_UNARY_OP_GEN(asin, Asin);
MATX_UNARY_OP_GEN(acos, Acos);
MATX_UNARY_OP_GEN(atan, Atan);
MATX_UNARY_OP_GEN(sinh, Sinh);
MATX_UNARY_OP_GEN(cosh, Cosh);
MATX_UNARY_OP_GEN(tanh, Tanh);
MATX_UNARY_OP_GEN(asinh, Asinh);
MATX_UNARY_OP_GEN(acosh, Acosh);
MATX_UNARY_OP_GEN(atanh, Atanh);


template <typename T>
static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto internal_rsqrt(T v1) {
  if constexpr (is_matx_type_v<T>){
    return rsqrt(v1);
  }
  else {
#ifdef __CUDACC__
    return ::rsqrt(v1);
#else
    return static_cast<T>(1) / sqrt(v1);
#endif
  }  
}
MATX_UNARY_OP_GEN_NOFUNC(rsqrt, RSqrt);

template <typename T>
static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto internal_csqrt(T v1) {
  return sqrt(static_cast<cuda::std::complex<T>>(v1));
}
MATX_UNARY_OP_GEN_NOFUNC(csqrt, CSqrt);


template <typename T>
static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto internal_conj(T v1) {
  if constexpr (is_cuda_complex_v<T>) {
    return cuda::std::conj(v1);
  }
  else {
    return v1;
  }
}
MATX_UNARY_OP_GEN_NOFUNC(conj, Conj);

template <typename T>
static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto internal_sin(T v1) {
  if constexpr (is_matx_type_v<T>) {
    return matx::sin(v1);
  }
  else {
    return cuda::std::sin(v1);
  }
}
MATX_UNARY_OP_GEN_NOFUNC(sin, Sin);


template <typename T>
static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto internal_cos(T v1) {
  if constexpr (is_matx_type_v<T>) {
    return matx::cos(v1);
  }
  else {
    return cuda::std::cos(v1);
  }
}
MATX_UNARY_OP_GEN_NOFUNC(cos, Cos);

template <typename T, std::enable_if_t<std::is_floating_point_v<T>, bool> = true>
static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto internal_expj(T v1) {
  if constexpr (is_matx_type_v<T>) {
    return matxHalfComplex<T>{_internal_cos(v1), _internal_sin(v1)};
  }
  else {
    return cuda::std::complex<T>{_internal_cos(v1), _internal_sin(v1)};
  }  
}
MATX_UNARY_OP_GEN_NOFUNC(expj, Expj);


template <typename T>
static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto internal_abs2(T v1) {
  if constexpr (is_complex_v<T>) {
    return v1.real() * v1.real() + v1.imag() * v1.imag();
  }
  else {
    return v1 * v1;
  }
}
MATX_UNARY_OP_GEN_NOFUNC(abs2, Abs2);

template <typename T>
static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto internal_normcdf(T v1) {
  return normcdf(v1);
}
MATX_UNARY_OP_GEN_NOFUNC(normcdf, NormCdf);


template <typename T>
static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto internal_real(T v1) {
  return v1.real();
}
MATX_UNARY_OP_GEN_NOFUNC(real, Real);

template <typename T>
static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto internal_imag(T v1) {
  return v1.imag();
}
MATX_UNARY_OP_GEN_NOFUNC(imag, Imag);


template <typename T>
static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto internal_angle(T v1) {
  if constexpr (is_cuda_complex_v<T>) {
    return cuda::std::atan2(v1.imag(), v1.real());
  }
  else {
    return atan2(v1.imag(), v1.real());
  }
}
MATX_UNARY_OP_GEN_NOFUNC(angle, Angle);

template <typename T>
static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto internal_subneg(T v1) {
  return -v1;
}
MATX_UNARY_OP_GEN_NOFUNC(subneg, SubNeg);

template <typename T>
static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto internal_not(T v1) {
  return !v1;
}
MATX_UNARY_OP_GEN_NOFUNC(not, Not);

template <typename T>
static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto internal_isnan(T v1) {
  using conversionType = typename matx::detail::value_promote_t<T>;
  if constexpr(!std::is_floating_point_v<conversionType>) {
      return false;
  }

  using castType = matx::detail::matx_convert_complex_type<T>;
  if constexpr(is_complex_v<T>) {
    return cuda::std::isnan(static_cast<typename castType::value_type>(v1.real())) || cuda::std::isnan(static_cast<typename castType::value_type>(v1.imag()));
  } else {
    return cuda::std::isnan(static_cast<castType>(v1));
  }
}
MATX_UNARY_OP_GEN_NOFUNC(isnan, IsNan);

template <typename T>
static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto internal_isinf(T v1) {
  using conversionType = typename matx::detail::value_promote_t<T>;
  if constexpr(!std::is_floating_point_v<conversionType>) {
    return false;
  }

  using castType = matx::detail::matx_convert_complex_type<T>;
  if constexpr(is_complex_v<T>) {
    return cuda::std::isinf(static_cast<typename castType::value_type>(v1.real())) || cuda::std::isinf(static_cast<typename castType::value_type>(v1.imag()));
  } else {
    return cuda::std::isinf(static_cast<castType>(v1));
  } 
}
MATX_UNARY_OP_GEN_NOFUNC(isinf, IsInf);


// Binary Operators
MATX_BINARY_OP_GEN_OPERATOR(add, Add, +);
MATX_BINARY_OP_GEN_OPERATOR(sub, Sub, -);
MATX_BINARY_OP_GEN_OPERATOR(mul, Mul, *);
MATX_BINARY_OP_GEN_OPERATOR(div, Div, /);
MATX_BINARY_OP_GEN_OPERATOR(mod, Mod, %);

template <typename T1, typename T2>
static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto internal_fmod(T1 v1, T2 v2) {
  return cuda::std::fmod(v1, v2);
}  
MATX_BINARY_OP_NOFUNC(fmod, FMod);

template <typename T1, typename T2>
static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto internal_atan2(T1 v1, T2 v2) {
  if constexpr (is_matx_half_v<T1> || is_matx_half_v<T2>) {
    return atan2(v1, v2);
  }
  else {
    return cuda::std::atan2(v1, v2);
  }
}  
MATX_BINARY_OP_NOFUNC(atan2, Atan2);

MATX_BINARY_OP_GEN(pow, Pow);

template <typename T1, typename T2>
static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto internal_max(T1 v1, T2 v2) {
  return cuda::std::max(v1, v2);
}  
MATX_BINARY_OP_NOFUNC(max, Maximum);

template <typename T1, typename T2>
static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto internal_min(T1 v1, T2 v2) {
  return cuda::std::min(v1, v2);
}  
MATX_BINARY_OP_NOFUNC(min, Minimum);

// Logical Operators
MATX_BINARY_OP_GEN_OPERATOR(LT, LT, <);
MATX_BINARY_OP_GEN_OPERATOR(GT, GT, >);
MATX_BINARY_OP_GEN_OPERATOR(LTE, LTE, <=);
MATX_BINARY_OP_GEN_OPERATOR(GTE, GTE, >=);
MATX_BINARY_OP_GEN_OPERATOR(EQ, EQ, ==);
MATX_BINARY_OP_GEN_OPERATOR(NE, NE, !=);
MATX_BINARY_OP_GEN_OPERATOR(andand, AndAnd, &&);
MATX_BINARY_OP_GEN_OPERATOR(oror, OrOr, ||);
MATX_BINARY_OP_GEN_OPERATOR(bitand, And, &);
MATX_BINARY_OP_GEN_OPERATOR(bitor, Or, |);
MATX_BINARY_OP_GEN_OPERATOR(bitxor, Xor, ^);


} // end namespace detail
} // end namespace matx