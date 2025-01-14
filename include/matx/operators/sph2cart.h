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


#include "matx/core/type_utils.h"
#include "matx/operators/base_operator.h"

namespace matx
{
  /**
   * Operator that computes the transform from spherical to cartesian indexing
   */
  namespace detail {
    template <typename T1, typename T2, typename T3, int WHICH>
      class Sph2CartOp : public BaseOp<Sph2CartOp<T1, T2, T3, WHICH>>
    {
      private:
        typename detail::base_type_t<T1> theta_;
        typename detail::base_type_t<T2> phi_;
        typename detail::base_type_t<T3> r_;

      public:
        using matxop = bool;
        using matx_width = bool;
        using value_type = typename T1::value_type;

        __MATX_INLINE__ std::string str() const { return "sph2cart(" + get_type_str(theta_) + 
          "," + get_type_str(phi_) + "," + get_type_str(r_) + ")"; }

        VecWidth GetMaxWidth() const {
          return MaxCompatibleWidth(MaxCompatibleWidth(theta_, phi_), r_);
        }          

        __MATX_INLINE__ Sph2CartOp(const T1 &theta, const T2 &phi, const T3 &r) : theta_(theta), phi_(phi), r_(r)
        {
          ASSERT_COMPATIBLE_OP_SIZES(theta);
          ASSERT_COMPATIBLE_OP_SIZES(phi);
          ASSERT_COMPATIBLE_OP_SIZES(r);
        }

        template <VecWidth InWidth, VecWidth OutWidth, typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto operator()(Is... indices) const 
        {
          [[maybe_unused]] auto theta = get_value<InWidth, OutWidth>(theta_, indices...);
          [[maybe_unused]] auto phi = get_value<InWidth, OutWidth>(phi_, indices...);
          auto r = get_value<InWidth, OutWidth>(r_, indices...);

          if constexpr (WHICH==0) { // X
            const auto phi_cos   = internal_cos<decltype(phi), InWidth>(phi);
            const auto theta_cos = internal_cos<decltype(theta), InWidth>(theta);
            const auto r_mul_phi = internal_mul<decltype(r), decltype(phi), InWidth>(r, phi_cos);
            const auto p_mul_theta = internal_mul<decltype(r_mul_phi), decltype(theta_cos), InWidth>(r_mul_phi, theta_cos);
            return p_mul_theta;
          } else if constexpr (WHICH==1) { // Y
            const auto phi_cos   = internal_cos<decltype(phi), InWidth>(phi);
            const auto theta_sin = internal_sin<decltype(theta), InWidth>(theta);
            const auto r_mul_phi = internal_mul<decltype(r), decltype(phi_cos), InWidth>(r, phi_cos);
            const auto p_mul_theta = internal_mul<decltype(r_mul_phi), decltype(theta_sin), InWidth>(r_mul_phi, theta_sin);
            return p_mul_theta;
          } else {  // Z
            const auto phi_sin   = internal_sin<decltype(phi), InWidth>(phi);
            const auto r_mul_phi = internal_mul<decltype(r), decltype(phi_sin), InWidth>(r, phi_sin);
            return r_mul_phi;
          }
        }

        template <typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto operator()(Is... indices) const 
        {
          return this->template operator()<VecWidth::SCALAR, VecWidth::SCALAR>(indices...);
        }        

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PreRun([[maybe_unused]] ShapeType &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<T1>()) {
            theta_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }

          if constexpr (is_matx_op<T2>()) {
            phi_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }

          if constexpr (is_matx_op<T3>()) {
            r_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PostRun(ShapeType &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<T1>()) {
            theta_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }

          if constexpr (is_matx_op<T2>()) {
            phi_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }

          if constexpr (is_matx_op<T3>()) {
            r_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }
        }

        static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
        {
          return matx_max(get_rank<T1>(), get_rank<T2>(), get_rank<T3>());
        }

        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto Size(int dim) const noexcept
        {
					index_t size1 = get_expanded_size<Rank()>(theta_, dim);
					index_t size2 = get_expanded_size<Rank()>(phi_, dim);
					index_t size3 = get_expanded_size<Rank()>(r_, dim);
					return detail::matx_max(size1, size2, size3);  
        }      
    };
  }
  /**
   * Operator to compute cartesian coordiantes from spherical coordinates
   * TODO
   * @tparam T1
   *   Operator type defining theta
   *
   * @tparam T2
   *   Operator type defining phi
   *
   * @tparam T3
   *   Operator type defining radius
   *
   * @param theta
   *   Operator defining theta
   *   
   * @param phi
   *   Operator defining phi
   *
   * @param r
   *   Operator defining radius
   *
   * @returns
   *   Tuple of operators for x, y, and z.
   */
  template <typename T1, typename T2, typename T3>
    auto __MATX_INLINE__ sph2cart(const T1 &theta, const T2 &phi, const T3 &r)
    {
      return cuda::std::tuple{ 
        detail::Sph2CartOp<T1,T2,T3,0>(theta, phi, r),
        detail::Sph2CartOp<T1,T2,T3,1>(theta, phi, r),
        detail::Sph2CartOp<T1,T2,T3,2>(theta, phi, r)};
    };

} // end namespace matx
