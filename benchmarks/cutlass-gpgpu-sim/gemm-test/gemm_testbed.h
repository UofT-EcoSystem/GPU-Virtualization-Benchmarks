/***************************************************************************************************
 * Copyright (c) 2017-2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Test environment for GEMM
*/

#pragma once

#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

#include <cutlass/matrix_traits.h>
#include <cutlass/util/platform.h>

#include <util/host_tensor.h>
#include <util/tensor_view_io.h>
#include <util/type_traits.h>

namespace cutlass {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <cutlass::GemmOperand::Kind kOperand_,
          cutlass::MatrixLayout::Kind kLayout_,
          typename Scalar_,
          typename WmmaShape_>
struct WmmaMatrix;
}

namespace test {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct GemmTestbedTraits : public cutlass::TypeTraits<T> {};

template <cutlass::GemmOperand::Kind kOperand_,
          cutlass::MatrixLayout::Kind kLayout_,
          typename Scalar_,
          typename WmmaShape_>
struct GemmTestbedTraits<cutlass::WmmaMatrix<kOperand_, kLayout_, Scalar_, WmmaShape_> > {
  static cudaDataType_t const cublas_type = cutlass::TypeTraits<Scalar_>::cublas_type;
  typedef Scalar_ host_type;
  typedef Scalar_ device_type;
  static inline double remove_negative_zero(double x) { return x == -0.0 ? 0.0 : x; }
  static inline double to_print(double x) { return x; }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename AType, typename BType, typename CType, typename Accumulator, typename Scalar>
struct GemmTestbed {
  //
  // Type definitions
  //

  /// Host tensor for operand A
  typedef cutlass::HostTensor<AType> HostTensorA;

  /// Host tensor for operand B
  typedef cutlass::HostTensor<BType> HostTensorB;

  /// Host tensor for operand C
  typedef cutlass::HostTensor<CType> HostTensorC;

  /// Functor to print errors
  struct PrintErrors {
    /// Equivalently sized integer type
    typedef typename GemmTestbedTraits<CType>::integer_type integer_t;

    /// Output stream to write to
    std::ostream& out;

    /// Reference tensor view
    cutlass::HostTensorView<CType> const& reference;

    /// Computed tensor view
    cutlass::HostTensorView<CType> const& experimental;

    /// Errors greater than or this amount result in printing
    integer_t ulps_threshold;

    ///
    PrintErrors(std::ostream& _out,
                cutlass::HostTensorView<CType> const& _reference,
                cutlass::HostTensorView<CType> const& _experimental,
                integer_t _ulps_threshold = 1)
        : out(_out),
          reference(_reference),
          experimental(_experimental),
          ulps_threshold(_ulps_threshold) {}

    /// Compares one element
    void operator()(CType const& element, typename HostTensorC::Coord_t coord) {
      CType exp = experimental.at(coord);
      CType ref = reference.at(coord);

      int64_t int_exp = 0;
      int64_t int_ref = 0;

      *reinterpret_cast<CType*>(&int_exp) = exp;
      *reinterpret_cast<CType*>(&int_ref) = ref;

      integer_t ulps = integer_t(int_exp - int_ref);

      if (std::abs(ulps) >= ulps_threshold) {
        // width in hexadecimal digits of value
        int const width = sizeof(integer_t) * 2;

        double relative = double(exp) - double(ref);
        if (ref != CType(0)) {
          relative /= double(ref);
        }

        out << "[" << coord << "] expected: " << GemmTestbedTraits<CType>::to_print(ref) << " (0x"
            << std::hex << std::setw(width) << std::setfill('0') << integer_t(int_ref) << std::dec
            << ")"
            << ",  got: " << GemmTestbedTraits<CType>::to_print(exp) << " (0x" << std::hex
            << std::setw(width) << std::setfill('0') << integer_t(int_exp) << std::dec << ")"
            << "  relative error: " << relative << ", ulps: " << ulps << "\n";
      }
    }
  };

  /// Generates random elements
  template <typename T>
  struct RandomGenerator {
    RandomGenerator(int seed = -1, bool only_ones_ = false) : only_ones(only_ones_) { srand(seed); }

    T operator()() {
      if (only_ones) {
        return T(1);
      } else {
        int val = (rand() % 16) - 8;
        return T(val);
      }
    }

    bool only_ones;
  };

  //
  // Data members
  //

  /// A matrix operand
  HostTensorA A;

  /// B matrix operand
  HostTensorB B;

  /// C matrix operand
  HostTensorC C_initial;

  /// Reference result computed on the host
  cutlass::HostTensor<CType, false> ref_host;

  /// Computed result
  HostTensorC computed;

  /// Linear scalaring factor
  Scalar alpha;

  /// Linear scaling factor
  Scalar beta;

  /// Helper to resize a matrix with a given size and layout
  template <typename T, bool DeviceBacked>
  static void resize(cutlass::HostTensor<T, DeviceBacked>& tensor,
                     int rows,
                     int columns,
                     int layout,
                     int ldm = 0) {
    if (!ldm) {
      ldm = (layout ? rows : columns);
    }

    typedef cutlass::Coord<cutlass::HostTensor<T>::Rank> Coord_t;
    size_t matrix_stride = layout ? columns * ldm : rows * ldm;
    // TODO: Remove that (int) cast.
    Coord_t stride = cutlass::make_Coord(
        (int)matrix_stride, layout ? 1 : ldm, layout? ldm : 1, 1);
    Coord_t size = cutlass::make_Coord(1, rows, columns, 1);
    tensor.reset(stride, size);
  }

  //
  // Methods
  //

  /// Constructs a workspace for verifying GEMM.
  GemmTestbed(int M_,
              int N_,
              int K_,
              int lda,
              int ldb,
              int ldc,
              int layout_a,
              int layout_b,
              Scalar alpha_ = Scalar(1),
              Scalar beta_ = Scalar(0))
      :  alpha(alpha_), beta(beta_) {
    resize(A, M_, K_,layout_a,  lda);
    resize(B, K_, N_,layout_b,  ldb);
    resize(C_initial, M_, N_,1,  ldc);
    resize(ref_host, M_, N_,1,  ldc);
    resize(computed, M_, N_,1,  ldc);
  }

  /// Returns a pointer to the A operand
  typename HostTensorA::DeviceType* ptr_A() const {
	 return A.device_data();
  }

  /// Stride of A matrix
  int lda() const { 
	return std::max(A.stride(HostTensorA::Dim_H), A.stride(HostTensorA::Dim_W)); 
  }

  /// Returns a pointer to the B operand
  typename HostTensorB::DeviceType* ptr_B() const {
    return B.device_data();
  }

  /// Stride of B matrix
  int ldb() const {
	 return std::max(B.stride(HostTensorB::Dim_H), B.stride(HostTensorB::Dim_W));
  }

  /// Returns a pointer to the initial state of the result tensor in device memory
  typename HostTensorC::DeviceType* ptr_C_initial() const {
	 return C_initial.device_data();
  }

  /// Returns a pointer to the result tensor in device memory
  typename HostTensorC::DeviceType* ptr_computed() const {
	 return computed.device_data();
  }

  /// Stride of C matrix
  int ldc() const {
    return std::max(C_initial.stride(HostTensorC::Dim_H), C_initial.stride(HostTensorC::Dim_W));
  }

  /// Returns the number of flops implied by the computation (1 multiply-accumulate = 2 flops)
  uint64_t flops() const { return uint64_t(M()) * uint64_t(N()) * uint64_t(K()) * 2ULL; }

  /// Computes the speed of the computation in GFLOPs/s
  double GFLOPs_per_sec(double runtime_ms) const { return double(flops()) / runtime_ms / 1.0e6; }

  /// Number of rows of problem
  int M() const {
	 return C_initial.size(HostTensorC::Dim_H);
  }

  /// Number of columns of problem
  int N() const { 
	return C_initial.size(HostTensorC::Dim_W);
  }

  /// Number of columns of problem
  int K() const { 
	return A.size(HostTensorA::Dim_W); 
  }

  /// Initializes data, randomly
  void initialize(int seed = -1) {
    A.fill_random(RandomGenerator<AType>(seed));
    B.fill_random(RandomGenerator<BType>(seed + 11));
    C_initial.fill_random(RandomGenerator<CType>(seed + 13,1));
  }

  /// Computes the matrix product on the host
  void compute_host() {
    ref_host.fill(C_initial); 
    ref_host.template gemm<AType, BType, Accumulator, Scalar>(A, B, alpha, beta);
  }

  /// Names a probelm based on data type and problem size
  std::string workspace_name() const {
    std::stringstream ss;
    ss << "gemm_" << "t"
       << "t" << "_" << typeid(AType).name() << "_"
       << typeid(BType).name() << "_" << typeid(CType).name() << "_" << typeid(Accumulator).name()
       << "_" << typeid(Scalar).name() << "_" << M() << "x" << N() << "x" << K();

    return ss.str();
  }

  /// Writes the workspace to an ostream
  std::ostream& write(std::ostream& out) const {
    out << "A = " << A << "\nB = " << B << "\nC_initial = " << C_initial
        << "\ncomputed = " << computed 
        << "\nref_host= " << ref_host<< std::endl;

    return out;
  }

  /// Outputs each mismatching element
  std::ostream& write_errors(std::ostream& out,
                             cutlass::HostTensorView<CType> const& experimental,
                             cutlass::HostTensorView<CType> const& ref) const {
    PrintErrors printer(out, ref, experimental);

    computed.visit(printer);

    return out;
  }

  /// Sync's all input tensors to device
  void sync_device() {
    A.sync_device();
    B.sync_device();
    C_initial.sync_device();

    ref_host.fill(C_initial);
    computed.fill(C_initial);

    computed.sync_device();
  }

  /// Sync's all output tensors to host
  void sync_host() {
    computed.sync_host();
  }

  /// Saves the workspace to files
  void save_workspace(cutlass::HostTensorView<CType> const& experimental,
                      cutlass::HostTensorView<CType> const& ref) {
    std::string name = workspace_name();

    std::string results_name = name + "_results.txt";
    std::string errors_name = name + "_errors.txt";

    std::ofstream results(results_name.c_str());
    std::ofstream errors(errors_name.c_str());

    write(results);
    write_errors(errors, experimental, ref);
  }

  /// Verifies the contents of C equal the host-side reference
  bool verify_with_host(bool save_on_error = true, bool always_print = false) {
    compute_host();
    computed.sync_host();

    bool passed = computed.bit_equals(ref_host);

    if ((!passed && save_on_error) || always_print) {
      save_workspace(computed, ref_host);
    }
    return passed;
  }
};

}  // namespace test

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
inline  int convert(cutlass::MatrixLayout::Kind layout) {
  switch (layout) {
    case cutlass::MatrixLayout::kRowMajor:
      return 0;//CUBLAS_OP_T;
    case cutlass::MatrixLayout::kColumnMajor:
      return 1;//CUBLAS_OP_N;
    default:
      break;
  }
  return 1;//CUBLAS_OP_N;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
}
