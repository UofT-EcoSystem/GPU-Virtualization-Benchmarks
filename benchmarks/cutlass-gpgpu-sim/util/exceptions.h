/******************************************************************************
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are not permitted.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

/**
 * \file
 * \brief C++ exception semantics for CUDA error codes
 */

#include <cuda_runtime.h>
#include <iosfwd>
#include <stdexcept>

#include <cutlass/util/platform.h>

namespace cutlass {

/// C++ exception wrapper for CUDA \p cudaError_t
class cuda_exception : public std::exception {
 public:
  /// Constructor
  cuda_exception(const char* msg = "", cudaError_t err = cudaErrorUnknown) : msg(msg), err(err) {}

  /// Returns the underlying CUDA \p cudaError_t
  cudaError_t cudaError() const { return err; }

 protected:
  /// Explanatory string
  const char* msg;

  /// Underlying CUDA \p cudaError_t
  cudaError_t err;
};

/// Writes a cudaError_t to an output stream
inline std::ostream& operator<<(std::ostream& out, cudaError_t result) {
  return out << cudaGetErrorString(result);
}

/// Writes a cuda_exception instance to an output stream
inline std::ostream& operator<<(std::ostream& out, cuda_exception const& e) {
  return out << e.what() << ": " << e.cudaError();
}

}  // namespace cutlass
