// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <tuple>
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/dlpack/dlpack_converter.h"

namespace onnxruntime {
namespace external_functions {

typedef std::tuple<DLManagedTensor*, DLManagedTensor*, DLManagedTensor*, DLManagedTensor*> (*ATenCudnnBatchNorm)(
    const DLManagedTensor* input, const DLManagedTensor* weight, const DLManagedTensor* bias,
    const DLManagedTensor* running_mean, const DLManagedTensor* running_var, float momentum, float eps);
typedef std::tuple<DLManagedTensor*, DLManagedTensor*, DLManagedTensor*> (*ATenCudnnBatchNormBackward)(
    const DLManagedTensor* grad_output, const DLManagedTensor* input, const DLManagedTensor* weight,
    const DLManagedTensor* running_mean, const DLManagedTensor* running_var, const DLManagedTensor* save_mean,
    const DLManagedTensor* save_var, const DLManagedTensor* reserve_space, float eps);

class ATenCudnnBatchNormFunction : public OpKernel {
 public:
  ATenCudnnBatchNormFunction(const OpKernelInfo& info, void* p_fn_raw);
  Status Compute(OpKernelContext* p_ctx) const override;

 private:
  float momentum_;
  float eps_;

  ATenCudnnBatchNorm p_fn_;
};

class ATenCudnnBatchNormBackwardFunction : public OpKernel {
 public:
  ATenCudnnBatchNormBackwardFunction(const OpKernelInfo& info, void* p_fn_raw);
  Status Compute(OpKernelContext* p_ctx) const override;

 private:
  float eps_;

  ATenCudnnBatchNormBackward p_fn_;
};

}  // namespace external_functions
}  // namespace onnxruntime
