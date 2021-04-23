// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/external_functions/batch_norm_function.h"
#include "core/external_functions/external_function_registry.h"
#include "core/external_functions/attributes_json_parser.h"
#include "core/dlpack/dlpack_converter.h"
#include "core/framework/op_kernel_context_internal.h"

namespace onnxruntime {
namespace external_functions {

SPECIALIZED_EXTERNAL_FUNCTION_KERNEL_CREATOR(ATenCudnnBatchNormFunction)
SPECIALIZED_EXTERNAL_FUNCTION_KERNEL_CREATOR(ATenCudnnBatchNormBackwardFunction)

ATenCudnnBatchNormFunction::ATenCudnnBatchNormFunction(const OpKernelInfo& info, void* p_fn_raw) : OpKernel(info) {
  p_fn_ = reinterpret_cast<ATenCudnnBatchNorm>(p_fn_raw);
  std::string custom_attributes_json = info.GetAttrOrDefault<std::string>("custom_attributes_json", "{}");
  AttributesJsonParser parser(custom_attributes_json);
  momentum_ = parser.GetAttributeOrDefault<float>("momentum", .1f);
  eps_ = parser.GetAttributeOrDefault<float>("eps", 1e-5f);
}

Status ATenCudnnBatchNormFunction::Compute(OpKernelContext* p_ctx) const {
  auto* p_ctx_internal = static_cast<OpKernelContextInternal*>(p_ctx);
  OrtValue input = *p_ctx_internal->GetInputMLValue(0);
  OrtValue weight = *p_ctx_internal->GetInputMLValue(1);
  OrtValue bias = *p_ctx_internal->GetInputMLValue(2);
  OrtValue running_mean = *p_ctx_internal->GetInputMLValue(3);
  OrtValue running_var = *p_ctx_internal->GetInputMLValue(4);
  auto torch_result =
      p_fn_(dlpack::OrtValueToDlpack(input), dlpack::OrtValueToDlpack(weight), dlpack::OrtValueToDlpack(bias),
            dlpack::OrtValueToDlpack(running_mean), dlpack::OrtValueToDlpack(running_var), momentum_, eps_);
  ORT_RETURN_IF_ERROR(p_ctx_internal->SetOutputMLValue(0, dlpack::DlpackToOrtValue(std::get<0>(torch_result))));
  ORT_RETURN_IF_ERROR(p_ctx_internal->SetOutputMLValue(1, dlpack::DlpackToOrtValue(std::get<1>(torch_result))));
  ORT_RETURN_IF_ERROR(p_ctx_internal->SetOutputMLValue(2, dlpack::DlpackToOrtValue(std::get<2>(torch_result))));
  ORT_RETURN_IF_ERROR(p_ctx_internal->SetOutputMLValue(3, dlpack::DlpackToOrtValue(std::get<3>(torch_result))));
  return Status::OK();
}

ATenCudnnBatchNormBackwardFunction::ATenCudnnBatchNormBackwardFunction(const OpKernelInfo& info, void* p_fn_raw)
    : OpKernel(info) {
  p_fn_ = reinterpret_cast<ATenCudnnBatchNormBackward>(p_fn_raw);
  std::string custom_attributes_json = info.GetAttrOrDefault<std::string>("custom_attributes_json", "{}");
  AttributesJsonParser parser(custom_attributes_json);
  eps_ = parser.GetAttributeOrDefault<float>("eps", 1e-5f);
}

Status ATenCudnnBatchNormBackwardFunction::Compute(OpKernelContext* p_ctx) const {
  auto* p_ctx_internal = static_cast<OpKernelContextInternal*>(p_ctx);
  OrtValue grad = *p_ctx_internal->GetInputMLValue(0);
  OrtValue input = *p_ctx_internal->GetInputMLValue(1);
  OrtValue weight = *p_ctx_internal->GetInputMLValue(2);
  OrtValue running_mean = *p_ctx_internal->GetInputMLValue(3);
  OrtValue running_var = *p_ctx_internal->GetInputMLValue(4);
  OrtValue save_mean = *p_ctx_internal->GetInputMLValue(5);
  OrtValue save_var = *p_ctx_internal->GetInputMLValue(6);
  OrtValue reserve_space = *p_ctx_internal->GetInputMLValue(7);
  auto torch_result = p_fn_(dlpack::OrtValueToDlpack(grad), dlpack::OrtValueToDlpack(input),
                            dlpack::OrtValueToDlpack(weight), dlpack::OrtValueToDlpack(running_mean),
                            dlpack::OrtValueToDlpack(running_var), dlpack::OrtValueToDlpack(save_mean),
                            dlpack::OrtValueToDlpack(save_var), dlpack::OrtValueToDlpack(reserve_space), eps_);
  ORT_RETURN_IF_ERROR(p_ctx_internal->SetOutputMLValue(0, dlpack::DlpackToOrtValue(std::get<0>(torch_result))));
  ORT_RETURN_IF_ERROR(p_ctx_internal->SetOutputMLValue(1, dlpack::DlpackToOrtValue(std::get<1>(torch_result))));
  ORT_RETURN_IF_ERROR(p_ctx_internal->SetOutputMLValue(2, dlpack::DlpackToOrtValue(std::get<2>(torch_result))));
  return Status::OK();
}

}  // namespace external_functions
}  // namespace onnxruntime
