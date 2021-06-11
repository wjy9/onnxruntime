#pragma once
#include "core/providers/cpu/rnn/lstm_base.h"
// #include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace ort_dnnl {
class LSTM final : public OpKernel, public LSTMBase {
 public:
  LSTM(const OpKernelInfo& info) : OpKernel(info), LSTMBase(info) {}
  Status PrePack(const Tensor& tensor, int input_idx, bool& is_packed) override;
  Status Compute(OpKernelContext* context) const override;

  ~LSTM() override = default;

 private:
  Status TryPackWeights(const Tensor& weights, rnn::detail::PackedWeights& packed_weights, bool& is_packed);

//   template <typename T>
//   Status ComputeImpl(OpKernelContext& context) const;

  rnn::detail::PackedWeights packed_W_;
  rnn::detail::PackedWeights packed_R_;
};
}  // namespace ort_dnnl
}  // namespace onnxruntime
