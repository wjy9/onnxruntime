// #include "core/providers/shared_library/provider_api.h"
#include "lstm.h"
#include "core/providers/dnnl/dnnl_fwd.h"

// #include "dnnl.hpp"

namespace onnxruntime {
namespace ort_dnnl {

/* LSTM operator */
ONNX_OPERATOR_KERNEL_EX(LSTM,
                        kOnnxDomain,
                        7,
                        kDnnlExecutionProvider,
                        KernelDefBuilder()
                            .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
                            .TypeConstraint("T1", DataTypeImpl::GetTensorType<int32_t>()),
                        LSTM);

using namespace rnn::detail;

// LSTM details

Status LSTM::TryPackWeights(const Tensor& weights, PackedWeights& packed_weights, bool& is_packed) {
  const auto& shape = weights.Shape();
  if (shape.NumDimensions() != 3) {
    return Status::OK();
  }

  // weights: [num_directions, 4*hidden_size, input_size]
  // recurrence weights: [num_directions, 4*hidden_size, hidden_size]
  const size_t N = static_cast<size_t>(shape[1]);
  const size_t K = static_cast<size_t>(shape[2]);

  if ((shape[0] != num_directions_) || (N != static_cast<size_t>(hidden_size_ * 4))) {
    return Status::OK();
  }

  const size_t packed_weights_size = MlasGemmPackBSize(N, K);
  if (packed_weights_size == 0) {
    return Status::OK();
  }

  auto alloc = Info().GetAllocator(0, OrtMemTypeDefault);
  auto* packed_weights_data = alloc->Alloc(SafeInt<size_t>(packed_weights_size) * num_directions_);
  packed_weights.buffer_ = BufferUniquePtr(packed_weights_data, BufferDeleter(alloc));
  packed_weights.weights_size_ = packed_weights_size;
  packed_weights.shape_ = shape;

  const auto* weights_data = weights.Data<float>();
  for (int i = 0; i < num_directions_; i++) {
    MlasGemmPackB(CblasTrans, N, K, weights_data, K, packed_weights_data);
    packed_weights_data = static_cast<uint8_t*>(packed_weights_data) + packed_weights_size;
    weights_data += N * K;
  }

  is_packed = true;
  return Status::OK();
}

Status LSTM::PrePack(const Tensor& tensor, int input_idx, bool& is_packed) {
  is_packed = false;

  if (tensor.IsDataType<float>()) {
    if (input_idx == 1) {
      return TryPackWeights(tensor, packed_W_, is_packed);
    } else if (input_idx == 2) {
      return TryPackWeights(tensor, packed_R_, is_packed);
    }
  }

  return Status::OK();
}

Status LSTM::Compute(OpKernelContext* context) const {
  const Tensor& X = *context->Input<Tensor>(0);  // inputs. [seq_length, batch_size, input_size]

  Status status;
  // auto& logger = context->Logger();

  if (X.IsDataType<float>()) {
    const Tensor* W = packed_W_.buffer_ ? nullptr : context->Input<Tensor>(1);
    // weights. [num_directions, 4*hidden_size, input_size]
    const Tensor* R = packed_R_.buffer_ ? nullptr : context->Input<Tensor>(2);
    // recurrence weights. [num_directions, 4*hidden_size, hidden_size]

    const auto& W_shape = (W != nullptr) ? W->Shape() : packed_W_.shape_;
    const auto& R_shape = (R != nullptr) ? R->Shape() : packed_R_.shape_;

    const auto* input_weights = (W != nullptr) ? W->Data<float>() : nullptr;
    const auto* recurrent_weights = (R != nullptr) ? R->Data<float>() : nullptr;

    // spans for first direction
    const size_t input_weights_size_per_direction = W_shape[1] * W_shape[2];
    const size_t hidden_weights_size_per_direction = R_shape[1] * R_shape[2];

    GemmWeights<float> W_1(0, input_weights, input_weights_size_per_direction, packed_W_);
    GemmWeights<float> R_1(0, recurrent_weights, hidden_weights_size_per_direction, packed_R_);

    GemmWeights<float> W_2;
    GemmWeights<float> R_2;
    if (direction_ == Direction::kBidirectional) {
      W_2.Init(1, input_weights, input_weights_size_per_direction, packed_W_, nullptr);
      R_2.Init(1, recurrent_weights, hidden_weights_size_per_direction, packed_R_, nullptr);
    }

    return LSTMBase::ComputeImpl<float, float>(*context, W_1, W_2, R_1, R_2);
  } else if (X.IsDataType<double>()) {
    /* Need to update all the helpers to support double...
    status = ComputeImpl<double>(*context); */
    ORT_NOT_IMPLEMENTED("LSTM operator does not support double yet");
  } else {
    ORT_THROW("Invalid data type for LSTM operator of ", X.DataType());
  }
}
}  // namespace ort_dnnl
}  // namespace onnxruntime
