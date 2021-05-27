// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/nn/batch_norm_internal.h"
#include "core/providers/common.h"
#include "core/providers/cuda/cudnn_common.h"
#include "core/providers/cpu/nn/batch_norm_helper.h"
#include "core/providers/cuda/math/unary_elementwise_ops_impl.h"

using namespace std;
namespace onnxruntime {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T, U)                               \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      BatchNormInternal,                                          \
      kMSDomain,                                                  \
      1,                                                          \
      T##_##U,                                                    \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .Alias(3, 1)                                            \
          .Alias(4, 2)                                            \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())  \
          .TypeConstraint("U", DataTypeImpl::GetTensorType<U>()), \
      BatchNormInternal<T, U>);

template <typename T, typename U>
Status BatchNormInternal<T, U>::ComputeInternal(OpKernelContext* p_op_kernel_context) const {
  std::cout << "Entering BNI ...\n";
  typedef typename ToCudaType<T>::MappedType CudaT;
  typedef typename ToCudaType<U>::MappedType CudaU;

  const Tensor* X = p_op_kernel_context->Input<Tensor>(0);
  const Tensor* scale = p_op_kernel_context->Input<Tensor>(1);
  const Tensor* B = p_op_kernel_context->Input<Tensor>(2);
  const Tensor* mean = p_op_kernel_context->Input<Tensor>(3);
  const Tensor* var = p_op_kernel_context->Input<Tensor>(4);

  ORT_RETURN_IF_ERROR(BatchNormHelper::ValidateInputs(X, scale, B, mean, var, spatial_ == 1));

  const TensorShape& x_shape = X->Shape();
  const TensorShape& channel_shape = mean->Shape();

  Tensor* Y = p_op_kernel_context->Output(0, x_shape);
  Tensor* running_mean = p_op_kernel_context->Output(1, channel_shape);
  Tensor* running_var = p_op_kernel_context->Output(2, channel_shape);
  Tensor* saved_mean = p_op_kernel_context->Output(3, channel_shape);
  Tensor* saved_inv_var = p_op_kernel_context->Output(4, channel_shape);

  auto x_data = reinterpret_cast<const CudaT*>(X->template Data<T>());
  auto scale_data = reinterpret_cast<const CudaT*>(scale->template Data<T>());
  auto b_data = reinterpret_cast<const CudaT*>(B->template Data<T>());
  auto mean_data = reinterpret_cast<const CudaU*>(mean->template Data<U>());
  auto var_data = reinterpret_cast<const CudaU*>(var->template Data<U>());

  auto y_data = reinterpret_cast<CudaT*>(Y->template MutableData<T>());

  const auto alpha = Consts<CudaT>::One;
  const auto beta = Consts<CudaT>::Zero;

  CudnnTensor data_desc, bn_tensor_desc;
  vector<int64_t> new_dims;
  BatchNormHelper::NormalizeDims(x_shape, new_dims);
  ORT_RETURN_IF_ERROR(data_desc.Set(new_dims, CudnnTensor::GetDataType<CudaT>()));

  auto running_mean_data = reinterpret_cast<CudaU*>(running_mean->template MutableData<U>());
  auto running_var_data = reinterpret_cast<CudaU*>(running_var->template MutableData<U>());
  auto saved_mean_data = reinterpret_cast<CudaU*>(saved_mean->template MutableData<U>());
  auto saved_inv_var_data = reinterpret_cast<CudaU*>(saved_inv_var->template MutableData<U>());

  const int64_t C = x_shape.GetDims()[1];
  auto f_mean = GetScratchBuffer<float>(C);
  auto f_var = GetScratchBuffer<float>(C);
  Impl_Cast<CudaU, float>(Stream(), mean_data, f_mean.get(), C);
  Impl_Cast<CudaU, float>(Stream(), var_data, f_var.get(), C);

  if (X->IsDataType<MLFloat16>()) {
    CudnnTensor scale_desc;
    ORT_RETURN_IF_ERROR(scale_desc.Set(new_dims, CudnnTensor::GetDataType<float>()));
    ORT_RETURN_IF_ERROR(bn_tensor_desc.Set(scale_desc, cudnn_batch_norm_mode_));

    // Convert scale/B to float
    auto f_scale = GetScratchBuffer<float>(C);
    auto f_B = GetScratchBuffer<float>(C);

    Impl_Cast<CudaT, float>(Stream(), scale_data, f_scale.get(), C);
    Impl_Cast<CudaT, float>(Stream(), b_data, f_B.get(), C);

    if (mean->IsDataType<MLFloat16>()) {
      // Convert saved mean/inv_var to float
      auto f_saved_mean = GetScratchBuffer<float>(C);
      auto f_saved_inv_var = GetScratchBuffer<float>(C);

      CUDNN_RETURN_IF_ERROR(cudnnBatchNormalizationForwardTraining(
          CudnnHandle(),
          cudnn_batch_norm_mode_,
          &alpha,
          &beta,
          data_desc,
          x_data,
          data_desc,
          y_data,
          bn_tensor_desc,
          f_scale.get(),
          f_B.get(),
          momentum_,
          f_mean.get(),
          f_var.get(),
          epsilon_,
          f_saved_mean.get(),
          f_saved_inv_var.get()));

      Impl_Cast<float, CudaU>(Stream(), f_saved_mean.get(), saved_mean_data, C);
      Impl_Cast<float, CudaU>(Stream(), f_saved_inv_var.get(), saved_inv_var_data, C);
    } else {
      CUDNN_RETURN_IF_ERROR(cudnnBatchNormalizationForwardTraining(
          CudnnHandle(),
          cudnn_batch_norm_mode_,
          &alpha,
          &beta,
          data_desc,
          x_data,
          data_desc,
          y_data,
          bn_tensor_desc,
          f_scale.get(),
          f_B.get(),
          momentum_,
          f_mean.get(),
          f_var.get(),
          epsilon_,
          saved_mean_data,
          saved_inv_var_data));
    }
  } else {
    ORT_RETURN_IF_ERROR(bn_tensor_desc.Set(data_desc, cudnn_batch_norm_mode_));

    CUDNN_RETURN_IF_ERROR(cudnnBatchNormalizationForwardTraining(
        CudnnHandle(),
        cudnn_batch_norm_mode_,
        &alpha,
        &beta,
        data_desc,
        x_data,
        data_desc,
        y_data,
        bn_tensor_desc,
        scale_data,
        b_data,
        momentum_,
        f_mean.get(),
        f_var.get(),
        epsilon_,
        saved_mean_data,
        saved_inv_var_data));
  }

  Impl_Cast<float, CudaU>(Stream(), f_mean.get(), running_mean_data, C);
  Impl_Cast<float, CudaU>(Stream(), f_var.get(), running_var_data, C);

  return Status::OK();
}

#define SPECIALIZED_COMPUTE(T, U) \
  REGISTER_KERNEL_TYPED(T, U)     \
  template Status BatchNormInternal<T, U>::ComputeInternal(OpKernelContext* ctx) const;

SPECIALIZED_COMPUTE(float, float)
SPECIALIZED_COMPUTE(double, double)
SPECIALIZED_COMPUTE(MLFloat16, MLFloat16)
SPECIALIZED_COMPUTE(MLFloat16, float)

}  // namespace cuda
}  // namespace onnxruntime
