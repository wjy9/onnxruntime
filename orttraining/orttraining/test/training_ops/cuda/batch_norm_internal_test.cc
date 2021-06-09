// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/tensor.h"
#include "core/session/inference_session.h"
#include "test/providers/provider_test_utils.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

using namespace std;

namespace onnxruntime {
namespace contrib {
namespace test {

using namespace onnxruntime::test;

TEST(BatchNormInternalTest, CudaForwardTraining) {
  OpTester test("BatchNormInternal", 1, kMSDomain);
  float epsilon = 1e-05f;
  float momentum = 0.1f;
  test.AddAttribute("epsilon", epsilon);
  test.AddAttribute("momentum", momentum);
  std::vector<int64_t> input_output_dims{2, 2, 2, 2};
  std::vector<int64_t> channel_dims{2};
  test.AddInput<float>("X", input_output_dims, {-0.2953f, 0.1180f, 1.0973f, -0.1931f, -0.1999f, -0.0237f, 1.5181f, 0.0076f, -1.0830f, -1.5433f, 0.4327f, -0.9813f, 0.7875f, -0.4080f, -2.3144f, 1.5493f});
  test.AddInput<float>("scale", channel_dims, {1.0f, 1.0f});
  test.AddInput<float>("B", channel_dims, {0.0f, 0.0f});
  test.AddInput<float>("mean", channel_dims, {1.0f, 2.0f});
  test.AddInput<float>("var", channel_dims, {1.0f, 2.0f});

  test.AddOutput<float>("Y", input_output_dims, {0.0131f, 0.5210f, 1.7244f, 0.1387f, -0.2708f, -0.1191f, 1.2089f, -0.0922f, -0.9548f, -1.5203f, 0.9077f, -0.8298f, 0.5796f, -0.4501f, -2.0921f, 1.2358f});

  test.AddOutput<float>("running_mean", channel_dims, {0.8694f, 1.8115f});
  test.AddOutput<float>("running_var", channel_dims, {0.9757f, 1.9541f});
  test.AddOutput<float>("saved_mean", channel_dims, {-0.306f, 0.114562f});
  test.AddOutput<float>("saved_inv_std", channel_dims, {1.2288f, 0.861317f});

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kCpuExecutionProvider, kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(BatchNormInternalTest, CudaForwardTrainingDouble) {
  OpTester test("BatchNormInternal", 1, kMSDomain);
  float epsilon = 1e-05f;
  float momentum = 0.1f;
  test.AddAttribute("epsilon", epsilon);
  test.AddAttribute("momentum", momentum);
  std::vector<int64_t> input_output_dims{2, 2, 2, 2};
  std::vector<int64_t> channel_dims{2};
  test.AddInput<double>("X", input_output_dims, {-0.2953f, 0.1180f, 1.0973f, -0.1931f, -0.1999f, -0.0237f, 1.5181f, 0.0076f, -1.0830f, -1.5433f, 0.4327f, -0.9813f, 0.7875f, -0.4080f, -2.3144f, 1.5493f});
  test.AddInput<double>("scale", channel_dims, {1.0f, 1.0f});
  test.AddInput<double>("B", channel_dims, {0.0f, 0.0f});
  test.AddInput<double>("mean", channel_dims, {1.0f, 2.0f});
  test.AddInput<double>("var", channel_dims, {1.0f, 2.0f});

  test.AddOutput<double>("Y", input_output_dims, {0.0131f, 0.5210f, 1.7244f, 0.1387f, -0.2708f, -0.1191f, 1.2089f, -0.0922f, -0.9548f, -1.5203f, 0.9077f, -0.8298f, 0.5796f, -0.4501f, -2.0921f, 1.2358f});

  test.AddOutput<double>("running_mean", channel_dims, {0.8694f, 1.8115f});
  test.AddOutput<double>("running_var", channel_dims, {0.9757f, 1.9541f});
  test.AddOutput<double>("saved_mean", channel_dims, {-0.306f, 0.114562f});
  test.AddOutput<double>("saved_inv_std", channel_dims, {1.2288f, 0.861317f});

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kCpuExecutionProvider, kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(BatchNormInternalTest, CudaForwardTrainingHalf) {
  OpTester test("BatchNormInternal", 1, kMSDomain);
  float epsilon = 1e-05f;
  float momentum = 0.1f;
  test.AddAttribute("epsilon", epsilon);
  test.AddAttribute("momentum", momentum);
  std::vector<int64_t> input_output_dims{2, 2, 2, 2};
  std::vector<int64_t> channel_dims{2};
  std::vector<float> X = {-0.2953f, 0.1180f, 1.0973f, -0.1931f, -0.1999f, -0.0237f, 1.5181f, 0.0076f, -1.0830f, -1.5433f, 0.4327f, -0.9813f, 0.7875f, -0.4080f, -2.3144f, 1.5493f};
  std::vector<float> scale = {1.0f, 1.0f};
  std::vector<float> B = {0.0f, 0.0f};
  std::vector<float> mean = {1.0f, 2.0f};
  std::vector<float> var = {1.0f, 2.0f};

  std::vector<MLFloat16> X_half(16);
  std::vector<MLFloat16> scale_half(2);
  std::vector<MLFloat16> B_half(2);
  std::vector<MLFloat16> mean_half(2);
  std::vector<MLFloat16> var_half(2);

  ConvertFloatToMLFloat16(X.data(), X_half.data(), 16);
  ConvertFloatToMLFloat16(scale.data(), scale_half.data(), 2);
  ConvertFloatToMLFloat16(B.data(), B_half.data(), 2);
  ConvertFloatToMLFloat16(mean.data(), mean_half.data(), 2);
  ConvertFloatToMLFloat16(var.data(), var_half.data(), 2);

  test.AddInput<MLFloat16>("X", input_output_dims, X_half);
  test.AddInput<MLFloat16>("scale", channel_dims, scale_half);
  test.AddInput<MLFloat16>("B", channel_dims, B_half);
  test.AddInput<MLFloat16>("mean", channel_dims, mean_half);
  test.AddInput<MLFloat16>("var", channel_dims, var_half);

  std::vector<float> Y = {0.0131f, 0.5210f, 1.7244f, 0.1387f, -0.2708f, -0.1191f, 1.2089f, -0.0922f, -0.9548f, -1.5203f, 0.9077f, -0.8298f, 0.5796f, -0.4501f, -2.0921f, 1.2358f};
  std::vector<float> running_mean = {0.8694f, 1.8115f};
  std::vector<float> running_var = {0.9757f, 1.9541f};
  std::vector<float> saved_mean = {-0.306f, 0.114562f};
  std::vector<float> saved_inv_std = {1.2288f, 0.861317f};

  std::vector<MLFloat16> Y_half(16);
  std::vector<MLFloat16> running_mean_half(2);
  std::vector<MLFloat16> running_var_half(2);
  std::vector<MLFloat16> saved_mean_half(2);
  std::vector<MLFloat16> saved_inv_std_half(2);

  ConvertFloatToMLFloat16(Y.data(), Y_half.data(), 16);
  ConvertFloatToMLFloat16(running_mean.data(), running_mean_half.data(), 2);
  ConvertFloatToMLFloat16(running_var.data(), running_var_half.data(), 2);
  ConvertFloatToMLFloat16(saved_mean.data(), saved_mean_half.data(), 2);
  ConvertFloatToMLFloat16(saved_inv_std.data(), saved_inv_std_half.data(), 2);

  test.AddOutput<MLFloat16>("Y", input_output_dims, Y_half);
  test.AddOutput<MLFloat16>("running_mean", channel_dims, running_mean_half);
  test.AddOutput<MLFloat16>("running_var", channel_dims, running_var_half);
  test.AddOutput<MLFloat16>("saved_mean", channel_dims, saved_mean_half);
  test.AddOutput<MLFloat16>("saved_inv_std", channel_dims, saved_inv_std_half);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kCpuExecutionProvider, kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(BatchNormInternalTest, CudaForwardTrainingHalfAndFloat) {
  OpTester test("BatchNormInternal", 1, kMSDomain);
  float epsilon = 1e-05f;
  float momentum = 0.1f;
  test.AddAttribute("epsilon", epsilon);
  test.AddAttribute("momentum", momentum);
  std::vector<int64_t> input_output_dims{2, 2, 2, 2};
  std::vector<int64_t> channel_dims{2};
  std::vector<float> X = {-0.2953f, 0.1180f, 1.0973f, -0.1931f, -0.1999f, -0.0237f, 1.5181f, 0.0076f, -1.0830f, -1.5433f, 0.4327f, -0.9813f, 0.7875f, -0.4080f, -2.3144f, 1.5493f};
  std::vector<float> scale = {1.0f, 1.0f};
  std::vector<float> B = {0.0f, 0.0f};
  std::vector<float> mean = {1.0f, 2.0f};
  std::vector<float> var = {1.0f, 2.0f};

  std::vector<MLFloat16> X_half(16);
  std::vector<MLFloat16> scale_half(2);
  std::vector<MLFloat16> B_half(2);
  std::vector<MLFloat16> mean_half(2);
  std::vector<MLFloat16> var_half(2);

  ConvertFloatToMLFloat16(X.data(), X_half.data(), 16);
  ConvertFloatToMLFloat16(scale.data(), scale_half.data(), 2);
  ConvertFloatToMLFloat16(B.data(), B_half.data(), 2);

  test.AddInput<MLFloat16>("X", input_output_dims, X_half);
  test.AddInput<MLFloat16>("scale", channel_dims, scale_half);
  test.AddInput<MLFloat16>("B", channel_dims, B_half);
  test.AddInput<float>("mean", channel_dims, mean);
  test.AddInput<float>("var", channel_dims, var);

  std::vector<float> Y = {0.0131f, 0.5210f, 1.7244f, 0.1387f, -0.2708f, -0.1191f, 1.2089f, -0.0922f, -0.9548f, -1.5203f, 0.9077f, -0.8298f, 0.5796f, -0.4501f, -2.0921f, 1.2358f};
  std::vector<float> running_mean = {0.8694f, 1.8115f};
  std::vector<float> running_var = {0.9757f, 1.9541f};
  std::vector<float> saved_mean = {-0.306f, 0.114562f};
  std::vector<float> saved_inv_std = {1.2288f, 0.861317f};

  std::vector<MLFloat16> Y_half(16);

  ConvertFloatToMLFloat16(Y.data(), Y_half.data(), 16);

  test.AddOutput<MLFloat16>("Y", input_output_dims, Y_half);
  test.AddOutput<float>("running_mean", channel_dims, running_mean);
  test.AddOutput<float>("running_var", channel_dims, running_var);
  test.AddOutput<float>("saved_mean", channel_dims, saved_mean);
  test.AddOutput<float>("saved_inv_std", channel_dims, saved_inv_std);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kCpuExecutionProvider, kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

}  // namespace test
} // namespace contrib
}  // namespace onnxruntime
