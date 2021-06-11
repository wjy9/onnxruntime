#pragma once
#include "core/providers/dnnl/dnnl_fwd.h"
#include "core/providers/dnnl/dnnl_common.h"
#include "core/providers/dnnl/subgraph/dnnl_kernel.h"

namespace onnxruntime {
namespace ort_dnnl {

template <typename T>
class DnnlLSTM : public DnnlKernel {
 public:
  explicit DnnlLSTM(const DnnlNode& node,
                    DNNLExecutionProvider* provider,
                    const NodeAttributes& attributes,
                    const std::string attributes_prefix = "") : DnnlKernel(node, provider) {
    ReadAttributes(attributes, attributes_prefix);
  }

  void CreatePrimitives(const OrtCustomOpApi* api,
                        OrtKernelContext* context,
                        const std::unordered_map<dnnl::engine::kind, dnnl::engine>& dnnl_engine,
                        std::vector<dnnl::primitive>& net,
                        std::vector<std::unordered_map<int, dnnl::memory>>& net_args) override {
    dnnl::engine cpu_engine;
    dnnl::engine engine_to_use;
    std::unordered_map<dnnl::engine::kind, dnnl::engine>::const_iterator iter = dnnl_engine.find(dnnl::engine::kind::cpu);
    if (iter != dnnl_engine.end()) {
      cpu_engine = (dnnl::engine)iter->second;
      engine_to_use = cpu_engine;
    }
    Ort::CustomOpApi ort{*api};
    int num_inputs = mklnode_ptr_->num_inputs;
    printf("num_inputs %d\n", num_inputs);
    int input_index = mklnode_ptr_->input_start_index < 0 ? 0 : mklnode_ptr_->input_start_index;

    if (!mklnode_ptr_->parent_nodes.empty()) {
      printf("error mklnode_ptr_->parent_nodes.empty() == false\n");
    }

    auto get_shape_from_input = [&ort, context, input_index](int index, TensorShape& shape, size_t& dim) {
      const OrtValue* input_tensor = ort.KernelContext_GetInput(context, input_index + index);
      auto tensor_info = ort.GetTensorTypeAndShape(input_tensor);
      auto tensor_shape = ort.GetTensorShape(tensor_info);
      ort.ReleaseTensorTypeAndShapeInfo(tensor_info);
      auto xshape = tensor_shape.data();
      dim = tensor_shape.size();
      shape = TensorShape(xshape, dim);
    };
    TensorShape x_shape, y_shape;
    size_t x_dim;
    get_shape_from_input(0, x_shape, x_dim);
    y_shape = TensorShape(x_shape);

    y_shape[x_dim - 1] = hidden_size_;
    for (size_t i = 0; i < x_dim; i++) {
      printf("shape: x: %ld y: %ld\n", x_shape[i], y_shape[i]);
    }
    dnnl::memory::dims src_dims(
        x_shape.GetDims().begin(), x_shape.GetDims().end());
    src_md_ = std::make_unique<dnnl::memory::desc>(
        src_dims, DnnnType<T>(), dnnl::memory::format_tag::tnc);

    dnnl::memory::dims dst_dims(
        y_shape.GetDims().begin(), y_shape.GetDims().end());

    dst_md_ = std::make_unique<dnnl::memory::desc>(dst_dims, DnnnType<T>(), dnnl::memory::format_tag::tnc);

    TensorShape weight_shape;
    size_t weight_dim;
    get_shape_from_input(1, weight_shape, weight_dim);
    TensorShape iter_shape;
    size_t iter_dim;
    get_shape_from_input(2, iter_shape, iter_dim);
    TensorShape bias_shape;
    size_t bias_dim;
    get_shape_from_input(3, bias_shape, bias_dim);

    // for (size_t i = 0; i < weight_dim; i++) {
    //   printf("w: %ld\n", weight_shape[i]);
    // }
    // for (size_t i = 0; i < iter_dim; i++) {
    //   printf("i: %ld\n", iter_shape[i]);
    // }
    bias_shape[bias_dim - 1] = bias_shape[bias_dim - 1] / 2;
    // for (size_t i = 0; i < bias_dim; i++) {
    //   printf("b: %ld\n", bias_shape[i]);
    // }

    dnnl::memory::dims weight_input_dims({1, 1, 4, weight_shape[1] / 4, weight_shape[2]});
    dnnl::memory::dims iter_input_dims({1, 1, 4, iter_shape[1] / 4, iter_shape[2]});
    dnnl::memory::dims bias_dims({1, 1, 4, bias_shape[1] / 4});

    dnnl::memory::dims weight_dims({1, 1, weight_shape[2], 4, weight_shape[1] / 4});
    dnnl::memory::dims iter_dims({1, 1, iter_shape[2], 4, iter_shape[1] / 4});
    dnnl::memory::desc weight_md = dnnl::memory::desc(weight_dims, DnnnType<T>(), dnnl::memory::format_tag::any);
    dnnl::memory::desc iter_md = dnnl::memory::desc(iter_dims, DnnnType<T>(), dnnl::memory::format_tag::any);

    weight_input_md_ = std::make_unique<dnnl::memory::desc>(weight_dims, DnnnType<T>(), dnnl::memory::format_tag::ldgoi);
    iter_input_md_ = std::make_unique<dnnl::memory::desc>(iter_dims, DnnnType<T>(), dnnl::memory::format_tag::ldgoi);
    bias_md_ = std::make_unique<dnnl::memory::desc>(bias_dims, DnnnType<T>(), dnnl::memory::format_tag::ldgo);

    // dnnl::memory::desc src_iter_md({1, 1, 1, iter_shape[2]}, DnnnType<T>(), dnnl::memory::format_tag::ldnc);
    // dnnl::memory::desc src_iter_c_md({1, 1, 1, iter_shape[2]}, DnnnType<T>(), dnnl::memory::format_tag::ldnc);
    // dnnl::memory::desc dst_iter_c_md({1, 1, 1, iter_shape[2]}, DnnnType<T>(), dnnl::memory::format_tag::ldnc);
    dnnl::memory::desc src_iter_md = dnnl::memory::desc();
    dnnl::memory::desc src_iter_c_md = dnnl::memory::desc();
    dnnl::memory::dims dst_iter_dim({1, 1, 1, iter_shape[2]});
    dst_iter_md_ = std::make_unique<dnnl::memory::desc>(dst_iter_dim, DnnnType<T>(), dnnl::memory::format_tag::ldnc);
    dnnl::memory::desc dst_iter_c_md = dnnl::memory::desc(dst_iter_dim, DnnnType<T>(), dnnl::memory::format_tag::ldnc);
    lstm_desc_ = std::make_unique<dnnl::lstm_forward::desc>(dnnl::prop_kind::forward_inference,
                                                            dnnl::rnn_direction::unidirectional_left2right, *src_md_, src_iter_md,
                                                            src_iter_c_md, weight_md, iter_md, *bias_md_,
                                                            *dst_md_, *dst_iter_md_, dst_iter_c_md);

    lstm_pd_ = std::make_unique<dnnl::lstm_forward::primitive_desc>(*lstm_desc_, engine_to_use);

#define create_mem_without_alloc(name) name##_mem_ = std::make_unique<dnnl::memory>(*name##_md_, engine_to_use, nullptr)
    create_mem_without_alloc(src);
    create_mem_without_alloc(weight_input);
    create_mem_without_alloc(iter_input);
    create_mem_without_alloc(bias);
    create_mem_without_alloc(dst);
    create_mem_without_alloc(dst_iter);
#undef create_mem_without_alloc

    weight_mem_ = std::make_unique<dnnl::memory>(lstm_pd_->weights_desc(), engine_to_use);
    iter_mem_ = std::make_unique<dnnl::memory>(lstm_pd_->weights_iter_desc(), engine_to_use);

#define create_mem_with_alloc_workspace(name) name##_mem_ = std::make_unique<dnnl::memory>(lstm_pd_->name##_desc(), engine_to_use)
    create_mem_with_alloc_workspace(src_iter);
    create_mem_with_alloc_workspace(src_iter_c);
    create_mem_with_alloc_workspace(dst_iter_c);
    create_mem_with_alloc_workspace(workspace);
#undef create_mem_with_alloc_workspace

    // auto dst_iter_desc = lstm_pd_->dst_iter_desc();
    // printf("dst_iter_desc\n");
    // for (auto i : dst_iter_desc.dims()) {
    //   printf("%ld\n", i);
    // }
    // auto dst_iter_c_desc = lstm_pd_->dst_iter_c_desc();
    // printf("dst_iter_c_desc\n");
    // for (auto i : dst_iter_c_desc.dims()) {
    //   printf("%ld\n", i);
    // }
    // auto dst_desc = lstm_pd_->dst_desc();
    // printf("dst_desc\n");
    // for (auto i : dst_desc.dims()) {
    //   printf("%ld\n", i);
    // }

    // auto src_iter_desc = lstm_pd_->src_iter_desc();
    // printf("src_iter_desc\n");
    // for (auto i : src_iter_desc.dims()) {
    //   printf("%ld\n", i);
    // }

    lstm_forward_ = std::make_unique<dnnl::lstm_forward>(*lstm_pd_);
    net.push_back(*lstm_forward_);
    net_args.push_back({{DNNL_ARG_SRC_LAYER, *src_mem_},
                        {DNNL_ARG_WEIGHTS_LAYER, *weight_mem_},
                        {DNNL_ARG_WEIGHTS_ITER, *iter_mem_},
                        {DNNL_ARG_BIAS, *bias_mem_},
                        {DNNL_ARG_DST_LAYER, *dst_mem_},
                        {DNNL_ARG_SRC_ITER, *src_iter_mem_},
                        {DNNL_ARG_SRC_ITER_C, *src_iter_c_mem_},
                        {DNNL_ARG_DST_ITER, *dst_iter_mem_},
                        {DNNL_ARG_DST_ITER_C, *dst_iter_c_mem_},
                        {DNNL_ARG_WORKSPACE, *workspace_mem_}});
  }

  void ReorderWeights(const OrtCustomOpApi* api, OrtKernelContext* context, const dnnl::engine&, const dnnl::stream& stream) override {
    Ort::CustomOpApi ort{*api};
    int input_index = mklnode_ptr_->input_start_index < 0 ? 0 : mklnode_ptr_->input_start_index;

    const OrtValue* weight_tensor = ort.KernelContext_GetInput(context, input_index + 1);
    const OrtValue* iter_tensor = ort.KernelContext_GetInput(context, input_index + 2);
    T* weight_data = const_cast<T*>(ort.GetTensorData<T>(weight_tensor));
    T* iter_data = const_cast<T*>(ort.GetTensorData<T>(iter_tensor));

    weight_input_mem_->set_data_handle(weight_data);
    iter_input_mem_->set_data_handle(iter_data);

    // printf("weight_input dim\n");
    // for (auto i : weight_input_mem_->get_desc().dims()) {
    //   printf("%ld\n", i);
    // }
    // printf("weight dim from prim\n");
    // for (auto i : weight_mem_->get_desc().dims()) {
    //   printf("%ld\n", i);
    // }
    // printf("reorder weights %lu %lu\n", weight_input_mem_->get_desc().get_size(), weight_mem_->get_desc().get_size());
    auto rpd = dnnl::reorder::primitive_desc(*weight_input_mem_, *weight_mem_);
    dnnl::reorder(rpd)
        .execute(stream, *weight_input_mem_, *weight_mem_);
    dnnl::reorder(*iter_input_mem_, *iter_mem_)
        .execute(stream, *iter_input_mem_, *iter_mem_);
  }

  Status Bind(const OrtCustomOpApi* api, OrtKernelContext* context) override {
    Ort::CustomOpApi ort{*api};

    // int num_inputs = mklnode_ptr_->num_inputs;
    int input_index = mklnode_ptr_->input_start_index < 0 ? 0 : mklnode_ptr_->input_start_index;

    const OrtValue* input_tensor = ort.KernelContext_GetInput(context, input_index);
    T* src_data = const_cast<T*>(ort.GetTensorData<T>(input_tensor));
    src_mem_->set_data_handle(src_data);
    const OrtValue* b_tensor = ort.KernelContext_GetInput(context, input_index + 3);
    T* b_data = const_cast<T*>(ort.GetTensorData<T>(b_tensor));
    bias_mem_->set_data_handle(b_data);

    if (mklnode_ptr_->output_index >= 0) {
      const auto& dst_shape = dst_md_->dims();
      OrtValue* output_dst = ort.KernelContext_GetOutput(context, 0, dst_shape.data(), dst_shape.size());
      T* dst_data = ort.GetTensorMutableData<T>(output_dst);

      dst_mem_->set_data_handle(dst_data);

      const auto& dst_iter_shape = dst_iter_md_->dims();
      dnnl::memory::dims output_shape = dst_iter_shape;
      if (dst_iter_shape.size() > 3 && dst_iter_shape[0] == 1) {
        output_shape.resize(3);
        output_shape[0] = dst_iter_shape[1];
        output_shape[1] = dst_iter_shape[2];
        output_shape[2] = dst_iter_shape[3];
      }
      OrtValue* output_dst_iter = ort.KernelContext_GetOutput(context, 1, output_shape.data(), output_shape.size());
      T* dst_iter_data = ort.GetTensorMutableData<T>(output_dst_iter);

      dst_iter_mem_->set_data_handle(dst_iter_data);
    }
    return Status::OK();
  }

 private:
  void ReadAttributes(const NodeAttributes& attributes,
                      const std::string attributes_prefix = "") override {
    auto attr = attributes.find(attributes_prefix + "hidden_size");
    if (attr != attributes.end()) {
      auto& proto = attr->second();
      Status status = GetIntAttr(proto, hidden_size_);
      if (status != Status::OK()) {
        printf("error ReadAttributes hidden_size\n");
      }
    }
  }

  int64_t hidden_size_;
  std::unique_ptr<dnnl::memory::desc> src_md_;
  // std::unique_ptr<dnnl::memory::desc> weight_md_;
  // std::unique_ptr<dnnl::memory::desc> iter_md_;
  std::unique_ptr<dnnl::memory::desc> bias_md_;
  std::unique_ptr<dnnl::memory::desc> dst_md_;

  std::unique_ptr<dnnl::memory::desc> weight_input_md_;
  std::unique_ptr<dnnl::memory::desc> iter_input_md_;

  std::unique_ptr<dnnl::memory> weight_input_mem_;
  std::unique_ptr<dnnl::memory> iter_input_mem_;

  std::unique_ptr<dnnl::memory> src_mem_;
  std::unique_ptr<dnnl::memory> weight_mem_;
  std::unique_ptr<dnnl::memory> iter_mem_;
  std::unique_ptr<dnnl::memory> bias_mem_;
  std::unique_ptr<dnnl::memory> dst_mem_;

  std::unique_ptr<dnnl::memory::desc> dst_iter_md_;

  std::unique_ptr<dnnl::memory> src_iter_mem_;
  std::unique_ptr<dnnl::memory> src_iter_c_mem_;
  std::unique_ptr<dnnl::memory> dst_iter_mem_;
  std::unique_ptr<dnnl::memory> dst_iter_c_mem_;
  std::unique_ptr<dnnl::memory> workspace_mem_;

  std::unique_ptr<dnnl::lstm_forward::desc> lstm_desc_;

  std::unique_ptr<dnnl::lstm_forward::primitive_desc> lstm_pd_;
  std::unique_ptr<dnnl::lstm_forward> lstm_forward_;
};
}  // namespace ort_dnnl
}  // namespace onnxruntime
