// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/data_types.h"
#include "core/framework/sparse_tensor.h"
#include "core/framework/data_transfer_manager.h"

using namespace onnxruntime::common;

namespace onnxruntime {

std::ostream& operator<<(std::ostream& os, SparseFormatFlags flags) {
  return os << std::hex << static_cast<uint32_t>(flags);
}

SparseTensor::SparseTensor(MLDataType elt_type,
                           const TensorShape& dense_shape,
                           size_t nnz,
                           void* values_data,
                           const OrtMemoryInfo& memory_info)
    : format_flags_(SparseFormatFlags::kUndefined),
      values_(elt_type, TensorShape({static_cast<int64_t>(nnz)}), values_data, memory_info),
      dense_shape_(dense_shape),
      allocator_(),
      rep_() {}

SparseTensor::SparseTensor(MLDataType elt_type,
                           const TensorShape& dense_shape,
                           size_t nnz,
                           std::shared_ptr<IAllocator> allocator)
    : format_flags_(SparseFormatFlags::kUndefined),
      values_(elt_type, TensorShape({static_cast<int64_t>(nnz)}), allocator),
      dense_shape_(dense_shape),
      allocator_(std::move(allocator)),
      rep_() {}

SparseTensor::~SparseTensor() = default;

SparseRep::~SparseRep() = default;

Status SparseTensor::Copy(const DataTransferManager& data_transfer_manager, int exec_q_id, SparseTensor& dst_tensor) const {
  // Do not copy same destination
  if (values_.DataRaw() == dst_tensor.MutableValues().MutableDataRaw()) {
    return Status::OK();
  }
  ORT_RETURN_IF_NOT(format_flags_ != SparseFormatFlags::kUndefined, "This instance should not be empty");
  ORT_RETURN_IF_NOT(rep_ != nullptr, "This instance should not be empty");
  ORT_RETURN_IF_NOT(dst_tensor.FormatFlags() == SparseFormatFlags::kUndefined, "Destination should be empty");
  ORT_RETURN_IF_NOT(dst_tensor.allocator_ != nullptr, "Destination must have an allocator set");
  ORT_RETURN_IF_NOT(dst_tensor.dense_shape_.Size() >= dense_shape_.Size(), "Destination must have enough space");
  std::unique_ptr<SparseRep> rep_copy;
  ORT_RETURN_IF_ERROR(rep_->Copy(data_transfer_manager, dst_tensor.allocator_, exec_q_id, rep_copy));
  ORT_RETURN_IF_ERROR(data_transfer_manager.CopyTensor(values_, dst_tensor.MutableValues(), exec_q_id));
  dst_tensor.rep_ = std::move(rep_copy);
  dst_tensor.format_flags_ = format_flags_;
  // Allocator and shape must have been already set
  return Status::OK();
}

Status SparseTensor::Copy(const IDataTransfer& data_transfer, SparseTensor& dst_tensor, int exec_q_id) const {
  // Do not copy same destination
  if (this == &dst_tensor) {
    return Status::OK();
  }
  ORT_RETURN_IF_NOT(format_flags_ != SparseFormatFlags::kUndefined, "This instance should not be empty");
  ORT_RETURN_IF_NOT(rep_ != nullptr, "This instance should not be empty");
  ORT_RETURN_IF_NOT(dst_tensor.FormatFlags() == SparseFormatFlags::kUndefined, "Destination should be empty");
  std::unique_ptr<SparseRep> rep_copy;
  ORT_RETURN_IF_NOT(dst_tensor.allocator_ != nullptr, "Destination must have a CPU allocator set");
  ORT_RETURN_IF_NOT(dst_tensor.dense_shape_.Size() >= dense_shape_.Size(), "Destination must have enough space");
  ORT_RETURN_IF_ERROR(rep_->Copy(data_transfer, dst_tensor.allocator_, exec_q_id, rep_copy));
  ORT_RETURN_IF_ERROR(data_transfer.CopyTensor(values_, dst_tensor.MutableValues(), exec_q_id));
  dst_tensor.rep_ = std::move(rep_copy);
  dst_tensor.format_flags_ = format_flags_;
  // Allocator and shape must have been already set
  return Status::OK();
}

}  // namespace onnxruntime
