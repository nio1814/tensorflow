/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_KERNELS_CONV_OPS_3D_H
#define TENSORFLOW_KERNELS_CONV_OPS_3D_H

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {

class OpKernelContext;

template <typename Device, typename T>
struct LaunchConvOp {
  static void launch(OpKernelContext* context, bool cudnn_use_autotune,
                     const Tensor& input, const Tensor& filter,
                     const std::array<int64, 3>& strides, const Padding padding,
                     TensorFormat data_format, Tensor* output);
};

// Used to keep track of persistent memory buffers used within the op.
// It uses malloc and free to avoid the time cost of initializing the memory.
template <class T, size_t size>
struct Im2ColBufferResource : public ResourceBase {
  Im2ColBufferResource<T, size>() {
    data = static_cast<T*>(port::Malloc(size * sizeof(T)));
  }
  ~Im2ColBufferResource<T, size>() { port::Free(data); }
  // This mutex ensures that only a single operation at a time is able to use
  // the buffer memory held by this resource.
  mutex mu;
  T* data;
  string DebugString() { return "Im2ColBufferResource"; }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_CONV_OPS_H
