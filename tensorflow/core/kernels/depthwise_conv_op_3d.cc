/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#define EIGEN_USE_THREADS

#include <algorithm>
#include <cmath>
#include <type_traits>

#include "tensorflow/core/kernels/depthwise_conv_op_3d.h"

#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/kernels/conv_ops.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/use_cudnn.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

// In depthwise convolution, one input is convolved into depth_multipler
// outputs and the outputs don't need to be reduced again like what regular
// convolution does.
//  However, the way to apply filters to inputs is exactly the same as the
// regular convolution. Please refer to the regular convolution kernels for
// more details.

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, typename T>
class DepthwiseConv3dNativeOp: public BinaryOp<T> {
 public:
  explicit DepthwiseConv3dNativeOp(OpKernelConstruction* context)
      : BinaryOp<T>(context) {
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES(context, strides_.size() == 5,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 5 dimensions"));

    const int64 stride_z = GetTensorDim(strides_, data_format_, '0');
    const int64 stride_y = GetTensorDim(strides_, data_format_, '1');
    const int64 stride_x = GetTensorDim(strides_, data_format_, '2');
    const int64 stride_n = GetTensorDim(strides_, data_format_, 'N');
    const int64 stride_c = GetTensorDim(strides_, data_format_, 'C');

    OP_REQUIRES(context, stride_z == stride_y && stride_y == stride_x,
                errors::InvalidArgument(
                    "Current implementation only supports equal length "
                    "strides in the x, y and z dimensions."));
    OP_REQUIRES(
        context, (stride_n == 1 && stride_c == 1),
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
  }

  void Compute(OpKernelContext* context) override {
    // Input tensor is of the following dimensions:
    // [ batch, in_z, in_y, in_x, in_depth ]
    const Tensor& input = context->input(0);

    // Input filter is of the following dimensions:
    // [ filter_z, filter_y, filter_x, in_depth, depth_multiplier]
    const Tensor& filter = context->input(1);

    // For 2D convolution, there should be 4 dimensions.
    OP_REQUIRES(context, input.dims() == 5,
                errors::InvalidArgument("input must be 5-dimensional",
                                        input.shape().DebugString()));
    OP_REQUIRES(context, filter.dims() == 5,
                errors::InvalidArgument("filter must be 5-dimensional: ",
                                        filter.shape().DebugString()));

    // in_depth for input and filter must match.
    const int64 in_depth = GetTensorDim(input, data_format_, 'C');
    OP_REQUIRES(context, in_depth == filter.dim_size(3),
                errors::InvalidArgument(
                    "input and filter must have the same channels: ", in_depth,
                    " vs ", filter.dim_size(3)));

    // The last dimension for filter is depth multiplier.
    const int32 depth_multiplier = filter.dim_size(4);

    // The output depth is input depth x depth multipler
    const int32 out_depth = in_depth * depth_multiplier;

    // Dimension order for these arrays is: z, y, x.
    std::array<int64, 3> input_size = {
        {GetTensorDim(input, data_format_, '0'),
         GetTensorDim(input, data_format_, '1'),
         GetTensorDim(input, data_format_, '2')}};
    std::array<int64, 3> filter_size = {
        {filter.dim_size(0), filter.dim_size(1), filter.dim_size(2)}};
    std::array<int64, 3> strides = {{GetTensorDim(strides_, data_format_, '0'),
                                     GetTensorDim(strides_, data_format_, '1'),
                                     GetTensorDim(strides_, data_format_, '2')}};
    std::array<int64, 3> out, padding;

    OP_REQUIRES_OK(context, Get3dOutputSize(input_size, filter_size, strides,
                                            padding_, &out, &padding));

    // The first dimension for input is batch.
    const int32 batch = input.dim_size(0);

    TensorShape out_shape = ShapeFromFormat(
      data_format_, batch, {{out[0], out[1], out[2]}}, out_depth);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

    // If there is nothing to compute, return.
    if (out_shape.num_elements() == 0) {
      return;
    }

  }

 private:
  std::vector<int32> strides_;
  Padding padding_;
  TensorFormat data_format_;
};

#define REGISTER_CPU_KERNEL(T)                                                 \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("DepthwiseConv3dNative").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      DepthwiseConv3dNativeOp<CPUDevice, T>);

TF_CALL_float(REGISTER_CPU_KERNEL);
#if !defined(PLATFORM_WINDOWS) || !defined(_DEBUG)
TF_CALL_double(REGISTER_CPU_KERNEL);
#endif

} // namespace tensorflow
