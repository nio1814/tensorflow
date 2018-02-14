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

#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/kernels/conv_ops_3d.h"
#include "tensorflow/core/kernels/depthwise_conv_op_3d.h"
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

template <typename T>
struct DepthwiseConv3DKernel {
  static void Run(const Depthwise3DArgs& args,
                  const int64 padded_filter_inner_dim_size, const int64 out_r,
                  const int64 out_c, const int64 out_p, const T* filter, const T* input_buffer,
                  T* output, TensorFormat data_format) {
    typedef typename Eigen::internal::packet_traits<T>::type Packet;
    static const int64 kPacketSize = (sizeof(Packet) / sizeof(T));

    const int64 out_depth = args.out_depth;
    const int64 filter_spatial_size = args.filter_rows * args.filter_cols * args.filter_planes;
    const int64 output_scalar_size = out_depth % kPacketSize;
    const int64 output_vectorized_size =
        (out_depth / kPacketSize) * kPacketSize;
    const int64 base_output_index = (out_r * args.out_cols + out_c) * out_depth;

    for (int i = 0; i < output_vectorized_size; i += kPacketSize) {
      // Reset accumulator.
      auto vaccum = Eigen::internal::pset1<Packet>(static_cast<T>(0));
      for (int j = 0; j < filter_spatial_size; ++j) {
        // Calculate index.
        const int64 index = i + j * padded_filter_inner_dim_size;
        // Load filter.
        // TODO(andydavis) Unroll 'out_c' loop in caller so we can load
        // multiple inputs here to amortize the cost of each filter block load.
        const auto filter_block =
            Eigen::internal::ploadu<Packet>(filter + index);
        // Load input.
        const auto data_block =
            Eigen::internal::ploadu<Packet>(input_buffer + index);
        // Vector multiply-add.
        vaccum =
            Eigen::internal::pmadd<Packet>(filter_block, data_block, vaccum);
      }
      // Store vector accumulator to output.
      Eigen::internal::pstoreu<T>(output + base_output_index + i, vaccum);
    }

    if (output_scalar_size > 0) {
      auto vaccum = Eigen::internal::pset1<Packet>(static_cast<T>(0));
      for (int j = 0; j < filter_spatial_size; ++j) {
        const int64 index =
            output_vectorized_size + j * padded_filter_inner_dim_size;
        const auto filter_block =
            Eigen::internal::ploadu<Packet>(filter + index);
        const auto data_block =
            Eigen::internal::ploadu<Packet>(input_buffer + index);
        vaccum =
            Eigen::internal::pmadd<Packet>(filter_block, data_block, vaccum);
      }
      // Load accumulator into an array and loop through output.
      T out_buf[kPacketSize];
      Eigen::internal::pstoreu<T>(out_buf, vaccum);
      const int64 last_output_index =
          base_output_index + output_vectorized_size;
      for (int j = 0; j < output_scalar_size; ++j) {
        output[last_output_index + j] = out_buf[j];
      }
    }
  }
};

template <typename T>
struct LaunchDepthwiseConv3dOp<CPUDevice, T> {
  typedef typename Eigen::internal::packet_traits<T>::type Packet;

  void operator()(OpKernelContext* ctx, const Depthwise3DArgs& args,
                  const T* input, const T* depthwise_filter, T* output,
                  TensorFormat data_format) {
    OP_REQUIRES(
        ctx, data_format == FORMAT_NHWC,
        errors::Unimplemented(
            "Depthwise convolution on CPU is only supported for NDHWC format"));
    static const int64 kPacketSize = (sizeof(Packet) / sizeof(T));

    // Pad 'depthwise_filter' to vector register width (if needed).
    const bool pad_filter = (args.out_depth % kPacketSize) == 0 ? false : true;
    Tensor padded_filter;
    if (pad_filter) {
      // Allocate space for padded filter.
      const int64 filter_spatial_size = args.filter_rows * args.filter_cols * args.filter_planes;
      const int64 padded_filter_inner_dim_size =
          ((args.out_depth + kPacketSize - 1) / kPacketSize) * kPacketSize;
      OP_REQUIRES_OK(
          ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                  TensorShape({filter_spatial_size,
                                               padded_filter_inner_dim_size}),
                                  &padded_filter));
      // Write out padded filter.
      functor::DepthwiseFilterPadOp<T>()(
          args, depthwise_filter, padded_filter.template flat<T>().data());
    }
    const T* filter_data =
        pad_filter ? padded_filter.template flat<T>().data() : depthwise_filter;

    // Computes one shard of depthwise conv2d output.
    auto shard = [&ctx, &args, &input, &filter_data, &output, data_format](
                     int64 start, int64 limit) {
      static const int64 kPacketSize = (sizeof(Packet) / sizeof(T));
      const int64 input_image_size =
          args.in_rows * args.in_cols * args.in_planes * args.in_depth;
      const int64 output_image_size =
          args.out_rows * args.out_cols * args.out_planes * args.out_depth;
      const int64 filter_spatial_size = args.filter_rows * args.filter_cols * args.filter_planes;
      const int64 padded_filter_inner_dim_size =
          ((args.out_depth + kPacketSize - 1) / kPacketSize) * kPacketSize;

      // Allocate buffer for local input regions.
      Tensor input_buffer;
      OP_REQUIRES_OK(
          ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                  TensorShape({filter_spatial_size,
                                               padded_filter_inner_dim_size}),
                                  &input_buffer));
      T* input_buffer_data = input_buffer.template flat<T>().data();

      for (int64 i = start; i < limit; ++i) {
        const int64 b = i / args.out_rows;
        const int64 in_base = b * input_image_size;
        const int64 out_base = b * output_image_size;

        const int64 out_r = i % args.out_rows;

        for (int64 out_c = 0; out_c < args.out_cols; ++out_c) {
          for (int64 out_p = 0; out_p < args.out_planes; ++out_p){
            // Populate 'input_buffer_data' with data from local input region.
            functor::DepthwiseInputCopyOp<T>()(args, padded_filter_inner_dim_size,
                                               out_r, out_c, out_p, input + in_base,
                                               input_buffer_data);

            // Process buffered input across all filters and store to output.
            DepthwiseConv3DKernel<T>::Run(
                  args, padded_filter_inner_dim_size, out_r, out_c, out_p, filter_data,
                  input_buffer_data, output + out_base, data_format);
          }
        }
      }
    };

    const int64 total_shards = args.batch * args.out_rows;

    // Empirically tested to give reasonable performance boosts at batch size 1
    // without reducing throughput at batch size 32.
    const float kCostMultiplier = 2.5f;

    // TODO(andydavis): Estimate shard cost (in cycles) based on the number of
    // flops/loads/stores required to compute one shard.
    const int64 shard_cost = kCostMultiplier * args.out_cols * args.out_depth;

    auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
    Shard(worker_threads.num_threads, worker_threads.workers, total_shards,
          shard_cost, shard);
  }
};
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

    // For 3D convolution, there should be 4 dimensions.
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
