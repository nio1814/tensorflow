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

#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"

#include <cstdio>

namespace tensorflow {

static void SetConstOp(const string& name, TensorShape dimensions, DataType data_type, NodeDef* node) {
  Tensor tensor(data_type, dimensions);
  for (int64 i=0; i<tensor.NumElements(); i++) {
    switch (data_type) {
      case DT_FLOAT:
        tensor.flat<float>()(i) = i/10.0f;
        break;
      case DT_HALF:
        tensor.flat<Eigen::half>()(i) = Eigen::half(1/10.0f);
        break;
      default:
        LOG(FATAL) << "Unknown data type " << data_type;
    }
  }
  TF_CHECK_OK(NodeDefBuilder(name, "Const")
              .Attr("dtype", data_type)
              .Attr("value", tensor)
              .Finalize(node));
}

class Conv3dOpTest : public OpsTestBase
{
protected:
  void HandwrittenConv(const Tensor& image, const Tensor& filter, int stride, const Tensor& expected) {
    TF_EXPECT_OK(NodeDefBuilder("conv3d", "Conv3D")
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Attr("T", DT_FLOAT)
                     .Attr("strides", {1, stride, stride, stride, 1})
                     .Attr("padding", "SAME")
                     .Attr("dilations", {1,1,1,1})
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());

    // The image matrix is:
    // |  1 |  2 |  3 |  4 |
    // |  5 |  6 |  7 |  8 |
    // |  9 | 10 | 11 | 12 |


    // The filter matrix is:
    // | 1 | 4 | 7 |
    // | 2 | 5 | 8 |
    // | 3 | 6 | 9 |

    AddInputFromArray<float>(image.shape(), image.flat<float>());
    AddInputFromArray<float>(filter.shape(), filter.flat<float>());
    TF_ASSERT_OK(RunOpKernel());

    // We're sliding the 3x3 filter across the 3x4 image, with accesses outside
    // the input set to zero because we're using the 'SAME' padding mode.
    // The calculations behind the expected output are:
    // (1*0)+(4*0)+(7*0)+(2*0)+(5*1)+(8*2)+(3*0)+(6*5)+(9*6)=105
    // (1*0)+(4*0)+(7*0)+(2*1)+(5*2)+(8*3)+(3*5)+(6*6)+(9*7)=150
    // (1*0)+(4*0)+(7*0)+(2*2)+(5*3)+(8*4)+(3*6)+(6*7)+(9*8)=183
    // (1*0)+(4*0)+(7*0)+(2*3)+(5*4)+(8*0)+(3*7)+(6*8)+(9*0)=95
    // (1*0)+(4*1)+(7*2)+(2*0)+(5*5)+(8*6)+(3*0)+(6*9)+(9*10)=235
    // (1*1)+(4*2)+(7*3)+(2*5)+(5*6)+(8*7)+(3*9)+(6*10)+(9*11)=312
    // (1*2)+(4*3)+(7*4)+(2*6)+(5*7)+(8*8)+(3*10)+(6*11)+(9*12)=357
    // (1*3)+(4*4)+(7*0)+(2*7)+(5*8)+(8*0)+(3*11)+(6*12)+(9*0)=178
    // (1*0)+(4*5)+(7*6)+(2*0)+(5*9)+(8*10)+(3*0)+(6*0)+(9*0)=187
    // (1*5)+(4*6)+(7*7)+(2*9)+(5*10)+(8*11)+(3*0)+(6*0)+(9*0)=234
    // (1*6)+(4*7)+(7*8)+(2*10)+(5*11)+(8*12)+(3*0)+(6*0)+(9*0)=261
    // (1*7)+(4*11)+(7*0)+(2*8)+(5*12)+(8*0)+(3*0)+(6*0)+(9*0)=121
    // This means we should end up with this matrix:
    // |  105  |  150  |  183  |   95  |
    // |  235  |  312  |  357  |  178  |
    // |  187  |  234  |  261  |  121  |
//    const int expected_depth = image.dim_size(1);
//    const int expected_width = image.dim_size(3);
//    const int expected_height = image.dim_size(2);
//    const int batch_size = image.dim_size(0);
//    const int filter_channels = filter.dim_size(4);
//    Tensor expected(DT_FLOAT, TensorShape({batch_size, expected_depth, expected_height, expected_width, filter_channels}));

    const Tensor& output = *GetOutput(0);
    test::ExpectTensorNear<float>(expected, output, 1e-5);
  }
};

void Conv3dFloatBenchmark(const Tensor& image, const Tensor& filter, int stride, Padding padding, int num_iterations, bool use_gpu, int num_threads, string label) {
  testing::SetLabel(label);

  SessionOptions session_options;
  session_options.config.set_intra_op_parallelism_threads(num_threads);

  GraphDef graph_def;

  SetConstOp("input", image.shape(), image.dtype(), graph_def.add_node());
  SetConstOp("filter", filter.shape(), filter.dtype(), graph_def.add_node());

  NodeDef* conv = graph_def.add_node();
  TF_EXPECT_OK(NodeDefBuilder("conv3d", "Conv3D")
                   .Input("input", 0, DT_FLOAT)
                   .Input("filter", 0, DT_FLOAT)
//                   .Attr("T", DT_FLOAT)
                   .Attr("strides", {1, stride, stride, stride, 1})
                   .Attr("padding", padding==VALID ? "VALID" : "SAME")
                   .Finalize(conv));

//  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  Graph* graph = new Graph(OpRegistry::Global());
  GraphConstructorOptions graph_options;
  TF_CHECK_OK(ConvertGraphDefToGraph(graph_options, graph_def, graph));
  string device = use_gpu ? "gpu" : "cpu";
//  testing::UseRealTime();
  test::Benchmark(device, graph, &session_options).Run(num_iterations);
}

void ConvSpatialSeparable3dFloatBenchmark(const Tensor& image, const Tensor& filter_planes, const Tensor& filter_rows, const Tensor& filter_cols, int stride, Padding padding, int num_iterations, bool use_gpu, int num_threads, string label) {
  testing::SetLabel(label);

  SessionOptions session_options;
  session_options.config.set_intra_op_parallelism_threads(num_threads);

  GraphDef graph_def;

  SetConstOp("input", image.shape(), image.dtype(), graph_def.add_node());
  SetConstOp("filter_planes", filter_planes.shape(), filter_planes.dtype(), graph_def.add_node());
  SetConstOp("filter_rows", filter_rows.shape(), filter_rows.dtype(), graph_def.add_node());
  SetConstOp("filter_cols", filter_cols.shape(), filter_cols.dtype(), graph_def.add_node());

  NodeDef* conv = graph_def.add_node();
  TF_EXPECT_OK(NodeDefBuilder("conv3d", "ConvSpatialSeparable3D")
                   .Input("input", 0, DT_FLOAT)
                   .Input("filter_planes", 0, DT_FLOAT)
                   .Input("filter_rows", 0, DT_FLOAT)
                   .Input("filter_cols", 0, DT_FLOAT)
//                   .Attr("T", DT_FLOAT)
                   .Attr("strides", {1, stride, stride, stride, 1})
                   .Attr("padding", padding==VALID ? "VALID" : "SAME")
                   .Finalize(conv));

//  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  Graph* graph = new Graph(OpRegistry::Global());
  GraphConstructorOptions graph_options;
  TF_CHECK_OK(ConvertGraphDefToGraph(graph_options, graph_def, graph));
  string device = use_gpu ? "gpu" : "cpu";
//  testing::UseRealTime();
  test::Benchmark(device, graph, &session_options).Run(num_iterations);
}

class ConvSpatialSeparable3DOpTest : public OpsTestBase
{
protected:
  void HandwrittenConv(const Tensor& image, const Tensor& filter_planes, const Tensor& filter_rows, const Tensor& filter_cols, int stride, Padding padding, const Tensor& expected) {
    TF_EXPECT_OK(NodeDefBuilder("conv3d", "ConvSpatialSeparable3D")
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
//                     .Attr("T", DT_FLOAT)
                     .Attr("strides", {1, stride, stride, stride, 1})
                     .Attr("padding", padding==VALID ? "VALID" : "SAME")
//                     .Attr("dilations", {1,1,1,1})
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());

    AddInputFromArray<float>(image.shape(), image.flat<float>());
    AddInputFromArray<float>(filter_planes.shape(), filter_planes.flat<float>());
    AddInputFromArray<float>(filter_rows.shape(), filter_rows.flat<float>());
    AddInputFromArray<float>(filter_cols.shape(), filter_cols.flat<float>());
    TF_ASSERT_OK(RunOpKernel());

    const Tensor& output = *GetOutput(0);
    test::ExpectTensorNear<float>(expected, output, 1e-5);
  }
};

//static void Convolution3dFloat(int batch_size, std::array<int,4> input_shape, std::array<int64,4> filter_shape, int stride, Padding padding) {
//  tensorflow::SessionOptions session_options;
//  session_options.config.set_inter_op_parallelism_threads(1);

//  tensorflow::GraphDef graph_def;

//  SetConstOp("input", {batch_size, input_shape[0], input_shape[1], input_shape[2]}, DT_FLOAT, graph_def.add_node());
//  SetConstOp("filter", {filter_shape[0], filter_shape[1], filter_shape[2], input_shape[3], filter_shape[3]}, DT_FLOAT, graph_def.add_node());

//  NodeDef* conv = graph_def.add_node();
//  TF_CHECK_OK(NodeDefBuilder("conv3d", "Conv3D")
//              .Input("input", 0, DT_FLOAT)
//              .Input("filter", 0, DT_FLOAT)
//              .Attr("strides", {1,stride,stride,stride,1})
//              .Attr("padding", padding==VALID ? "VALID" : "SAME")
//              .Finalize(conv));

//  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
//  GraphConstructorOptions graph_options;
//  TF_CHECK_OK(ConvertGraphDefToGraph(graph_options, graph_def, graph.get()));

//  bool use_gpu = false;
//  string device = use_gpu ? "gpu" : "cpu";
//}

//TEST(Convolution3dTest, Small) {
//  Convolution3dFloat(16, {{4,4,4}}, {{3,3,3,2}}, 1, VALID);
//}

namespace tensorflow {

Tensor CreateImage(int batch_size, int image_channels, int image_cols, int image_rows, int image_planes) {
  Tensor image(DT_FLOAT, {batch_size, image_planes, image_rows, image_cols,
                          image_channels});
  std::vector<float> initial_values;
  int image_size = image_planes*image_rows*image_cols;
  for (int n = 0; n<image.NumElements(); n++) {
    initial_values.push_back(n%image_size + 1);
  }
  test::FillValues<float>(&image, initial_values);

  return image;
}

Tensor CreateFilter(int filter_size, int image_channels, int filter_channels) {
  Tensor filter(DT_FLOAT, {filter_size, filter_size, filter_size, image_channels, filter_channels});

  std::vector<float> initial_values;
  for (int n=0; n<filter.NumElements(); n++) {
    initial_values.push_back(n+1);
  }
  test::FillValues<float>(&filter, initial_values);

  return filter;
}

std::array<Tensor,3> CreateFilters(int filter_size, int image_channels, int filter_channels) {
  Tensor planes_filter(DT_FLOAT, {filter_size, 1, 1, image_channels, filter_channels});
  Tensor rows_filter(DT_FLOAT, {1, filter_size, 1, image_channels, filter_channels});
  Tensor cols_filter(DT_FLOAT, {1, 1, filter_size, image_channels, filter_channels});

  std::vector<float> initial_values_planes;
  std::vector<float> initial_values_rows;
  std::vector<float> initial_values_cols;
  for (int n=0; n<filter_size*image_channels*filter_channels; n++) {
    int index = n+1;
    initial_values_planes.push_back(index*filter_size*filter_size);
    initial_values_rows.push_back(index*filter_size);
    initial_values_cols.push_back(index);
  }
  test::FillValues<float>(&planes_filter, initial_values_planes);
  test::FillValues<float>(&rows_filter, initial_values_rows);
  test::FillValues<float>(&cols_filter, initial_values_cols);

  return {{planes_filter, rows_filter, cols_filter}};
}

Tensor CreateOutput(int batch_size, int channels, int cols, int rows, int planes, std::vector<float> values) {
  Tensor output(DT_FLOAT, {batch_size, planes, rows, cols, channels});
  test::FillValues<float>(
      &output, values);
  return output;
}

static void Conv3dFloatBenchmark_1(int num_iterations) {
  int channels = 16;
  Tensor image = tensorflow::CreateImage(10, channels, 14, 14, channels);
  Tensor filter = tensorflow::CreateFilter(5, channels, 32);

  Conv3dFloatBenchmark(image, filter, 1, SAME, num_iterations, false, 1, "140-140-32");
}

static void ConvSpatialSeparable3dFloatBenchmark_1(int num_iterations) {
  int channels = 16;
  Tensor image = tensorflow::CreateImage(10, channels, 14, 14, channels);
  std::array<Tensor,3> filters = tensorflow::CreateFilters(5, channels, 32);

  ConvSpatialSeparable3dFloatBenchmark(image, filters[0], filters[1], filters[2], 1, SAME, num_iterations, false, 1, "14-14-32");
}

BENCHMARK(Conv3dFloatBenchmark_1);
BENCHMARK(ConvSpatialSeparable3dFloatBenchmark_1);

}

TEST_F(Conv3dOpTest, SingleImageSingleFilter) {
  Tensor image = tensorflow::CreateImage(1, 1, 4, 3, 2);
  Tensor filter = tensorflow::CreateFilter(3, 1, 1);
  Tensor output = tensorflow::CreateOutput(1, 1, 4, 3, 2, {1800, 2768, 3008, 2036, 3045, 4638, 4971, 3339, 2132, 3224, 3428, 2288, 1116, 1688, 1820, 1208, 1803, 2694, 2865, 1881, 1160, 1712, 1808, 1172});
  HandwrittenConv(image, filter, 1, output);
}

TEST_F(Conv3dOpTest, BatchImageSingleFilter) {
  int batch_size = 10;
  Tensor image = tensorflow::CreateImage(batch_size, 1, 4, 3, 2);
  Tensor filter = tensorflow::CreateFilter(3, 1, 1);
  std::vector<float> output_values_1 = {1800, 2768, 3008, 2036, 3045, 4638, 4971, 3339, 2132, 3224, 3428, 2288, 1116, 1688, 1820, 1208, 1803, 2694, 2865, 1881, 1160, 1712, 1808, 1172};
  std::vector<float> output_values;
  for (int b=0; b<batch_size; b++) {
    output_values.insert(output_values.end(), output_values_1.begin(), output_values_1.end());
  }
  Tensor output = tensorflow::CreateOutput(batch_size, 1, 4, 3, 2, output_values);
  HandwrittenConv(image, filter, 1, output);
}

TEST_F(ConvSpatialSeparable3DOpTest, Small) {
  int batch_size = 10;
  Tensor image = tensorflow::CreateImage(batch_size, 1, 4, 3, 2);
  std::array<Tensor,3> filters = tensorflow::CreateFilters(3, 1, 1);
  std::vector<float> output_values_1 = {37800, 48330, 52380, 26865, 57240, 72252, 77112, 39366, 31320, 39366, 41796, 21303, 24300, 30942, 33372, 17091, 36288, 45684, 48600, 24786, 19764, 24786, 26244, 13365};
  std::vector<float> output_values;
  for (int b=0; b<batch_size; b++) {
    output_values.insert(output_values.end(), output_values_1.begin(), output_values_1.end());
  }
  Tensor output = tensorflow::CreateOutput(batch_size, 1, 4, 3, 2, output_values);
  HandwrittenConv(image, filters[0],filters[1], filters[2], 1, SAME, output);
}

//TEST_F(ConvSpatialSeparable3DOpTest, Full) {
//  int batch_size = 10;
//  Tensor image = tensorflow::CreateImage(batch_size, 8, 14, 14, 32);
//  std::array<Tensor,3> filters = tensorflow::CreateFilters(5, 8, 16);
////  std::vector<float> output_values_1 = {37800, 48330, 52380, 26865, 57240, 72252, 77112, 39366, 31320, 39366, 41796, 21303, 24300, 30942, 33372, 17091, 36288, 45684, 48600, 24786, 19764, 24786, 26244, 13365};
//  std::vector<float> output_values(image.NumElements()*2);
////  for (int b=0; b<batch_size; b++) {
////    output_values.insert(output_values.end(), output_values_1.begin(), output_values_1.end());
////  }
//  Tensor output = tensorflow::CreateOutput(batch_size, 16, 14, 14, 32, output_values);
//  HandwrittenConv(image, filters[0],filters[1], filters[2], 1, output);
//}

}
