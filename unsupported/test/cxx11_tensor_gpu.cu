// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define EIGEN_TEST_NO_LONGDOUBLE
#define EIGEN_TEST_NO_COMPLEX

#define EIGEN_USE_GPU

#include "main.h"
#include <unsupported/Eigen/CXX11/Tensor>

#include <unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h>

#define EIGEN_GPU_TEST_C99_MATH  EIGEN_HAS_CXX11

using Eigen::Tensor;

void test_gpu_nullary() {
  Tensor<float, 1, 0, int> in1(2);
  Tensor<float, 1, 0, int> in2(2);
  in1.setRandom();
  in2.setRandom();

  std::size_t tensor_bytes = in1.size() * sizeof(float);

  float* d_in1;
  float* d_in2;
  gpuMalloc((void**)(&d_in1), tensor_bytes);
  gpuMalloc((void**)(&d_in2), tensor_bytes);
  gpuMemcpy(d_in1, in1.data(), tensor_bytes, gpuMemcpyHostToDevice);
  gpuMemcpy(d_in2, in2.data(), tensor_bytes, gpuMemcpyHostToDevice);

  Eigen::GpuStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);

  Eigen::TensorMap<Eigen::Tensor<float, 1, 0, int>, Eigen::Aligned> gpu_in1(
      d_in1, 2);
  Eigen::TensorMap<Eigen::Tensor<float, 1, 0, int>, Eigen::Aligned> gpu_in2(
      d_in2, 2);

  gpu_in1.device(gpu_device) = gpu_in1.constant(3.14f);
  gpu_in2.device(gpu_device) = gpu_in2.random();

  Tensor<float, 1, 0, int> new1(2);
  Tensor<float, 1, 0, int> new2(2);

  assert(gpuMemcpyAsync(new1.data(), d_in1, tensor_bytes, gpuMemcpyDeviceToHost,
                         gpu_device.stream()) == gpuSuccess);
  assert(gpuMemcpyAsync(new2.data(), d_in2, tensor_bytes, gpuMemcpyDeviceToHost,
                         gpu_device.stream()) == gpuSuccess);

  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);

  for (int i = 0; i < 2; ++i) {
    VERIFY_IS_APPROX(new1(i), 3.14f);
    VERIFY_IS_NOT_EQUAL(new2(i), in2(i));
  }

  gpuFree(d_in1);
  gpuFree(d_in2);
}

void test_gpu_elementwise_small() {
  Tensor<float, 1> in1(Eigen::array<Eigen::DenseIndex, 1>(2));
  Tensor<float, 1> in2(Eigen::array<Eigen::DenseIndex, 1>(2));
  Tensor<float, 1> out(Eigen::array<Eigen::DenseIndex, 1>(2));
  in1.setRandom();
  in2.setRandom();

  std::size_t in1_bytes = in1.size() * sizeof(float);
  std::size_t in2_bytes = in2.size() * sizeof(float);
  std::size_t out_bytes = out.size() * sizeof(float);

  float* d_in1;
  float* d_in2;
  float* d_out;
  gpuMalloc((void**)(&d_in1), in1_bytes);
  gpuMalloc((void**)(&d_in2), in2_bytes);
  gpuMalloc((void**)(&d_out), out_bytes);

  gpuMemcpy(d_in1, in1.data(), in1_bytes, gpuMemcpyHostToDevice);
  gpuMemcpy(d_in2, in2.data(), in2_bytes, gpuMemcpyHostToDevice);

  Eigen::GpuStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);

  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_in1(
      d_in1, Eigen::array<Eigen::DenseIndex, 1>(2));
  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_in2(
      d_in2, Eigen::array<Eigen::DenseIndex, 1>(2));
  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_out(
      d_out, Eigen::array<Eigen::DenseIndex, 1>(2));

  gpu_out.device(gpu_device) = gpu_in1 + gpu_in2;

  assert(gpuMemcpyAsync(out.data(), d_out, out_bytes, gpuMemcpyDeviceToHost,
                         gpu_device.stream()) == gpuSuccess);
  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);

  for (int i = 0; i < 2; ++i) {
    VERIFY_IS_APPROX(
        out(Eigen::array<Eigen::DenseIndex, 1>(i)),
        in1(Eigen::array<Eigen::DenseIndex, 1>(i)) + in2(Eigen::array<Eigen::DenseIndex, 1>(i)));
  }

  gpuFree(d_in1);
  gpuFree(d_in2);
  gpuFree(d_out);
}

void test_gpu_elementwise()
{
  Tensor<float, 3> in1(Eigen::array<Eigen::DenseIndex, 3>(72,53,97));
  Tensor<float, 3> in2(Eigen::array<Eigen::DenseIndex, 3>(72,53,97));
  Tensor<float, 3> in3(Eigen::array<Eigen::DenseIndex, 3>(72,53,97));
  Tensor<float, 3> out(Eigen::array<Eigen::DenseIndex, 3>(72,53,97));
  in1.setRandom();
  in2.setRandom();
  in3.setRandom();

  std::size_t in1_bytes = in1.size() * sizeof(float);
  std::size_t in2_bytes = in2.size() * sizeof(float);
  std::size_t in3_bytes = in3.size() * sizeof(float);
  std::size_t out_bytes = out.size() * sizeof(float);

  float* d_in1;
  float* d_in2;
  float* d_in3;
  float* d_out;
  gpuMalloc((void**)(&d_in1), in1_bytes);
  gpuMalloc((void**)(&d_in2), in2_bytes);
  gpuMalloc((void**)(&d_in3), in3_bytes);
  gpuMalloc((void**)(&d_out), out_bytes);

  gpuMemcpy(d_in1, in1.data(), in1_bytes, gpuMemcpyHostToDevice);
  gpuMemcpy(d_in2, in2.data(), in2_bytes, gpuMemcpyHostToDevice);
  gpuMemcpy(d_in3, in3.data(), in3_bytes, gpuMemcpyHostToDevice);

  Eigen::GpuStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);

  Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_in1(d_in1, Eigen::array<Eigen::DenseIndex, 3>(72,53,97));
  Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_in2(d_in2, Eigen::array<Eigen::DenseIndex, 3>(72,53,97));
  Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_in3(d_in3, Eigen::array<Eigen::DenseIndex, 3>(72,53,97));
  Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_out(d_out, Eigen::array<Eigen::DenseIndex, 3>(72,53,97));

  gpu_out.device(gpu_device) = gpu_in1 + gpu_in2 * gpu_in3;

  assert(gpuMemcpyAsync(out.data(), d_out, out_bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);

  for (int i = 0; i < 72; ++i) {
    for (int j = 0; j < 53; ++j) {
      for (int k = 0; k < 97; ++k) {
        VERIFY_IS_APPROX(out(Eigen::array<Eigen::DenseIndex, 3>(i,j,k)), in1(Eigen::array<Eigen::DenseIndex, 3>(i,j,k)) + in2(Eigen::array<Eigen::DenseIndex, 3>(i,j,k)) * in3(Eigen::array<Eigen::DenseIndex, 3>(i,j,k)));
      }
    }
  }

  gpuFree(d_in1);
  gpuFree(d_in2);
  gpuFree(d_in3);
  gpuFree(d_out);
}

void test_gpu_props() {
  Tensor<float, 1> in1(200);
  Tensor<bool, 1> out(200);
  in1.setRandom();

  std::size_t in1_bytes = in1.size() * sizeof(float);
  std::size_t out_bytes = out.size() * sizeof(bool);

  float* d_in1;
  bool* d_out;
  gpuMalloc((void**)(&d_in1), in1_bytes);
  gpuMalloc((void**)(&d_out), out_bytes);

  gpuMemcpy(d_in1, in1.data(), in1_bytes, gpuMemcpyHostToDevice);

  Eigen::GpuStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);

  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_in1(
      d_in1, 200);
  Eigen::TensorMap<Eigen::Tensor<bool, 1>, Eigen::Aligned> gpu_out(
      d_out, 200);

  gpu_out.device(gpu_device) = (gpu_in1.isnan)();

  assert(gpuMemcpyAsync(out.data(), d_out, out_bytes, gpuMemcpyDeviceToHost,
                         gpu_device.stream()) == gpuSuccess);
  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);

  for (int i = 0; i < 200; ++i) {
    VERIFY_IS_EQUAL(out(i), (std::isnan)(in1(i)));
  }

  gpuFree(d_in1);
  gpuFree(d_out);
}

void test_gpu_reduction()
{
  Tensor<float, 4> in1(72,53,97,113);
  Tensor<float, 2> out(72,97);
  in1.setRandom();

  std::size_t in1_bytes = in1.size() * sizeof(float);
  std::size_t out_bytes = out.size() * sizeof(float);

  float* d_in1;
  float* d_out;
  gpuMalloc((void**)(&d_in1), in1_bytes);
  gpuMalloc((void**)(&d_out), out_bytes);

  gpuMemcpy(d_in1, in1.data(), in1_bytes, gpuMemcpyHostToDevice);

  Eigen::GpuStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);

  Eigen::TensorMap<Eigen::Tensor<float, 4> > gpu_in1(d_in1, 72,53,97,113);
  Eigen::TensorMap<Eigen::Tensor<float, 2> > gpu_out(d_out, 72,97);

  array<Eigen::DenseIndex, 2> reduction_axis;
  reduction_axis[0] = 1;
  reduction_axis[1] = 3;

  gpu_out.device(gpu_device) = gpu_in1.maximum(reduction_axis);

  assert(gpuMemcpyAsync(out.data(), d_out, out_bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);

  for (int i = 0; i < 72; ++i) {
    for (int j = 0; j < 97; ++j) {
      float expected = 0;
      for (int k = 0; k < 53; ++k) {
        for (int l = 0; l < 113; ++l) {
          expected =
              std::max<float>(expected, in1(i, k, j, l));
        }
      }
      VERIFY_IS_APPROX(out(i,j), expected);
    }
  }

  gpuFree(d_in1);
  gpuFree(d_out);
}

template<int DataLayout>
void test_gpu_contraction()
{
  // with these dimensions, the output has 300 * 140 elements, which is
  // more than 30 * 1024, which is the number of threads in blocks on
  // a 15 SM GK110 GPU
  Tensor<float, 4, DataLayout> t_left(6, 50, 3, 31);
  Tensor<float, 5, DataLayout> t_right(Eigen::array<Eigen::DenseIndex, 5>(3, 31, 7, 20, 1));
  Tensor<float, 5, DataLayout> t_result(Eigen::array<Eigen::DenseIndex, 5>(6, 50, 7, 20, 1));

  t_left.setRandom();
  t_right.setRandom();

  std::size_t t_left_bytes = t_left.size()  * sizeof(float);
  std::size_t t_right_bytes = t_right.size() * sizeof(float);
  std::size_t t_result_bytes = t_result.size() * sizeof(float);

  float* d_t_left;
  float* d_t_right;
  float* d_t_result;

  gpuMalloc((void**)(&d_t_left), t_left_bytes);
  gpuMalloc((void**)(&d_t_right), t_right_bytes);
  gpuMalloc((void**)(&d_t_result), t_result_bytes);

  gpuMemcpy(d_t_left, t_left.data(), t_left_bytes, gpuMemcpyHostToDevice);
  gpuMemcpy(d_t_right, t_right.data(), t_right_bytes, gpuMemcpyHostToDevice);

  Eigen::GpuStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);

  Eigen::TensorMap<Eigen::Tensor<float, 4, DataLayout> > gpu_t_left(d_t_left, 6, 50, 3, 31);
  Eigen::TensorMap<Eigen::Tensor<float, 5, DataLayout> > gpu_t_right(d_t_right, 3, 31, 7, 20, 1);
  Eigen::TensorMap<Eigen::Tensor<float, 5, DataLayout> > gpu_t_result(d_t_result, 6, 50, 7, 20, 1);

  typedef Eigen::Map<Eigen::Matrix<float, Dynamic, Dynamic, DataLayout> > MapXf;
  MapXf m_left(t_left.data(), 300, 93);
  MapXf m_right(t_right.data(), 93, 140);
  Eigen::Matrix<float, Dynamic, Dynamic, DataLayout> m_result(300, 140);

  typedef Tensor<float, 1>::DimensionPair DimPair;
  Eigen::array<DimPair, 2> dims;
  dims[0] = DimPair(2, 0);
  dims[1] = DimPair(3, 1);

  m_result = m_left * m_right;
  gpu_t_result.device(gpu_device) = gpu_t_left.contract(gpu_t_right, dims);

  gpuMemcpy(t_result.data(), d_t_result, t_result_bytes, gpuMemcpyDeviceToHost);

  for (DenseIndex i = 0; i < t_result.size(); i++) {
    if (fabs(t_result.data()[i] - m_result.data()[i]) >= 1e-4f) {
      std::cout << "mismatch detected at index " << i << ": " << t_result.data()[i] << " vs " <<  m_result.data()[i] << std::endl;
      assert(false);
    }
  }

  gpuFree(d_t_left);
  gpuFree(d_t_right);
  gpuFree(d_t_result);
}

template<int DataLayout>
void test_gpu_convolution_1d()
{
  Tensor<float, 4, DataLayout> input(74,37,11,137);
  Tensor<float, 1, DataLayout> kernel(4);
  Tensor<float, 4, DataLayout> out(74,34,11,137);
  input = input.constant(10.0f) + input.random();
  kernel = kernel.constant(7.0f) + kernel.random();

  std::size_t input_bytes = input.size() * sizeof(float);
  std::size_t kernel_bytes = kernel.size() * sizeof(float);
  std::size_t out_bytes = out.size() * sizeof(float);

  float* d_input;
  float* d_kernel;
  float* d_out;
  gpuMalloc((void**)(&d_input), input_bytes);
  gpuMalloc((void**)(&d_kernel), kernel_bytes);
  gpuMalloc((void**)(&d_out), out_bytes);

  gpuMemcpy(d_input, input.data(), input_bytes, gpuMemcpyHostToDevice);
  gpuMemcpy(d_kernel, kernel.data(), kernel_bytes, gpuMemcpyHostToDevice);

  Eigen::GpuStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);

  Eigen::TensorMap<Eigen::Tensor<float, 4, DataLayout> > gpu_input(d_input, 74,37,11,137);
  Eigen::TensorMap<Eigen::Tensor<float, 1, DataLayout> > gpu_kernel(d_kernel, 4);
  Eigen::TensorMap<Eigen::Tensor<float, 4, DataLayout> > gpu_out(d_out, 74,34,11,137);

  Eigen::array<Eigen::DenseIndex, 1> dims(1);
  gpu_out.device(gpu_device) = gpu_input.convolve(gpu_kernel, dims);

  assert(gpuMemcpyAsync(out.data(), d_out, out_bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);

  for (int i = 0; i < 74; ++i) {
    for (int j = 0; j < 34; ++j) {
      for (int k = 0; k < 11; ++k) {
        for (int l = 0; l < 137; ++l) {
          const float result = out(i,j,k,l);
          const float expected = input(i,j+0,k,l) * kernel(0) + input(i,j+1,k,l) * kernel(1) +
                                 input(i,j+2,k,l) * kernel(2) + input(i,j+3,k,l) * kernel(3);
          VERIFY_IS_APPROX(result, expected);
        }
      }
    }
  }

  gpuFree(d_input);
  gpuFree(d_kernel);
  gpuFree(d_out);
}

void test_gpu_convolution_inner_dim_col_major_1d()
{
  Tensor<float, 4, ColMajor> input(74,9,11,7);
  Tensor<float, 1, ColMajor> kernel(4);
  Tensor<float, 4, ColMajor> out(71,9,11,7);
  input = input.constant(10.0f) + input.random();
  kernel = kernel.constant(7.0f) + kernel.random();

  std::size_t input_bytes = input.size() * sizeof(float);
  std::size_t kernel_bytes = kernel.size() * sizeof(float);
  std::size_t out_bytes = out.size() * sizeof(float);

  float* d_input;
  float* d_kernel;
  float* d_out;
  gpuMalloc((void**)(&d_input), input_bytes);
  gpuMalloc((void**)(&d_kernel), kernel_bytes);
  gpuMalloc((void**)(&d_out), out_bytes);

  gpuMemcpy(d_input, input.data(), input_bytes, gpuMemcpyHostToDevice);
  gpuMemcpy(d_kernel, kernel.data(), kernel_bytes, gpuMemcpyHostToDevice);

  Eigen::GpuStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);

  Eigen::TensorMap<Eigen::Tensor<float, 4, ColMajor> > gpu_input(d_input,74,9,11,7);
  Eigen::TensorMap<Eigen::Tensor<float, 1, ColMajor> > gpu_kernel(d_kernel,4);
  Eigen::TensorMap<Eigen::Tensor<float, 4, ColMajor> > gpu_out(d_out,71,9,11,7);

  Eigen::array<Eigen::DenseIndex, 1> dims(0);
  gpu_out.device(gpu_device) = gpu_input.convolve(gpu_kernel, dims);

  assert(gpuMemcpyAsync(out.data(), d_out, out_bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);

  for (int i = 0; i < 71; ++i) {
    for (int j = 0; j < 9; ++j) {
      for (int k = 0; k < 11; ++k) {
        for (int l = 0; l < 7; ++l) {
          const float result = out(i,j,k,l);
          const float expected = input(i+0,j,k,l) * kernel(0) + input(i+1,j,k,l) * kernel(1) +
                                 input(i+2,j,k,l) * kernel(2) + input(i+3,j,k,l) * kernel(3);
          VERIFY_IS_APPROX(result, expected);
        }
      }
    }
  }

  gpuFree(d_input);
  gpuFree(d_kernel);
  gpuFree(d_out);
}

void test_gpu_convolution_inner_dim_row_major_1d()
{
  Tensor<float, 4, RowMajor> input(7,9,11,74);
  Tensor<float, 1, RowMajor> kernel(4);
  Tensor<float, 4, RowMajor> out(7,9,11,71);
  input = input.constant(10.0f) + input.random();
  kernel = kernel.constant(7.0f) + kernel.random();

  std::size_t input_bytes = input.size() * sizeof(float);
  std::size_t kernel_bytes = kernel.size() * sizeof(float);
  std::size_t out_bytes = out.size() * sizeof(float);

  float* d_input;
  float* d_kernel;
  float* d_out;
  gpuMalloc((void**)(&d_input), input_bytes);
  gpuMalloc((void**)(&d_kernel), kernel_bytes);
  gpuMalloc((void**)(&d_out), out_bytes);

  gpuMemcpy(d_input, input.data(), input_bytes, gpuMemcpyHostToDevice);
  gpuMemcpy(d_kernel, kernel.data(), kernel_bytes, gpuMemcpyHostToDevice);

  Eigen::GpuStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);

  Eigen::TensorMap<Eigen::Tensor<float, 4, RowMajor> > gpu_input(d_input, 7,9,11,74);
  Eigen::TensorMap<Eigen::Tensor<float, 1, RowMajor> > gpu_kernel(d_kernel, 4);
  Eigen::TensorMap<Eigen::Tensor<float, 4, RowMajor> > gpu_out(d_out, 7,9,11,71);

  Eigen::array<Eigen::DenseIndex, 1> dims(3);
  gpu_out.device(gpu_device) = gpu_input.convolve(gpu_kernel, dims);

  assert(gpuMemcpyAsync(out.data(), d_out, out_bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);

  for (int i = 0; i < 7; ++i) {
    for (int j = 0; j < 9; ++j) {
      for (int k = 0; k < 11; ++k) {
        for (int l = 0; l < 71; ++l) {
          const float result = out(i,j,k,l);
          const float expected = input(i,j,k,l+0) * kernel(0) + input(i,j,k,l+1) * kernel(1) +
                                 input(i,j,k,l+2) * kernel(2) + input(i,j,k,l+3) * kernel(3);
          VERIFY_IS_APPROX(result, expected);
        }
      }
    }
  }

  gpuFree(d_input);
  gpuFree(d_kernel);
  gpuFree(d_out);
}

template<int DataLayout>
void test_gpu_convolution_2d()
{
  Tensor<float, 4, DataLayout> input(74,37,11,137);
  Tensor<float, 2, DataLayout> kernel(3,4);
  Tensor<float, 4, DataLayout> out(74,35,8,137);
  input = input.constant(10.0f) + input.random();
  kernel = kernel.constant(7.0f) + kernel.random();

  std::size_t input_bytes = input.size() * sizeof(float);
  std::size_t kernel_bytes = kernel.size() * sizeof(float);
  std::size_t out_bytes = out.size() * sizeof(float);

  float* d_input;
  float* d_kernel;
  float* d_out;
  gpuMalloc((void**)(&d_input), input_bytes);
  gpuMalloc((void**)(&d_kernel), kernel_bytes);
  gpuMalloc((void**)(&d_out), out_bytes);

  gpuMemcpy(d_input, input.data(), input_bytes, gpuMemcpyHostToDevice);
  gpuMemcpy(d_kernel, kernel.data(), kernel_bytes, gpuMemcpyHostToDevice);

  Eigen::GpuStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);

  Eigen::TensorMap<Eigen::Tensor<float, 4, DataLayout> > gpu_input(d_input,74,37,11,137);
  Eigen::TensorMap<Eigen::Tensor<float, 2, DataLayout> > gpu_kernel(d_kernel,3,4);
  Eigen::TensorMap<Eigen::Tensor<float, 4, DataLayout> > gpu_out(d_out,74,35,8,137);

  Eigen::array<Eigen::DenseIndex, 2> dims(1,2);
  gpu_out.device(gpu_device) = gpu_input.convolve(gpu_kernel, dims);

  assert(gpuMemcpyAsync(out.data(), d_out, out_bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);

  for (int i = 0; i < 74; ++i) {
    for (int j = 0; j < 35; ++j) {
      for (int k = 0; k < 8; ++k) {
        for (int l = 0; l < 137; ++l) {
          const float result = out(i,j,k,l);
          const float expected = input(i,j+0,k+0,l) * kernel(0,0) +
                                 input(i,j+1,k+0,l) * kernel(1,0) +
                                 input(i,j+2,k+0,l) * kernel(2,0) +
                                 input(i,j+0,k+1,l) * kernel(0,1) +
                                 input(i,j+1,k+1,l) * kernel(1,1) +
                                 input(i,j+2,k+1,l) * kernel(2,1) +
                                 input(i,j+0,k+2,l) * kernel(0,2) +
                                 input(i,j+1,k+2,l) * kernel(1,2) +
                                 input(i,j+2,k+2,l) * kernel(2,2) +
                                 input(i,j+0,k+3,l) * kernel(0,3) +
                                 input(i,j+1,k+3,l) * kernel(1,3) +
                                 input(i,j+2,k+3,l) * kernel(2,3);
          VERIFY_IS_APPROX(result, expected);
        }
      }
    }
  }

  gpuFree(d_input);
  gpuFree(d_kernel);
  gpuFree(d_out);
}

template<int DataLayout>
void test_gpu_convolution_3d()
{
  Tensor<float, 5, DataLayout> input(Eigen::array<Eigen::DenseIndex, 5>(74,37,11,137,17));
  Tensor<float, 3, DataLayout> kernel(3,4,2);
  Tensor<float, 5, DataLayout> out(Eigen::array<Eigen::DenseIndex, 5>(74,35,8,136,17));
  input = input.constant(10.0f) + input.random();
  kernel = kernel.constant(7.0f) + kernel.random();

  std::size_t input_bytes = input.size() * sizeof(float);
  std::size_t kernel_bytes = kernel.size() * sizeof(float);
  std::size_t out_bytes = out.size() * sizeof(float);

  float* d_input;
  float* d_kernel;
  float* d_out;
  gpuMalloc((void**)(&d_input), input_bytes);
  gpuMalloc((void**)(&d_kernel), kernel_bytes);
  gpuMalloc((void**)(&d_out), out_bytes);

  gpuMemcpy(d_input, input.data(), input_bytes, gpuMemcpyHostToDevice);
  gpuMemcpy(d_kernel, kernel.data(), kernel_bytes, gpuMemcpyHostToDevice);

  Eigen::GpuStreamDevice stream;    
  Eigen::GpuDevice gpu_device(&stream);

  Eigen::TensorMap<Eigen::Tensor<float, 5, DataLayout> > gpu_input(d_input,74,37,11,137,17);
  Eigen::TensorMap<Eigen::Tensor<float, 3, DataLayout> > gpu_kernel(d_kernel,3,4,2);
  Eigen::TensorMap<Eigen::Tensor<float, 5, DataLayout> > gpu_out(d_out,74,35,8,136,17);

  Eigen::array<Eigen::DenseIndex, 3> dims(1,2,3);
  gpu_out.device(gpu_device) = gpu_input.convolve(gpu_kernel, dims);

  assert(gpuMemcpyAsync(out.data(), d_out, out_bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);

  for (int i = 0; i < 74; ++i) {
    for (int j = 0; j < 35; ++j) {
      for (int k = 0; k < 8; ++k) {
        for (int l = 0; l < 136; ++l) {
          for (int m = 0; m < 17; ++m) {
            const float result = out(i,j,k,l,m);
            const float expected = input(i,j+0,k+0,l+0,m) * kernel(0,0,0) +
                                   input(i,j+1,k+0,l+0,m) * kernel(1,0,0) +
                                   input(i,j+2,k+0,l+0,m) * kernel(2,0,0) +
                                   input(i,j+0,k+1,l+0,m) * kernel(0,1,0) +
                                   input(i,j+1,k+1,l+0,m) * kernel(1,1,0) +
                                   input(i,j+2,k+1,l+0,m) * kernel(2,1,0) +
                                   input(i,j+0,k+2,l+0,m) * kernel(0,2,0) +
                                   input(i,j+1,k+2,l+0,m) * kernel(1,2,0) +
                                   input(i,j+2,k+2,l+0,m) * kernel(2,2,0) +
                                   input(i,j+0,k+3,l+0,m) * kernel(0,3,0) +
                                   input(i,j+1,k+3,l+0,m) * kernel(1,3,0) +
                                   input(i,j+2,k+3,l+0,m) * kernel(2,3,0) +
                                   input(i,j+0,k+0,l+1,m) * kernel(0,0,1) +
                                   input(i,j+1,k+0,l+1,m) * kernel(1,0,1) +
                                   input(i,j+2,k+0,l+1,m) * kernel(2,0,1) +
                                   input(i,j+0,k+1,l+1,m) * kernel(0,1,1) +
                                   input(i,j+1,k+1,l+1,m) * kernel(1,1,1) +
                                   input(i,j+2,k+1,l+1,m) * kernel(2,1,1) +
                                   input(i,j+0,k+2,l+1,m) * kernel(0,2,1) +
                                   input(i,j+1,k+2,l+1,m) * kernel(1,2,1) +
                                   input(i,j+2,k+2,l+1,m) * kernel(2,2,1) +
                                   input(i,j+0,k+3,l+1,m) * kernel(0,3,1) +
                                   input(i,j+1,k+3,l+1,m) * kernel(1,3,1) +
                                   input(i,j+2,k+3,l+1,m) * kernel(2,3,1);
            VERIFY_IS_APPROX(result, expected);
          }
        }
      }
    }
  }

  gpuFree(d_input);
  gpuFree(d_kernel);
  gpuFree(d_out);
}


#if EIGEN_GPU_TEST_C99_MATH
template <typename Scalar>
void test_gpu_lgamma(const Scalar stddev)
{
  Tensor<Scalar, 2> in(72,97);
  in.setRandom();
  in *= in.constant(stddev);
  Tensor<Scalar, 2> out(72,97);
  out.setZero();

  std::size_t bytes = in.size() * sizeof(Scalar);

  Scalar* d_in;
  Scalar* d_out;
  gpuMalloc((void**)(&d_in), bytes);
  gpuMalloc((void**)(&d_out), bytes);

  gpuMemcpy(d_in, in.data(), bytes, gpuMemcpyHostToDevice);

  Eigen::GpuStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);

  Eigen::TensorMap<Eigen::Tensor<Scalar, 2> > gpu_in(d_in, 72, 97);
  Eigen::TensorMap<Eigen::Tensor<Scalar, 2> > gpu_out(d_out, 72, 97);

  gpu_out.device(gpu_device) = gpu_in.lgamma();

  assert(gpuMemcpyAsync(out.data(), d_out, bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);

  for (int i = 0; i < 72; ++i) {
    for (int j = 0; j < 97; ++j) {
      VERIFY_IS_APPROX(out(i,j), (std::lgamma)(in(i,j)));
    }
  }

  gpuFree(d_in);
  gpuFree(d_out);
}
#endif

template <typename Scalar>
void test_gpu_digamma()
{
  Tensor<Scalar, 1> in(7);
  Tensor<Scalar, 1> out(7);
  Tensor<Scalar, 1> expected_out(7);
  out.setZero();

  in(0) = Scalar(1);
  in(1) = Scalar(1.5);
  in(2) = Scalar(4);
  in(3) = Scalar(-10.5);
  in(4) = Scalar(10000.5);
  in(5) = Scalar(0);
  in(6) = Scalar(-1);

  expected_out(0) = Scalar(-0.5772156649015329);
  expected_out(1) = Scalar(0.03648997397857645);
  expected_out(2) = Scalar(1.2561176684318);
  expected_out(3) = Scalar(2.398239129535781);
  expected_out(4) = Scalar(9.210340372392849);
  expected_out(5) = std::numeric_limits<Scalar>::infinity();
  expected_out(6) = std::numeric_limits<Scalar>::infinity();

  std::size_t bytes = in.size() * sizeof(Scalar);

  Scalar* d_in;
  Scalar* d_out;
  gpuMalloc((void**)(&d_in), bytes);
  gpuMalloc((void**)(&d_out), bytes);

  gpuMemcpy(d_in, in.data(), bytes, gpuMemcpyHostToDevice);

  Eigen::GpuStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);

  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_in(d_in, 7);
  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_out(d_out, 7);

  gpu_out.device(gpu_device) = gpu_in.digamma();

  assert(gpuMemcpyAsync(out.data(), d_out, bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);

  for (int i = 0; i < 5; ++i) {
    VERIFY_IS_APPROX(out(i), expected_out(i));
  }
  for (int i = 5; i < 7; ++i) {
    VERIFY_IS_EQUAL(out(i), expected_out(i));
  }

  gpuFree(d_in);
  gpuFree(d_out);
}

template <typename Scalar>
void test_gpu_zeta()
{
  Tensor<Scalar, 1> in_x(6);
  Tensor<Scalar, 1> in_q(6);
  Tensor<Scalar, 1> out(6);
  Tensor<Scalar, 1> expected_out(6);
  out.setZero();

  in_x(0) = Scalar(1);
  in_x(1) = Scalar(1.5);
  in_x(2) = Scalar(4);
  in_x(3) = Scalar(-10.5);
  in_x(4) = Scalar(10000.5);
  in_x(5) = Scalar(3);
  
  in_q(0) = Scalar(1.2345);
  in_q(1) = Scalar(2);
  in_q(2) = Scalar(1.5);
  in_q(3) = Scalar(3);
  in_q(4) = Scalar(1.0001);
  in_q(5) = Scalar(-2.5);

  expected_out(0) = std::numeric_limits<Scalar>::infinity();
  expected_out(1) = Scalar(1.61237534869);
  expected_out(2) = Scalar(0.234848505667);
  expected_out(3) = Scalar(1.03086757337e-5);
  expected_out(4) = Scalar(0.367879440865);
  expected_out(5) = Scalar(0.054102025820864097);

  std::size_t bytes = in_x.size() * sizeof(Scalar);

  Scalar* d_in_x;
  Scalar* d_in_q;
  Scalar* d_out;
  gpuMalloc((void**)(&d_in_x), bytes);
  gpuMalloc((void**)(&d_in_q), bytes);
  gpuMalloc((void**)(&d_out), bytes);

  gpuMemcpy(d_in_x, in_x.data(), bytes, gpuMemcpyHostToDevice);
  gpuMemcpy(d_in_q, in_q.data(), bytes, gpuMemcpyHostToDevice);
  
  Eigen::GpuStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);

  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_in_x(d_in_x, 6);
  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_in_q(d_in_q, 6);
  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_out(d_out, 6);

  gpu_out.device(gpu_device) = gpu_in_x.zeta(gpu_in_q);

  assert(gpuMemcpyAsync(out.data(), d_out, bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);

  VERIFY_IS_EQUAL(out(0), expected_out(0));
  VERIFY((std::isnan)(out(3)));

  for (int i = 1; i < 6; ++i) {
    if (i != 3) {
      VERIFY_IS_APPROX(out(i), expected_out(i));
    }
  }

  gpuFree(d_in_x);
  gpuFree(d_in_q);
  gpuFree(d_out);
}

template <typename Scalar>
void test_gpu_polygamma()
{
  Tensor<Scalar, 1> in_x(7);
  Tensor<Scalar, 1> in_n(7);
  Tensor<Scalar, 1> out(7);
  Tensor<Scalar, 1> expected_out(7);
  out.setZero();

  in_n(0) = Scalar(1);
  in_n(1) = Scalar(1);
  in_n(2) = Scalar(1);
  in_n(3) = Scalar(17);
  in_n(4) = Scalar(31);
  in_n(5) = Scalar(28);
  in_n(6) = Scalar(8);
  
  in_x(0) = Scalar(2);
  in_x(1) = Scalar(3);
  in_x(2) = Scalar(25.5);
  in_x(3) = Scalar(4.7);
  in_x(4) = Scalar(11.8);
  in_x(5) = Scalar(17.7);
  in_x(6) = Scalar(30.2);

  expected_out(0) = Scalar(0.644934066848);
  expected_out(1) = Scalar(0.394934066848);
  expected_out(2) = Scalar(0.0399946696496);
  expected_out(3) = Scalar(293.334565435);
  expected_out(4) = Scalar(0.445487887616);
  expected_out(5) = Scalar(-2.47810300902e-07);
  expected_out(6) = Scalar(-8.29668781082e-09);

  std::size_t bytes = in_x.size() * sizeof(Scalar);

  Scalar* d_in_x;
  Scalar* d_in_n;
  Scalar* d_out;
  gpuMalloc((void**)(&d_in_x), bytes);
  gpuMalloc((void**)(&d_in_n), bytes);
  gpuMalloc((void**)(&d_out), bytes);

  gpuMemcpy(d_in_x, in_x.data(), bytes, gpuMemcpyHostToDevice);
  gpuMemcpy(d_in_n, in_n.data(), bytes, gpuMemcpyHostToDevice);
  
  Eigen::GpuStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);

  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_in_x(d_in_x, 7);
  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_in_n(d_in_n, 7);
  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_out(d_out, 7);

  gpu_out.device(gpu_device) = gpu_in_n.polygamma(gpu_in_x);

  assert(gpuMemcpyAsync(out.data(), d_out, bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);

  for (int i = 0; i < 7; ++i) {
    VERIFY_IS_APPROX(out(i), expected_out(i));
  }

  gpuFree(d_in_x);
  gpuFree(d_in_n);
  gpuFree(d_out);
}

template <typename Scalar>
void test_gpu_igamma()
{
  Tensor<Scalar, 2> a(6, 6);
  Tensor<Scalar, 2> x(6, 6);
  Tensor<Scalar, 2> out(6, 6);
  out.setZero();

  Scalar a_s[] = {Scalar(0), Scalar(1), Scalar(1.5), Scalar(4), Scalar(0.0001), Scalar(1000.5)};
  Scalar x_s[] = {Scalar(0), Scalar(1), Scalar(1.5), Scalar(4), Scalar(0.0001), Scalar(1000.5)};

  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      a(i, j) = a_s[i];
      x(i, j) = x_s[j];
    }
  }

  Scalar nan = std::numeric_limits<Scalar>::quiet_NaN();
  Scalar igamma_s[][6] = {{0.0, nan, nan, nan, nan, nan},
                          {0.0, 0.6321205588285578, 0.7768698398515702,
                           0.9816843611112658, 9.999500016666262e-05, 1.0},
                          {0.0, 0.4275932955291202, 0.608374823728911,
                           0.9539882943107686, 7.522076445089201e-07, 1.0},
                          {0.0, 0.01898815687615381, 0.06564245437845008,
                           0.5665298796332909, 4.166333347221828e-18, 1.0},
                          {0.0, 0.9999780593618628, 0.9999899967080838,
                           0.9999996219837988, 0.9991370418689945, 1.0},
                          {0.0, 0.0, 0.0, 0.0, 0.0, 0.5042041932513908}};



  std::size_t bytes = a.size() * sizeof(Scalar);

  Scalar* d_a;
  Scalar* d_x;
  Scalar* d_out;
  assert(gpuMalloc((void**)(&d_a), bytes) == gpuSuccess);
  assert(gpuMalloc((void**)(&d_x), bytes) == gpuSuccess);
  assert(gpuMalloc((void**)(&d_out), bytes) == gpuSuccess);

  gpuMemcpy(d_a, a.data(), bytes, gpuMemcpyHostToDevice);
  gpuMemcpy(d_x, x.data(), bytes, gpuMemcpyHostToDevice);

  Eigen::GpuStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);

  Eigen::TensorMap<Eigen::Tensor<Scalar, 2> > gpu_a(d_a, 6, 6);
  Eigen::TensorMap<Eigen::Tensor<Scalar, 2> > gpu_x(d_x, 6, 6);
  Eigen::TensorMap<Eigen::Tensor<Scalar, 2> > gpu_out(d_out, 6, 6);

  gpu_out.device(gpu_device) = gpu_a.igamma(gpu_x);

  assert(gpuMemcpyAsync(out.data(), d_out, bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);

  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      if ((std::isnan)(igamma_s[i][j])) {
        VERIFY((std::isnan)(out(i, j)));
      } else {
        VERIFY_IS_APPROX(out(i, j), igamma_s[i][j]);
      }
    }
  }

  gpuFree(d_a);
  gpuFree(d_x);
  gpuFree(d_out);
}

template <typename Scalar>
void test_gpu_igammac()
{
  Tensor<Scalar, 2> a(6, 6);
  Tensor<Scalar, 2> x(6, 6);
  Tensor<Scalar, 2> out(6, 6);
  out.setZero();

  Scalar a_s[] = {Scalar(0), Scalar(1), Scalar(1.5), Scalar(4), Scalar(0.0001), Scalar(1000.5)};
  Scalar x_s[] = {Scalar(0), Scalar(1), Scalar(1.5), Scalar(4), Scalar(0.0001), Scalar(1000.5)};

  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      a(i, j) = a_s[i];
      x(i, j) = x_s[j];
    }
  }

  Scalar nan = std::numeric_limits<Scalar>::quiet_NaN();
  Scalar igammac_s[][6] = {{nan, nan, nan, nan, nan, nan},
                           {1.0, 0.36787944117144233, 0.22313016014842982,
                            0.018315638888734182, 0.9999000049998333, 0.0},
                           {1.0, 0.5724067044708798, 0.3916251762710878,
                            0.04601170568923136, 0.9999992477923555, 0.0},
                           {1.0, 0.9810118431238462, 0.9343575456215499,
                            0.4334701203667089, 1.0, 0.0},
                           {1.0, 2.1940638138146658e-05, 1.0003291916285e-05,
                            3.7801620118431334e-07, 0.0008629581310054535,
                            0.0},
                           {1.0, 1.0, 1.0, 1.0, 1.0, 0.49579580674813944}};

  std::size_t bytes = a.size() * sizeof(Scalar);

  Scalar* d_a;
  Scalar* d_x;
  Scalar* d_out;
  gpuMalloc((void**)(&d_a), bytes);
  gpuMalloc((void**)(&d_x), bytes);
  gpuMalloc((void**)(&d_out), bytes);

  gpuMemcpy(d_a, a.data(), bytes, gpuMemcpyHostToDevice);
  gpuMemcpy(d_x, x.data(), bytes, gpuMemcpyHostToDevice);

  Eigen::GpuStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);

  Eigen::TensorMap<Eigen::Tensor<Scalar, 2> > gpu_a(d_a, 6, 6);
  Eigen::TensorMap<Eigen::Tensor<Scalar, 2> > gpu_x(d_x, 6, 6);
  Eigen::TensorMap<Eigen::Tensor<Scalar, 2> > gpu_out(d_out, 6, 6);

  gpu_out.device(gpu_device) = gpu_a.igammac(gpu_x);

  assert(gpuMemcpyAsync(out.data(), d_out, bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);

  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      if ((std::isnan)(igammac_s[i][j])) {
        VERIFY((std::isnan)(out(i, j)));
      } else {
        VERIFY_IS_APPROX(out(i, j), igammac_s[i][j]);
      }
    }
  }

  gpuFree(d_a);
  gpuFree(d_x);
  gpuFree(d_out);
}

#if EIGEN_GPU_TEST_C99_MATH
template <typename Scalar>
void test_gpu_erf(const Scalar stddev)
{
  Tensor<Scalar, 2> in(72,97);
  in.setRandom();
  in *= in.constant(stddev);
  Tensor<Scalar, 2> out(72,97);
  out.setZero();

  std::size_t bytes = in.size() * sizeof(Scalar);

  Scalar* d_in;
  Scalar* d_out;
  assert(gpuMalloc((void**)(&d_in), bytes) == gpuSuccess);
  assert(gpuMalloc((void**)(&d_out), bytes) == gpuSuccess);

  gpuMemcpy(d_in, in.data(), bytes, gpuMemcpyHostToDevice);

  Eigen::GpuStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);

  Eigen::TensorMap<Eigen::Tensor<Scalar, 2> > gpu_in(d_in, 72, 97);
  Eigen::TensorMap<Eigen::Tensor<Scalar, 2> > gpu_out(d_out, 72, 97);

  gpu_out.device(gpu_device) = gpu_in.erf();

  assert(gpuMemcpyAsync(out.data(), d_out, bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);

  for (int i = 0; i < 72; ++i) {
    for (int j = 0; j < 97; ++j) {
      VERIFY_IS_APPROX(out(i,j), (std::erf)(in(i,j)));
    }
  }

  gpuFree(d_in);
  gpuFree(d_out);
}

template <typename Scalar>
void test_gpu_erfc(const Scalar stddev)
{
  Tensor<Scalar, 2> in(72,97);
  in.setRandom();
  in *= in.constant(stddev);
  Tensor<Scalar, 2> out(72,97);
  out.setZero();

  std::size_t bytes = in.size() * sizeof(Scalar);

  Scalar* d_in;
  Scalar* d_out;
  gpuMalloc((void**)(&d_in), bytes);
  gpuMalloc((void**)(&d_out), bytes);

  gpuMemcpy(d_in, in.data(), bytes, gpuMemcpyHostToDevice);

  Eigen::GpuStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);

  Eigen::TensorMap<Eigen::Tensor<Scalar, 2> > gpu_in(d_in, 72, 97);
  Eigen::TensorMap<Eigen::Tensor<Scalar, 2> > gpu_out(d_out, 72, 97);

  gpu_out.device(gpu_device) = gpu_in.erfc();

  assert(gpuMemcpyAsync(out.data(), d_out, bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);

  for (int i = 0; i < 72; ++i) {
    for (int j = 0; j < 97; ++j) {
      VERIFY_IS_APPROX(out(i,j), (std::erfc)(in(i,j)));
    }
  }

  gpuFree(d_in);
  gpuFree(d_out);
}
#endif
template <typename Scalar>
void test_gpu_ndtri()
{
  Tensor<Scalar, 1> in_x(8);
  Tensor<Scalar, 1> out(8);
  Tensor<Scalar, 1> expected_out(8);
  out.setZero();

  in_x(0) = Scalar(1);
  in_x(1) = Scalar(0.);
  in_x(2) = Scalar(0.5);
  in_x(3) = Scalar(0.2);
  in_x(4) = Scalar(0.8);
  in_x(5) = Scalar(0.9);
  in_x(6) = Scalar(0.1);
  in_x(7) = Scalar(0.99);
  in_x(8) = Scalar(0.01);

  expected_out(0) = std::numeric_limits<Scalar>::infinity();
  expected_out(1) = -std::numeric_limits<Scalar>::infinity();
  expected_out(2) = Scalar(0.0);
  expected_out(3) = Scalar(-0.8416212335729142);
  expected_out(4) = Scalar(0.8416212335729142);
  expected_out(5) = Scalar(1.2815515655446004);
  expected_out(6) = Scalar(-1.2815515655446004);
  expected_out(7) = Scalar(2.3263478740408408);
  expected_out(8) = Scalar(-2.3263478740408408);

  std::size_t bytes = in_x.size() * sizeof(Scalar);

  Scalar* d_in_x;
  Scalar* d_out;
  gpuMalloc((void**)(&d_in_x), bytes);
  gpuMalloc((void**)(&d_out), bytes);

  gpuMemcpy(d_in_x, in_x.data(), bytes, gpuMemcpyHostToDevice);

  Eigen::GpuStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);

  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_in_x(d_in_x, 6);
  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_out(d_out, 6);

  gpu_out.device(gpu_device) = gpu_in_x.ndtri();

  assert(gpuMemcpyAsync(out.data(), d_out, bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);

  VERIFY_IS_EQUAL(out(0), expected_out(0));
  VERIFY((std::isnan)(out(3)));

  for (int i = 1; i < 6; ++i) {
    if (i != 3) {
      VERIFY_IS_APPROX(out(i), expected_out(i));
    }
  }

  gpuFree(d_in_x);
  gpuFree(d_out);
}

template <typename Scalar>
void test_gpu_betainc()
{
  Tensor<Scalar, 1> in_x(125);
  Tensor<Scalar, 1> in_a(125);
  Tensor<Scalar, 1> in_b(125);
  Tensor<Scalar, 1> out(125);
  Tensor<Scalar, 1> expected_out(125);
  out.setZero();

  Scalar nan = std::numeric_limits<Scalar>::quiet_NaN();

  Array<Scalar, 1, Dynamic> x(125);
  Array<Scalar, 1, Dynamic> a(125);
  Array<Scalar, 1, Dynamic> b(125);
  Array<Scalar, 1, Dynamic> v(125);

  a << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.03062277660168379, 0.03062277660168379, 0.03062277660168379,
      0.03062277660168379, 0.03062277660168379, 0.03062277660168379,
      0.03062277660168379, 0.03062277660168379, 0.03062277660168379,
      0.03062277660168379, 0.03062277660168379, 0.03062277660168379,
      0.03062277660168379, 0.03062277660168379, 0.03062277660168379,
      0.03062277660168379, 0.03062277660168379, 0.03062277660168379,
      0.03062277660168379, 0.03062277660168379, 0.03062277660168379,
      0.03062277660168379, 0.03062277660168379, 0.03062277660168379,
      0.03062277660168379, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999,
      0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999,
      0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 31.62177660168379,
      31.62177660168379, 31.62177660168379, 31.62177660168379,
      31.62177660168379, 31.62177660168379, 31.62177660168379,
      31.62177660168379, 31.62177660168379, 31.62177660168379,
      31.62177660168379, 31.62177660168379, 31.62177660168379,
      31.62177660168379, 31.62177660168379, 31.62177660168379,
      31.62177660168379, 31.62177660168379, 31.62177660168379,
      31.62177660168379, 31.62177660168379, 31.62177660168379,
      31.62177660168379, 31.62177660168379, 31.62177660168379, 999.999, 999.999,
      999.999, 999.999, 999.999, 999.999, 999.999, 999.999, 999.999, 999.999,
      999.999, 999.999, 999.999, 999.999, 999.999, 999.999, 999.999, 999.999,
      999.999, 999.999, 999.999, 999.999, 999.999, 999.999, 999.999;

  b << 0.0, 0.0, 0.0, 0.0, 0.0, 0.03062277660168379, 0.03062277660168379,
      0.03062277660168379, 0.03062277660168379, 0.03062277660168379, 0.999,
      0.999, 0.999, 0.999, 0.999, 31.62177660168379, 31.62177660168379,
      31.62177660168379, 31.62177660168379, 31.62177660168379, 999.999, 999.999,
      999.999, 999.999, 999.999, 0.0, 0.0, 0.0, 0.0, 0.0, 0.03062277660168379,
      0.03062277660168379, 0.03062277660168379, 0.03062277660168379,
      0.03062277660168379, 0.999, 0.999, 0.999, 0.999, 0.999, 31.62177660168379,
      31.62177660168379, 31.62177660168379, 31.62177660168379,
      31.62177660168379, 999.999, 999.999, 999.999, 999.999, 999.999, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.03062277660168379, 0.03062277660168379,
      0.03062277660168379, 0.03062277660168379, 0.03062277660168379, 0.999,
      0.999, 0.999, 0.999, 0.999, 31.62177660168379, 31.62177660168379,
      31.62177660168379, 31.62177660168379, 31.62177660168379, 999.999, 999.999,
      999.999, 999.999, 999.999, 0.0, 0.0, 0.0, 0.0, 0.0, 0.03062277660168379,
      0.03062277660168379, 0.03062277660168379, 0.03062277660168379,
      0.03062277660168379, 0.999, 0.999, 0.999, 0.999, 0.999, 31.62177660168379,
      31.62177660168379, 31.62177660168379, 31.62177660168379,
      31.62177660168379, 999.999, 999.999, 999.999, 999.999, 999.999, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.03062277660168379, 0.03062277660168379,
      0.03062277660168379, 0.03062277660168379, 0.03062277660168379, 0.999,
      0.999, 0.999, 0.999, 0.999, 31.62177660168379, 31.62177660168379,
      31.62177660168379, 31.62177660168379, 31.62177660168379, 999.999, 999.999,
      999.999, 999.999, 999.999;

  x << -0.1, 0.2, 0.5, 0.8, 1.1, -0.1, 0.2, 0.5, 0.8, 1.1, -0.1, 0.2, 0.5, 0.8,
      1.1, -0.1, 0.2, 0.5, 0.8, 1.1, -0.1, 0.2, 0.5, 0.8, 1.1, -0.1, 0.2, 0.5,
      0.8, 1.1, -0.1, 0.2, 0.5, 0.8, 1.1, -0.1, 0.2, 0.5, 0.8, 1.1, -0.1, 0.2,
      0.5, 0.8, 1.1, -0.1, 0.2, 0.5, 0.8, 1.1, -0.1, 0.2, 0.5, 0.8, 1.1, -0.1,
      0.2, 0.5, 0.8, 1.1, -0.1, 0.2, 0.5, 0.8, 1.1, -0.1, 0.2, 0.5, 0.8, 1.1,
      -0.1, 0.2, 0.5, 0.8, 1.1, -0.1, 0.2, 0.5, 0.8, 1.1, -0.1, 0.2, 0.5, 0.8,
      1.1, -0.1, 0.2, 0.5, 0.8, 1.1, -0.1, 0.2, 0.5, 0.8, 1.1, -0.1, 0.2, 0.5,
      0.8, 1.1, -0.1, 0.2, 0.5, 0.8, 1.1, -0.1, 0.2, 0.5, 0.8, 1.1, -0.1, 0.2,
      0.5, 0.8, 1.1, -0.1, 0.2, 0.5, 0.8, 1.1, -0.1, 0.2, 0.5, 0.8, 1.1;

  v << nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
      nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
      nan, nan, 0.47972119876364683, 0.5, 0.5202788012363533, nan, nan,
      0.9518683957740043, 0.9789663010413743, 0.9931729188073435, nan, nan,
      0.999995949033062, 0.9999999999993698, 0.9999999999999999, nan, nan,
      0.9999999999999999, 0.9999999999999999, 0.9999999999999999, nan, nan, nan,
      nan, nan, nan, nan, 0.006827081192655869, 0.0210336989586256,
      0.04813160422599567, nan, nan, 0.20014344256217678, 0.5000000000000001,
      0.7998565574378232, nan, nan, 0.9991401428435834, 0.999999999698403,
      0.9999999999999999, nan, nan, 0.9999999999999999, 0.9999999999999999,
      0.9999999999999999, nan, nan, nan, nan, nan, nan, nan,
      1.0646600232370887e-25, 6.301722877826246e-13, 4.050966937974938e-06, nan,
      nan, 7.864342668429763e-23, 3.015969667594166e-10, 0.0008598571564165444,
      nan, nan, 6.031987710123844e-08, 0.5000000000000007, 0.9999999396801229,
      nan, nan, 0.9999999999999999, 0.9999999999999999, 0.9999999999999999, nan,
      nan, nan, nan, nan, nan, nan, 0.0, 7.029920380986636e-306,
      2.2450728208591345e-101, nan, nan, 0.0, 9.275871147869727e-302,
      1.2232913026152827e-97, nan, nan, 0.0, 3.0891393081932924e-252,
      2.9303043666183996e-60, nan, nan, 2.248913486879199e-196,
      0.5000000000004947, 0.9999999999999999, nan;

  for (int i = 0; i < 125; ++i) {
    in_x(i) = x(i);
    in_a(i) = a(i);
    in_b(i) = b(i);
    expected_out(i) = v(i);
  }

  std::size_t bytes = in_x.size() * sizeof(Scalar);

  Scalar* d_in_x;
  Scalar* d_in_a;
  Scalar* d_in_b;
  Scalar* d_out;
  gpuMalloc((void**)(&d_in_x), bytes);
  gpuMalloc((void**)(&d_in_a), bytes);
  gpuMalloc((void**)(&d_in_b), bytes);
  gpuMalloc((void**)(&d_out), bytes);

  gpuMemcpy(d_in_x, in_x.data(), bytes, gpuMemcpyHostToDevice);
  gpuMemcpy(d_in_a, in_a.data(), bytes, gpuMemcpyHostToDevice);
  gpuMemcpy(d_in_b, in_b.data(), bytes, gpuMemcpyHostToDevice);

  Eigen::GpuStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);

  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_in_x(d_in_x, 125);
  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_in_a(d_in_a, 125);
  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_in_b(d_in_b, 125);
  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_out(d_out, 125);

  gpu_out.device(gpu_device) = betainc(gpu_in_a, gpu_in_b, gpu_in_x);

  assert(gpuMemcpyAsync(out.data(), d_out, bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);

  for (int i = 1; i < 125; ++i) {
    if ((std::isnan)(expected_out(i))) {
      VERIFY((std::isnan)(out(i)));
    } else {
      VERIFY_IS_APPROX(out(i), expected_out(i));
    }
  }

  gpuFree(d_in_x);
  gpuFree(d_in_a);
  gpuFree(d_in_b);
  gpuFree(d_out);
}

template <typename Scalar>
void test_gpu_i0e()
{
  Tensor<Scalar, 1> in_x(21);
  Tensor<Scalar, 1> out(21);
  Tensor<Scalar, 1> expected_out(21);
  out.setZero();

  Array<Scalar, 1, Dynamic> in_x_array(21);
  Array<Scalar, 1, Dynamic> expected_out_array(21);

  in_x_array << -20.0, -18.0, -16.0, -14.0, -12.0, -10.0, -8.0, -6.0, -4.0,
      -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0;

  expected_out_array << 0.0897803118848, 0.0947062952128, 0.100544127361,
      0.107615251671, 0.116426221213, 0.127833337163, 0.143431781857,
      0.16665743264, 0.207001921224, 0.308508322554, 1.0, 0.308508322554,
      0.207001921224, 0.16665743264, 0.143431781857, 0.127833337163,
      0.116426221213, 0.107615251671, 0.100544127361, 0.0947062952128,
      0.0897803118848;

  for (int i = 0; i < 21; ++i) {
    in_x(i) = in_x_array(i);
    expected_out(i) = expected_out_array(i);
  }

  std::size_t bytes = in_x.size() * sizeof(Scalar);

  Scalar* d_in;
  Scalar* d_out;
  gpuMalloc((void**)(&d_in), bytes);
  gpuMalloc((void**)(&d_out), bytes);

  gpuMemcpy(d_in, in_x.data(), bytes, gpuMemcpyHostToDevice);

  Eigen::GpuStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);

  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_in(d_in, 21);
  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_out(d_out, 21);

  gpu_out.device(gpu_device) = gpu_in.bessel_i0e();

  assert(gpuMemcpyAsync(out.data(), d_out, bytes, gpuMemcpyDeviceToHost,
                         gpu_device.stream()) == gpuSuccess);
  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);

  for (int i = 0; i < 21; ++i) {
    VERIFY_IS_APPROX(out(i), expected_out(i));
  }

  gpuFree(d_in);
  gpuFree(d_out);
}

template <typename Scalar>
void test_gpu_i1e()
{
  Tensor<Scalar, 1> in_x(21);
  Tensor<Scalar, 1> out(21);
  Tensor<Scalar, 1> expected_out(21);
  out.setZero();

  Array<Scalar, 1, Dynamic> in_x_array(21);
  Array<Scalar, 1, Dynamic> expected_out_array(21);

  in_x_array << -20.0, -18.0, -16.0, -14.0, -12.0, -10.0, -8.0, -6.0, -4.0,
      -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0;

  expected_out_array << -0.0875062221833, -0.092036796872, -0.0973496147565,
      -0.103697667463, -0.11146429929, -0.121262681384, -0.134142493293,
      -0.152051459309, -0.178750839502, -0.215269289249, 0.0, 0.215269289249,
      0.178750839502, 0.152051459309, 0.134142493293, 0.121262681384,
      0.11146429929, 0.103697667463, 0.0973496147565, 0.092036796872,
      0.0875062221833;

  for (int i = 0; i < 21; ++i) {
    in_x(i) = in_x_array(i);
    expected_out(i) = expected_out_array(i);
  }

  std::size_t bytes = in_x.size() * sizeof(Scalar);

  Scalar* d_in;
  Scalar* d_out;
  gpuMalloc((void**)(&d_in), bytes);
  gpuMalloc((void**)(&d_out), bytes);

  gpuMemcpy(d_in, in_x.data(), bytes, gpuMemcpyHostToDevice);

  Eigen::GpuStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);

  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_in(d_in, 21);
  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_out(d_out, 21);

  gpu_out.device(gpu_device) = gpu_in.bessel_i1e();

  assert(gpuMemcpyAsync(out.data(), d_out, bytes, gpuMemcpyDeviceToHost,
                         gpu_device.stream()) == gpuSuccess);
  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);

  for (int i = 0; i < 21; ++i) {
    VERIFY_IS_APPROX(out(i), expected_out(i));
  }

  gpuFree(d_in);
  gpuFree(d_out);
}

template <typename Scalar>
void test_gpu_igamma_der_a()
{
  Tensor<Scalar, 1> in_x(30);
  Tensor<Scalar, 1> in_a(30);
  Tensor<Scalar, 1> out(30);
  Tensor<Scalar, 1> expected_out(30);
  out.setZero();

  Array<Scalar, 1, Dynamic> in_a_array(30);
  Array<Scalar, 1, Dynamic> in_x_array(30);
  Array<Scalar, 1, Dynamic> expected_out_array(30);

  // See special_functions.cpp for the Python code that generates the test data.

  in_a_array << 0.01, 0.01, 0.01, 0.01, 0.01, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0,
      1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0, 10.0, 100.0, 100.0, 100.0, 100.0,
      100.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0;

  in_x_array << 1.25668890405e-26, 1.17549435082e-38, 1.20938905072e-05,
      1.17549435082e-38, 1.17549435082e-38, 5.66572070696e-16, 0.0132865061065,
      0.0200034203853, 6.29263709118e-17, 1.37160367764e-06, 0.333412038288,
      1.18135687766, 0.580629033777, 0.170631439426, 0.786686768458,
      7.63873279537, 13.1944344379, 11.896042354, 10.5830172417, 10.5020942233,
      92.8918587747, 95.003720371, 86.3715926467, 96.0330217672, 82.6389930677,
      968.702906754, 969.463546828, 1001.79726022, 955.047416547, 1044.27458568;

  expected_out_array << -32.7256441441, -36.4394150514, -9.66467612263,
      -36.4394150514, -36.4394150514, -1.0891900302, -2.66351229645,
      -2.48666868596, -0.929700494428, -3.56327722764, -0.455320135314,
      -0.391437214323, -0.491352055991, -0.350454834292, -0.471773162921,
      -0.104084440522, -0.0723646747909, -0.0992828975532, -0.121638215446,
      -0.122619605294, -0.0317670267286, -0.0359974812869, -0.0154359225363,
      -0.0375775365921, -0.00794899153653, -0.00777303219211, -0.00796085782042,
      -0.0125850719397, -0.00455500206958, -0.00476436993148;

  for (int i = 0; i < 30; ++i) {
    in_x(i) = in_x_array(i);
    in_a(i) = in_a_array(i);
    expected_out(i) = expected_out_array(i);
  }

  std::size_t bytes = in_x.size() * sizeof(Scalar);

  Scalar* d_a;
  Scalar* d_x;
  Scalar* d_out;
  gpuMalloc((void**)(&d_a), bytes);
  gpuMalloc((void**)(&d_x), bytes);
  gpuMalloc((void**)(&d_out), bytes);

  gpuMemcpy(d_a, in_a.data(), bytes, gpuMemcpyHostToDevice);
  gpuMemcpy(d_x, in_x.data(), bytes, gpuMemcpyHostToDevice);

  Eigen::GpuStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);

  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_a(d_a, 30);
  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_x(d_x, 30);
  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_out(d_out, 30);

  gpu_out.device(gpu_device) = gpu_a.igamma_der_a(gpu_x);

  assert(gpuMemcpyAsync(out.data(), d_out, bytes, gpuMemcpyDeviceToHost,
                         gpu_device.stream()) == gpuSuccess);
  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);

  for (int i = 0; i < 30; ++i) {
    VERIFY_IS_APPROX(out(i), expected_out(i));
  }

  gpuFree(d_a);
  gpuFree(d_x);
  gpuFree(d_out);
}

template <typename Scalar>
void test_gpu_gamma_sample_der_alpha()
{
  Tensor<Scalar, 1> in_alpha(30);
  Tensor<Scalar, 1> in_sample(30);
  Tensor<Scalar, 1> out(30);
  Tensor<Scalar, 1> expected_out(30);
  out.setZero();

  Array<Scalar, 1, Dynamic> in_alpha_array(30);
  Array<Scalar, 1, Dynamic> in_sample_array(30);
  Array<Scalar, 1, Dynamic> expected_out_array(30);

  // See special_functions.cpp for the Python code that generates the test data.

  in_alpha_array << 0.01, 0.01, 0.01, 0.01, 0.01, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0,
      1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0, 10.0, 100.0, 100.0, 100.0,
      100.0, 100.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0;

  in_sample_array << 1.25668890405e-26, 1.17549435082e-38, 1.20938905072e-05,
      1.17549435082e-38, 1.17549435082e-38, 5.66572070696e-16, 0.0132865061065,
      0.0200034203853, 6.29263709118e-17, 1.37160367764e-06, 0.333412038288,
      1.18135687766, 0.580629033777, 0.170631439426, 0.786686768458,
      7.63873279537, 13.1944344379, 11.896042354, 10.5830172417, 10.5020942233,
      92.8918587747, 95.003720371, 86.3715926467, 96.0330217672, 82.6389930677,
      968.702906754, 969.463546828, 1001.79726022, 955.047416547, 1044.27458568;

  expected_out_array << 7.42424742367e-23, 1.02004297287e-34, 0.0130155240738,
      1.02004297287e-34, 1.02004297287e-34, 1.96505168277e-13, 0.525575786243,
      0.713903991771, 2.32077561808e-14, 0.000179348049886, 0.635500453302,
      1.27561284917, 0.878125852156, 0.41565819538, 1.03606488534,
      0.885964824887, 1.16424049334, 1.10764479598, 1.04590810812,
      1.04193666963, 0.965193152414, 0.976217589464, 0.93008035061,
      0.98153216096, 0.909196397698, 0.98434963993, 0.984738050206,
      1.00106492525, 0.97734200649, 1.02198794179;

  for (int i = 0; i < 30; ++i) {
    in_alpha(i) = in_alpha_array(i);
    in_sample(i) = in_sample_array(i);
    expected_out(i) = expected_out_array(i);
  }

  std::size_t bytes = in_alpha.size() * sizeof(Scalar);

  Scalar* d_alpha;
  Scalar* d_sample;
  Scalar* d_out;
  gpuMalloc((void**)(&d_alpha), bytes);
  gpuMalloc((void**)(&d_sample), bytes);
  gpuMalloc((void**)(&d_out), bytes);

  gpuMemcpy(d_alpha, in_alpha.data(), bytes, gpuMemcpyHostToDevice);
  gpuMemcpy(d_sample, in_sample.data(), bytes, gpuMemcpyHostToDevice);

  Eigen::GpuStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);

  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_alpha(d_alpha, 30);
  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_sample(d_sample, 30);
  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_out(d_out, 30);

  gpu_out.device(gpu_device) = gpu_alpha.gamma_sample_der_alpha(gpu_sample);

  assert(gpuMemcpyAsync(out.data(), d_out, bytes, gpuMemcpyDeviceToHost,
                         gpu_device.stream()) == gpuSuccess);
  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);

  for (int i = 0; i < 30; ++i) {
    VERIFY_IS_APPROX(out(i), expected_out(i));
  }

  gpuFree(d_alpha);
  gpuFree(d_sample);
  gpuFree(d_out);
}

template <typename Scalar>
void test_gpu_dawsn()
{
  Tensor<Scalar, 1> in_x(60);
  Tensor<Scalar, 1> out(60);
  Tensor<Scalar, 1> expected_out(60);
  out.setZero();

  Array<Scalar, 1, Dynamic> in_x_array(60);
  Array<Scalar, 1, Dynamic> expected_out_array(60);

  // Compare against scipy.special.dawsn.

  in_x_array << -30.        , -28.98305085, -27.96610169, -26.94915254,
       -25.93220339, -24.91525424, -23.89830508, -22.88135593,
       -21.86440678, -20.84745763, -19.83050847, -18.81355932,
       -17.79661017, -16.77966102, -15.76271186, -14.74576271,
       -13.72881356, -12.71186441, -11.69491525, -10.6779661 ,
        -9.66101695,  -8.6440678 ,  -7.62711864,  -6.61016949,
        -5.59322034,  -4.57627119,  -3.55932203,  -2.54237288,
        -1.52542373,  -0.50847458,   0.50847458,   1.52542373,
         2.54237288,   3.55932203,   4.57627119,   5.59322034,
         6.61016949,   7.62711864,   8.6440678 ,   9.66101695,
        10.6779661 ,  11.69491525,  12.71186441,  13.72881356,
        14.74576271,  15.76271186,  16.77966102,  17.79661017,
        18.81355932,  19.83050847,  20.84745763,  21.86440678,
        22.88135593,  23.89830508,  24.91525424,  25.93220339,
        26.94915254,  27.96610169,  28.98305085,  30.;
  expected_out_array << -0.01667594, -0.01726175, -0.01789024, -0.01856626, -0.01929541,
      -0.02008423, -0.02094035, -0.02187278, -0.02289221, -0.02401143,
      -0.02524586, -0.02661428, -0.0281398 , -0.02985118, -0.03178465,
      -0.03398656, -0.03651715, -0.03945619, -0.04291167, -0.0470335 ,
      -0.05203622, -0.05823825, -0.0661342 , -0.07653816, -0.09089738,
      -0.11208184, -0.14686808, -0.21830822, -0.42101208, -0.4292628 ,
       0.4292628 ,  0.42101208,  0.21830822,  0.14686808,  0.11208184,
       0.09089738,  0.07653816,  0.0661342 ,  0.05823825,  0.05203622,
       0.0470335 ,  0.04291167,  0.03945619,  0.03651715,  0.03398656,
       0.03178465,  0.02985118,  0.0281398 ,  0.02661428,  0.02524586,
       0.02401143,  0.02289221,  0.02187278,  0.02094035,  0.02008423,
       0.01929541,  0.01856626,  0.01789024,  0.01726175,  0.01667594;


  for (int i = 0; i < 60; ++i) {
    in_x(i) = in_x_array(i);
    expected_out(i) = expected_out_array(i);
  }

  std::size_t bytes = in_x.size() * sizeof(Scalar);

  Scalar* d_in;
  Scalar* d_out;
  gpuMalloc((void**)(&d_in), bytes);
  gpuMalloc((void**)(&d_out), bytes);

  gpuMemcpy(d_in, in_x.data(), bytes, gpuMemcpyHostToDevice);

  Eigen::GpuStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);

  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_in(d_in, 60);
  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_out(d_out, 60);

  gpu_out.device(gpu_device) = gpu_in.dawsn();

  assert(gpuMemcpyAsync(out.data(), d_out, bytes, gpuMemcpyDeviceToHost,
                         gpu_device.stream()) == gpuSuccess);
  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);

  for (int i = 0; i < 60; ++i) {
    VERIFY_IS_APPROX(out(i), expected_out(i));
  }

  gpuFree(d_in);
  gpuFree(d_out);
}

template <typename Scalar>
void test_gpu_expi()
{
  Tensor<Scalar, 1> in_x(83);
  Tensor<Scalar, 1> out(83);
  Tensor<Scalar, 1> expected_out(83);
  out.setZero();

  Array<Scalar, 1, Dynamic> in_x_array(83);
  Array<Scalar, 1, Dynamic> expected_out_array(83);

  in_x_array <<  0.        ,   1.07563025,   2.1512605 ,   3.22689076,
         4.30252101,   5.37815126,   6.45378151,   7.52941176,
         8.60504202,   9.68067227,  10.75630252,  11.83193277,
        12.90756303,  13.98319328,  15.05882353,  16.13445378,
        17.21008403,  18.28571429,  19.36134454,  20.43697479,
        21.51260504,  22.58823529,  23.66386555,  24.7394958 ,
        25.81512605,  26.8907563 ,  27.96638655,  29.04201681,
        30.11764706,  31.19327731,  32.26890756,  33.34453782,
        34.42016807,  35.49579832,  36.57142857,  37.64705882,
        38.72268908,  39.79831933,  40.87394958,  41.94957983,
        43.02521008,  44.10084034,  45.17647059,  46.25210084,
        47.32773109,  48.40336134,  49.4789916 ,  50.55462185,
        51.6302521 ,  52.70588235,  53.78151261,  54.85714286,
        55.93277311,  57.00840336,  58.08403361,  59.15966387,
        60.23529412,  61.31092437,  62.38655462,  63.46218487,
        64.53781513,  65.61344538,  66.68907563,  67.76470588,
        68.84033613,  69.91596639,  70.99159664,  72.06722689,
        73.14285714,  74.21848739,  75.29411765,  76.3697479 ,
        77.44537815,  78.5210084 ,  79.59663866,  80.67226891,
        81.74789916,  82.82352941,  83.89915966,  84.97478992,
        86.05042017,  87.12605042,  88.20168067;

  expected_out_array <<   -plusinf, 2.10089121e+00, 5.53528985e+00, 1.15753783e+01,
       2.42706304e+01, 5.33055760e+01, 1.23108954e+02, 2.96569919e+02,
       7.37905242e+02, 1.88081172e+03, 4.88150242e+03, 1.28468967e+04,
       3.41823396e+04, 9.17600177e+04, 2.48134005e+05, 6.75145588e+05,
       1.84670692e+06, 5.07434607e+06, 1.39989488e+07, 3.87561319e+07,
       1.07633111e+08, 2.99757166e+08, 8.36932884e+08, 2.34210043e+09,
       6.56787329e+09, 1.84531523e+10, 5.19367514e+10, 1.46412247e+11,
       4.13357022e+11, 1.16861448e+12, 3.30805712e+12, 9.37546109e+12,
       2.66008817e+13, 7.55531072e+13, 2.14799634e+14, 6.11243402e+14,
       1.74088881e+15, 4.96229772e+15, 1.41556570e+16, 4.04105470e+16,
       1.15441080e+17, 3.29998307e+17, 9.43918678e+17, 2.70156782e+18,
       7.73648003e+18, 2.21669450e+19, 6.35466415e+19, 1.82261026e+20,
       5.22998247e+20, 1.50142352e+21, 4.31215780e+21, 1.23898742e+22,
       3.56133479e+22, 1.02406049e+23, 2.94577085e+23, 8.47670055e+23,
       2.44007728e+24, 7.02625342e+24, 2.02386922e+25, 5.83142468e+25,
       1.68072212e+26, 4.84553619e+26, 1.39736005e+27, 4.03080178e+27,
       1.16302074e+28, 3.35654768e+28, 9.68956171e+28, 2.79780963e+29,
       8.08038054e+29, 2.33422190e+30, 6.74444651e+30, 1.94913373e+31,
       5.63411260e+31, 1.62890401e+32, 4.71030651e+32, 1.36233614e+33,
       3.94092917e+33, 1.14022387e+34, 3.29956549e+34, 9.54985039e+34,
       2.76444343e+35, 8.00365528e+35, 2.31759127e+36;



  for (int i = 0; i < 83; ++i) {
    in_x(i) = in_x_array(i);
    expected_out(i) = expected_out_array(i);
  }

  std::size_t bytes = in_x.size() * sizeof(Scalar);

  Scalar* d_in;
  Scalar* d_out;
  gpuMalloc((void**)(&d_in), bytes);
  gpuMalloc((void**)(&d_out), bytes);

  gpuMemcpy(d_in, in_x.data(), bytes, gpuMemcpyHostToDevice);

  Eigen::GpuStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);

  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_in(d_in, 83);
  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_out(d_out, 83);

  gpu_out.device(gpu_device) = gpu_in.expi();

  assert(gpuMemcpyAsync(out.data(), d_out, bytes, gpuMemcpyDeviceToHost,
                         gpu_device.stream()) == gpuSuccess);
  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);

  for (int i = 0; i < 83; ++i) {
    VERIFY_IS_APPROX(out(i), expected_out(i));
  }

  gpuFree(d_in);
  gpuFree(d_out);
}

template <typename Scalar>
void test_gpu_fresnel()
{
  Tensor<Scalar, 1> in_x(120);
  Tensor<Scalar, 1> out1(120);
  Tensor<Scalar, 1> expected_out1(120);
  Tensor<Scalar, 1> out2(120);
  Tensor<Scalar, 1> expected_out2(120);
  out.setZero();

  Array<Scalar, 1, Dynamic> in_x_array(120);
  Array<Scalar, 1, Dynamic> expected_out1_array(120);
  Array<Scalar, 1, Dynamic> expected_out2_array(120);

  in_x_array << 0.        ,   0.84033613,   1.68067227,   2.5210084 ,
         3.36134454,   4.20168067,   5.04201681,   5.88235294,
         6.72268908,   7.56302521,   8.40336134,   9.24369748,
        10.08403361,  10.92436975,  11.76470588,  12.60504202,
        13.44537815,  14.28571429,  15.12605042,  15.96638655,
        16.80672269,  17.64705882,  18.48739496,  19.32773109,
        20.16806723,  21.00840336,  21.8487395 ,  22.68907563,
        23.52941176,  24.3697479 ,  25.21008403,  26.05042017,
        26.8907563 ,  27.73109244,  28.57142857,  29.41176471,
        30.25210084,  31.09243697,  31.93277311,  32.77310924,
        33.61344538,  34.45378151,  35.29411765,  36.13445378,
        36.97478992,  37.81512605,  38.65546218,  39.49579832,
        40.33613445,  41.17647059,  42.01680672,  42.85714286,
        43.69747899,  44.53781513,  45.37815126,  46.21848739,
        47.05882353,  47.89915966,  48.7394958 ,  49.57983193,
        50.42016807,  51.2605042 ,  52.10084034,  52.94117647,
        53.78151261,  54.62184874,  55.46218487,  56.30252101,
        57.14285714,  57.98319328,  58.82352941,  59.66386555,
        60.50420168,  61.34453782,  62.18487395,  63.02521008,
        63.86554622,  64.70588235,  65.54621849,  66.38655462,
        67.22689076,  68.06722689,  68.90756303,  69.74789916,
        70.58823529,  71.42857143,  72.26890756,  73.1092437 ,
        73.94957983,  74.78991597,  75.6302521 ,  76.47058824,
        77.31092437,  78.1512605 ,  78.99159664,  79.83193277,
        80.67226891,  81.51260504,  82.35294118,  83.19327731,
        84.03361345,  84.87394958,  85.71428571,  86.55462185,
        87.39495798,  88.23529412,  89.07563025,  89.91596639,
        90.75630252,  91.59663866,  92.43697479,  93.27731092,
        94.11764706,  94.95798319,  95.79831933,  96.63865546,
        97.4789916 ,  98.31932773,  99.15966387, 100.;

  expected_out1_array << 0.        , 0.74266522, 0.32812213, 0.43875588, 0.41453588,
       0.54029029, 0.55022786, 0.45642045, 0.54524838, 0.54010938,
       0.46888788, 0.5264084 , 0.51495646, 0.47492414, 0.48386262,
       0.47515281, 0.52223741, 0.50281483, 0.51997787, 0.4802031 ,
       0.48735932, 0.4857135 , 0.50575129, 0.51048487, 0.48541441,
       0.51288696, 0.51221434, 0.48670058, 0.50737654, 0.50236137,
       0.49177003, 0.48984997, 0.48834718, 0.51147597, 0.50546322,
       0.51078691, 0.48994025, 0.49060812, 0.49549969, 0.49883568,
       0.50201447, 0.49080629, 0.50441198, 0.50401653, 0.49158384,
       0.5002166 , 0.49691317, 0.49896422, 0.4921087 , 0.49454764,
       0.5060444 , 0.50679103, 0.50539112, 0.49595384, 0.4932533 ,
       0.50159178, 0.49497724, 0.49671284, 0.49566851, 0.49840677,
       0.49811273, 0.49666623, 0.49569701, 0.49438201, 0.50385107,
       0.49618958, 0.5004852 , 0.50023254, 0.50493898, 0.49956376,
       0.50173312, 0.49816757, 0.50488652, 0.49495883, 0.49489211,
       0.5013866 , 0.49524103, 0.4952145 , 0.50225019, 0.49538438,
       0.49642285, 0.50455173, 0.50178249, 0.50426757, 0.49598558,
       0.49971476, 0.49582196, 0.50434592, 0.50323022, 0.50285698,
       0.49957101, 0.49841209, 0.50411503, 0.49770791, 0.49801616,
       0.50389467, 0.50009259, 0.50179876, 0.49995817, 0.50375684,
       0.50198849, 0.49773544, 0.49630356, 0.49834349, 0.50069004,
       0.50267925, 0.49760382, 0.5034784 , 0.50314093, 0.50030427,
       0.50276759, 0.50292828, 0.49930655, 0.50335069, 0.50291682,
       0.49670977, 0.49921886, 0.49713836, 0.50270743, 0.4999999;

  expected_out2_array << 0.        , 0.28445073, 0.56803166, 0.60957677, 0.45964101,
       0.56408363, 0.5382052 , 0.53205885, 0.51392546, 0.51274102,
       0.52160323, 0.52209767, 0.5277966 , 0.48516185, 0.52171661,
       0.50450528, 0.49187834, 0.47789696, 0.49338769, 0.50235243,
       0.51410366, 0.48898874, 0.51622867, 0.51270028, 0.50602981,
       0.50796839, 0.50794099, 0.50446585, 0.51134009, 0.51284645,
       0.49042447, 0.50680298, 0.49791918, 0.50023842, 0.49029065,
       0.50087741, 0.49691587, 0.50407426, 0.49110558, 0.50964249,
       0.50925297, 0.49908881, 0.50786592, 0.50784007, 0.49818886,
       0.50841474, 0.50763407, 0.4920075 , 0.49995515, 0.49451998,
       0.50456702, 0.4969924 , 0.5048988 , 0.49410869, 0.49807991,
       0.49329941, 0.50453042, 0.50577548, 0.49511225, 0.50621932,
       0.50602445, 0.49476113, 0.50433707, 0.50214211, 0.49550569,
       0.49559084, 0.49428132, 0.50564878, 0.50257605, 0.50547233,
       0.49487378, 0.49498951, 0.49805076, 0.49877073, 0.50033348,
       0.49514355, 0.50148089, 0.50113966, 0.49569652, 0.49870148,
       0.49689792, 0.50107262, 0.49573839, 0.49838279, 0.50205403,
       0.5044472 , 0.50139418, 0.49973669, 0.49715506, 0.50315462,
       0.49581316, 0.49615226, 0.49986416, 0.49663316, 0.49649249,
       0.50085425, 0.49605537, 0.49653391, 0.50386497, 0.50072498,
       0.50322397, 0.49701049, 0.50035675, 0.49671664, 0.50357624,
       0.50241573, 0.50265105, 0.49934204, 0.49843933, 0.50346178,
       0.49795101, 0.49824773, 0.50331019, 0.50009766, 0.50159141,
       0.49984635, 0.50317061, 0.5015141 , 0.49827535, 0.4968169;


  for (int i = 0; i < 120; ++i) {
    in_x(i) = in_x_array(i);
    expected_out1(i) = expected_out1_array(i);
    expected_out2(i) = expected_out2_array(i);
  }

  std::size_t bytes = in_x.size() * sizeof(Scalar);

  Scalar* d_in;
  Scalar* d_out1;
  Scalar* d_out2;
  gpuMalloc((void**)(&d_in), bytes);
  gpuMalloc((void**)(&d_out1), bytes);
  gpuMalloc((void**)(&d_out2), bytes);

  gpuMemcpy(d_in, in_x.data(), bytes, gpuMemcpyHostToDevice);

  Eigen::GpuStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);

  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_in(d_in, 120);
  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_out1(d_out1, 120);
  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_out2(d_out2, 120);

  gpu_out1.device(gpu_device) = gpu_in.fresnel_cos();
  gpu_out2.device(gpu_device) = gpu_in.fresnel_sin();

  assert(gpuMemcpyAsync(out1.data(), d_out1, bytes, gpuMemcpyDeviceToHost,
                         gpu_device.stream()) == gpuSuccess);
  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);
  assert(gpuMemcpyAsync(out2.data(), d_out2, bytes, gpuMemcpyDeviceToHost,
                         gpu_device.stream()) == gpuSuccess);
  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);

  for (int i = 0; i < 120; ++i) {
    VERIFY_IS_APPROX(out1(i), expected_out1(i));
    VERIFY_IS_APPROX(out2(i), expected_out2(i));
  }

  gpuFree(d_in);
  gpuFree(d_out1);
  gpuFree(d_out2);
}

template <typename Scalar>
void test_gpu_spence()
{
  Tensor<Scalar, 1> in_x(120);
  Tensor<Scalar, 1> out(120);
  Tensor<Scalar, 1> expected_out(120);
  out.setZero();

  Array<Scalar, 1, Dynamic> in_x_array(120);
  Array<Scalar, 1, Dynamic> expected_out_array(120);

  in_x_array <<   0.        ,   1.07563025,   2.1512605 ,   3.22689076,
         4.30252101,   5.37815126,   6.45378151,   7.52941176,
         8.60504202,   9.68067227,  10.75630252,  11.83193277,
        12.90756303,  13.98319328,  15.05882353,  16.13445378,
        17.21008403,  18.28571429,  19.36134454,  20.43697479,
        21.51260504,  22.58823529,  23.66386555,  24.7394958 ,
        25.81512605,  26.8907563 ,  27.96638655,  29.04201681,
        30.11764706,  31.19327731,  32.26890756,  33.34453782,
        34.42016807,  35.49579832,  36.57142857,  37.64705882,
        38.72268908,  39.79831933,  40.87394958,  41.94957983,
        43.02521008,  44.10084034,  45.17647059,  46.25210084,
        47.32773109,  48.40336134,  49.4789916 ,  50.55462185,
        51.6302521 ,  52.70588235,  53.78151261,  54.85714286,
        55.93277311,  57.00840336,  58.08403361,  59.15966387,
        60.23529412,  61.31092437,  62.38655462,  63.46218487,
        64.53781513,  65.61344538,  66.68907563,  67.76470588,
        68.84033613,  69.91596639,  70.99159664,  72.06722689,
        73.14285714,  74.21848739,  75.29411765,  76.3697479 ,
        77.44537815,  78.5210084 ,  79.59663866,  80.67226891,
        81.74789916,  82.82352941,  83.89915966,  84.97478992,
        86.05042017,  87.12605042,  88.20168067,  89.27731092,
        90.35294118,  91.42857143,  92.50420168,  93.57983193,
        94.65546218,  95.73109244,  96.80672269,  97.88235294,
        98.95798319, 100.03361345, 101.1092437 , 102.18487395,
       103.2605042 , 104.33613445, 105.41176471, 106.48739496,
       107.56302521, 108.63865546, 109.71428571, 110.78991597,
       111.86554622, 112.94117647, 114.01680672, 115.09243697,
       116.16806723, 117.24369748, 118.31932773, 119.39495798,
       120.47058824, 121.54621849, 122.62184874, 123.69747899,
       124.77310924, 125.8487395 , 126.92436975, 128.;

  expected_out_array <<   1.64493407,  -0.07424638,  -0.92517856,  -1.55869701,
        -2.07605073,  -2.5186072 ,  -2.908091  ,  -3.25755955,
        -3.57556745,  -3.86806807,  -4.13939443,  -4.3928116 ,
        -4.63084902,  -4.85551113,  -5.06841659,  -5.27089367,
        -5.46404732,  -5.64880782,  -5.82596653,  -5.99620293,
        -6.16010524,  -6.31818641,  -6.47089677,  -6.61863403,
        -6.76175137,  -6.900564  ,  -7.03535452,  -7.16637741,
        -7.29386268,  -7.41801901,  -7.53903635,  -7.65708816,
        -7.7723333 ,  -7.88491766,  -7.99497556,  -8.10263099,
        -8.20799864,  -8.31118484,  -8.41228837,  -8.51140117,
        -8.608609  ,  -8.70399192,  -8.79762489,  -8.88957813,
        -8.97991756,  -9.06870512,  -9.15599912,  -9.24185452,
        -9.32632318,  -9.40945407,  -9.49129353,  -9.57188543,
        -9.65127137,  -9.7294908 ,  -9.8065812 ,  -9.8825782 ,
        -9.95751571, -10.03142601, -10.10433989, -10.17628671,
       -10.2472945 , -10.31739005, -10.38659896, -10.45494576,
       -10.5224539 , -10.58914586, -10.65504321, -10.72016663,
       -10.78453597, -10.84817031, -10.91108798, -10.97330659,
       -11.03484312, -11.0957139 , -11.15593464, -11.2155205 ,
       -11.2744861 , -11.33284553, -11.39061239, -11.44779982,
       -11.50442049, -11.56048667, -11.6160102 , -11.67100254,
       -11.72547479, -11.77943766, -11.83290156, -11.88587655,
       -11.93837237, -11.99039848, -12.04196404, -12.09307795,
       -12.14374883, -12.19398505, -12.24379474, -12.2931858 ,
       -12.34216589, -12.39074247, -12.43892277, -12.48671386,
       -12.53412257, -12.58115558, -12.62781937, -12.67412026,
       -12.72006439, -12.76565776, -12.81090619, -12.85581537,
       -12.90039084, -12.94463799, -12.98856211, -13.03216831,
       -13.07546161, -13.1184469 , -13.16112895, -13.20351241,
       -13.24560183, -13.28740166, -13.32891622, -13.37014976;

  for (int i = 0; i < 120; ++i) {
    in_x(i) = in_x_array(i);
    expected_out(i) = expected_out_array(i);
  }

  std::size_t bytes = in_x.size() * sizeof(Scalar);

  Scalar* d_in;
  Scalar* d_out;
  gpuMalloc((void**)(&d_in), bytes);
  gpuMalloc((void**)(&d_out), bytes);

  gpuMemcpy(d_in, in_x.data(), bytes, gpuMemcpyHostToDevice);

  Eigen::GpuStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);

  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_in(d_in, 120);
  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_out(d_out, 120);

  gpu_out.device(gpu_device) = gpu_in.spence();

  assert(gpuMemcpyAsync(out.data(), d_out, bytes, gpuMemcpyDeviceToHost,
                         gpu_device.stream()) == gpuSuccess);
  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);

  for (int i = 0; i < 120; ++i) {
    VERIFY_IS_APPROX(out(i), expected_out(i));
  }

  gpuFree(d_in);
  gpuFree(d_out);
}


EIGEN_DECLARE_TEST(cxx11_tensor_gpu)
{
  CALL_SUBTEST_1(test_gpu_nullary());
  CALL_SUBTEST_1(test_gpu_elementwise_small());
  CALL_SUBTEST_1(test_gpu_elementwise());
  CALL_SUBTEST_1(test_gpu_props());
  CALL_SUBTEST_1(test_gpu_reduction());
  CALL_SUBTEST_2(test_gpu_contraction<ColMajor>());
  CALL_SUBTEST_2(test_gpu_contraction<RowMajor>());
  CALL_SUBTEST_3(test_gpu_convolution_1d<ColMajor>());
  CALL_SUBTEST_3(test_gpu_convolution_1d<RowMajor>());
  CALL_SUBTEST_3(test_gpu_convolution_inner_dim_col_major_1d());
  CALL_SUBTEST_3(test_gpu_convolution_inner_dim_row_major_1d());
  CALL_SUBTEST_3(test_gpu_convolution_2d<ColMajor>());
  CALL_SUBTEST_3(test_gpu_convolution_2d<RowMajor>());
#if !defined(EIGEN_USE_HIP)
// disable these tests on HIP for now.
// they hang..need to investigate and fix
  CALL_SUBTEST_3(test_gpu_convolution_3d<ColMajor>());
  CALL_SUBTEST_3(test_gpu_convolution_3d<RowMajor>());
#endif

#if EIGEN_GPU_TEST_C99_MATH
  // std::erf, std::erfc, and so on where only added in c++11. We use them
  // as a golden reference to validate the results produced by Eigen. Therefore
  // we can only run these tests if we use a c++11 compiler.
  CALL_SUBTEST_4(test_gpu_lgamma<float>(1.0f));
  CALL_SUBTEST_4(test_gpu_lgamma<float>(100.0f));
  CALL_SUBTEST_4(test_gpu_lgamma<float>(0.01f));
  CALL_SUBTEST_4(test_gpu_lgamma<float>(0.001f));

  CALL_SUBTEST_4(test_gpu_lgamma<double>(1.0));
  CALL_SUBTEST_4(test_gpu_lgamma<double>(100.0));
  CALL_SUBTEST_4(test_gpu_lgamma<double>(0.01));
  CALL_SUBTEST_4(test_gpu_lgamma<double>(0.001));

  CALL_SUBTEST_4(test_gpu_erf<float>(1.0f));
  CALL_SUBTEST_4(test_gpu_erf<float>(100.0f));
  CALL_SUBTEST_4(test_gpu_erf<float>(0.01f));
  CALL_SUBTEST_4(test_gpu_erf<float>(0.001f));

  CALL_SUBTEST_4(test_gpu_erfc<float>(1.0f));
  // CALL_SUBTEST(test_gpu_erfc<float>(100.0f));
  CALL_SUBTEST_4(test_gpu_erfc<float>(5.0f)); // GPU erfc lacks precision for large inputs
  CALL_SUBTEST_4(test_gpu_erfc<float>(0.01f));
  CALL_SUBTEST_4(test_gpu_erfc<float>(0.001f));

  CALL_SUBTEST_4(test_gpu_erf<double>(1.0));
  CALL_SUBTEST_4(test_gpu_erf<double>(100.0));
  CALL_SUBTEST_4(test_gpu_erf<double>(0.01));
  CALL_SUBTEST_4(test_gpu_erf<double>(0.001));

  CALL_SUBTEST_4(test_gpu_erfc<double>(1.0));
  // CALL_SUBTEST(test_gpu_erfc<double>(100.0));
  CALL_SUBTEST_4(test_gpu_erfc<double>(5.0)); // GPU erfc lacks precision for large inputs
  CALL_SUBTEST_4(test_gpu_erfc<double>(0.01));
  CALL_SUBTEST_4(test_gpu_erfc<double>(0.001));

#if !defined(EIGEN_USE_HIP)
// disable these tests on HIP for now.

  CALL_SUBTEST_5(test_gpu_ndtri<float>());
  CALL_SUBTEST_5(test_gpu_ndtri<double>());

  CALL_SUBTEST_5(test_gpu_digamma<float>());
  CALL_SUBTEST_5(test_gpu_digamma<double>());

  CALL_SUBTEST_5(test_gpu_polygamma<float>());
  CALL_SUBTEST_5(test_gpu_polygamma<double>());

  CALL_SUBTEST_5(test_gpu_zeta<float>());
  CALL_SUBTEST_5(test_gpu_zeta<double>());
#endif

  CALL_SUBTEST_5(test_gpu_igamma<float>());
  CALL_SUBTEST_5(test_gpu_igammac<float>());

  CALL_SUBTEST_5(test_gpu_igamma<double>());
  CALL_SUBTEST_5(test_gpu_igammac<double>());

#if !defined(EIGEN_USE_HIP)
// disable these tests on HIP for now.
  CALL_SUBTEST_6(test_gpu_betainc<float>());
  CALL_SUBTEST_6(test_gpu_betainc<double>());

  CALL_SUBTEST_6(test_gpu_i0e<float>());
  CALL_SUBTEST_6(test_gpu_i0e<double>());

  CALL_SUBTEST_6(test_gpu_i1e<float>());
  CALL_SUBTEST_6(test_gpu_i1e<double>());

  CALL_SUBTEST_6(test_gpu_i1e<float>());
  CALL_SUBTEST_6(test_gpu_i1e<double>());

  CALL_SUBTEST_6(test_gpu_igamma_der_a<float>());
  CALL_SUBTEST_6(test_gpu_igamma_der_a<double>());

  CALL_SUBTEST_6(test_gpu_gamma_sample_der_alpha<float>());
  CALL_SUBTEST_6(test_gpu_gamma_sample_der_alpha<double>());
#endif

#if !defined(EIGEN_USE_HIP)
// disable these tests on HIP for now.
  CALL_SUBTEST_7(test_gpu_dawsn<float>());
  CALL_SUBTEST_7(test_gpu_dawsn<double>());

  CALL_SUBTEST_7(test_gpu_expi<float>());
  CALL_SUBTEST_7(test_gpu_expi<double>());

  CALL_SUBTEST_7(test_gpu_fresnel<float>());
  CALL_SUBTEST_7(test_gpu_fresnel<double>());

  CALL_SUBTEST_7(test_gpu_spence<float>());
  CALL_SUBTEST_7(test_gpu_spence<double>());
#endif


#endif
}
