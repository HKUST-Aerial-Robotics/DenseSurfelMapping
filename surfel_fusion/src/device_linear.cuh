#pragma once

#include <assert.h>
#include <cuda_runtime.h>
#include <cuda_exception.cuh>

template<typename ElementType>
struct DeviceLinear
{
  __host__
  DeviceLinear(size_t _length)
    : length(_length)
  {
    cudaError err = cudaMalloc(&data, _length * sizeof(ElementType));

    err = cudaMalloc(
          &dev_ptr,
          sizeof(*this));

    err = cudaMemcpy(
          dev_ptr,
          this,
          sizeof(*this),
          cudaMemcpyHostToDevice);
  }

  __device__
  ElementType & operator()(size_t x)
  {
    return at(x);
  }

  __device__
  const ElementType & operator()(size_t x) const
  {
    return at(x);
  }

  __device__
  ElementType &at(size_t x)
  {
    return data[x];
  }

  __device__
  const ElementType &at(size_t x) const
  {
    return data[x];
  }

  /// Upload aligned_data_row_major to device memory
  __host__
  void setDevData(const ElementType *aligned_data)
  {
    const cudaError err = cudaMemcpy(
          data,
          aligned_data,
          length * sizeof(ElementType),
          cudaMemcpyHostToDevice);
  }

  /// Download the data from the device memory to aligned_data_row_major, a preallocated array in host memory
  __host__
  void getDevData(ElementType* aligned_data) const
  {
    const cudaError err = cudaMemcpy(
          aligned_data,
          data,
          length * sizeof(ElementType),
          cudaMemcpyDeviceToHost);
  }

  __host__
  ~DeviceLinear()
  {
    cudaError err = cudaFree(data);
    err = cudaFree(dev_ptr);
  }

  __host__
  cudaChannelFormatDesc getCudaChannelFormatDesc() const
  {
    return cudaCreateChannelDesc<ElementType>();
  }

  __host__
  void zero()
  {
    const cudaError err = cudaMemset(
          data,
          0,
          length*sizeof(ElementType));
  }

  __host__
  DeviceLinear<ElementType> & operator= (const DeviceLinear<ElementType> &other_linear)
  {
    if(this != & other_linear)
    {
      assert(length  == other_linear.length);
      const cudaError err = cudaMemcpy( data,
                                        other_linear.data,
                                        length * sizeof(ElementType),
                                        cudaMemcpyDeviceToDevice);
    }
    return *this;
  }

  // fields
  size_t length;
  ElementType *data;
  DeviceLinear<ElementType> *dev_ptr;
};