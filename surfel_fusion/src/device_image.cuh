// This file is part of REMODE - REgularized MOnocular Depth Estimation.
//
// Copyright (C) 2014 Matia Pizzoli <matia dot pizzoli at gmail dot com>
// Robotics and Perception Group, University of Zurich, Switzerland
// http://rpg.ifi.uzh.ch
//
// REMODE is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// REMODE is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#pragma once

#include <assert.h>
#include <cuda_runtime.h>
#include <cuda_exception.cuh>
template <typename ElementType>
struct DeviceImage
{
    __host__
    DeviceImage(size_t width, size_t height)
        : width(width),
          height(height)
    {
        cudaError err = cudaMalloc(&data, width*height * sizeof(ElementType));

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
        ElementType &
        operator()(size_t x, size_t y)
    {
        return atXY(x, y);
    }

    __device__ const ElementType &operator()(size_t x, size_t y) const
    {
        return atXY(x, y);
    }

    __device__
        ElementType &
        atXY(size_t x, size_t y)
    {
        return data[y * width + x];
    }

    __device__ const ElementType &atXY(size_t x, size_t y) const
    {
        return data[y * width + x];
    }

    /// Upload aligned_data_row_major to device memory
    __host__ void setDevData(const ElementType *aligned_data_row_major)
    {
        const cudaError err = cudaMemcpy(
            data,
            aligned_data_row_major,
            width * height * sizeof(ElementType),
            cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
            throw CudaException("Image: unable to copy data from host to device.", err);
    }

    /// Download the data from the device memory to aligned_data_row_major, a preallocated array in host memory
    __host__ void getDevData(ElementType *aligned_data_row_major) const
    {
        const cudaError err = cudaMemcpy(
            aligned_data_row_major,
            data,
            width * height * sizeof(ElementType),
            cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
            throw CudaException("Image: unable to copy data from device to host.", err);
    }

    __host__ ~DeviceImage()
    {
        cudaFree(data);
        cudaFree(dev_ptr);
    }

    __host__
        cudaChannelFormatDesc
        getCudaChannelFormatDesc() const
    {
        return cudaCreateChannelDesc<ElementType>();
    }

    __host__ void zero()
    {
        const cudaError err = cudaMemset(
            data,
            0,
            width * height * sizeof(ElementType));
    }

    __host__
        DeviceImage<ElementType> &
        operator=(const DeviceImage<ElementType> &other_image)
    {
        if (this != &other_image)
        {
            assert(width == other_image.width &&
                   height == other_image.height);
            const cudaError err = cudaMemcpy(data,
                                             other_image.data,
                                             width * height * sizeof(ElementType),
                                             cudaMemcpyDeviceToDevice);
        }
        return *this;
    }

    // fields
    size_t width;
    size_t height;
    ElementType *data;
    DeviceImage<ElementType> *dev_ptr;
};