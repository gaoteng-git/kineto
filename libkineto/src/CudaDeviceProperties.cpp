/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cuda_occupancy.h>
#include "CudaDeviceProperties.h"

namespace KINETO_NAMESPACE {

std::vector<cudaOccDeviceProp> occProps_;

#ifdef HAS_CUPTI

std::vector<cudaOccDeviceProp> getOccDeviceProp() {
  std::vector<cudaOccDeviceProp> occProps;
  int device_count;
  cudaError_t error_id = cudaGetDeviceCount(&device_count);
  // Return empty vector if error.
  if (error_id != cudaSuccess) {
    return std::vector<cudaOccDeviceProp>();
  }
  for (int i = 0; i < device_count; ++i) {
    cudaDeviceProp prop;
    error_id = cudaGetDeviceProperties(&prop, i);
    // Return empty vector if any device property fail to get.
    if (error_id != cudaSuccess) {
      return std::vector<cudaOccDeviceProp>();
    }
    cudaOccDeviceProp occProp;
    occProp = prop;
    occProps.push_back(occProp);
  }
  return occProps;
}

void initOccDeviceProps() {
  occProps_ = getOccDeviceProp();
}

#endif

float getKernelOccupancy(CUpti_ActivityKernel4* kernel) {
  // Calculate occupancy
  float occupancy = -1.0;
  if (kernel->deviceId < occProps_.size()) {
    cudaOccFuncAttributes occFuncAttr;
    occFuncAttr.maxThreadsPerBlock = INT_MAX;
    occFuncAttr.numRegs = kernel->registersPerThread;
    occFuncAttr.sharedSizeBytes = kernel->staticSharedMemory;
    occFuncAttr.partitionedGCConfig = PARTITIONED_GC_OFF;
    occFuncAttr.shmemLimitConfig = FUNC_SHMEM_LIMIT_DEFAULT;
    occFuncAttr.maxDynamicSharedSizeBytes = 0;
    const cudaOccDeviceState occDeviceState = {};
    int blockSize = kernel->blockX * kernel->blockY * kernel->blockZ;
    size_t dynamicSmemSize = kernel->dynamicSharedMemory;
    cudaOccResult occ_result;
    cudaOccError status = cudaOccMaxActiveBlocksPerMultiprocessor(
          &occ_result, &occProps_[kernel->deviceId], &occFuncAttr, &occDeviceState,
          blockSize, dynamicSmemSize);
    if (status == CUDA_OCC_SUCCESS) {
      occupancy = occ_result.activeBlocksPerMultiprocessor * blockSize /
          (float) occProps_[kernel->deviceId].maxThreadsPerMultiprocessor;
    }
  }
  return occupancy;
}

} // namespace KINETO_NAMESPACE