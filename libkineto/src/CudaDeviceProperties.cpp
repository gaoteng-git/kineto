/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "CudaDeviceProperties.h"

#include <vector>

#include <cuda_runtime.h>
#include <cuda_occupancy.h>

namespace KINETO_NAMESPACE {

std::vector<cudaOccDeviceProp> occProps;

void initOccDeviceProp() {
  occProps.clear();
  int device_count;
  cudaError_t error_id = cudaGetDeviceCount(&device_count);
  // Return empty vector if error.
  if (error_id != cudaSuccess) {
    return;
  }
  for (int i = 0; i < device_count; ++i) {
    cudaDeviceProp prop;
    error_id = cudaGetDeviceProperties(&prop, i);
    // Return empty vector if any device property fail to get.
    if (error_id != cudaSuccess) {
      return;
    }
    cudaOccDeviceProp occProp;
    occProp = prop;
    occProps.push_back(occProp);
  }
}

float getKernelOccupancy(uint32_t deviceId, uint16_t registersPerThread, 
                         int32_t staticSharedMemory, int32_t dynamicSharedMemory,
                         int32_t blockX, int32_t blockY, int32_t blockZ) {
  // Calculate occupancy
  float occupancy = -1.0;
  if (deviceId < occProps.size()) {
    cudaOccFuncAttributes occFuncAttr;
    occFuncAttr.maxThreadsPerBlock = INT_MAX;
    occFuncAttr.numRegs = registersPerThread;
    occFuncAttr.sharedSizeBytes = staticSharedMemory;
    occFuncAttr.partitionedGCConfig = PARTITIONED_GC_OFF;
    occFuncAttr.shmemLimitConfig = FUNC_SHMEM_LIMIT_DEFAULT;
    occFuncAttr.maxDynamicSharedSizeBytes = 0;
    const cudaOccDeviceState occDeviceState = {};
    int blockSize = blockX * blockY * blockZ;
    size_t dynamicSmemSize = dynamicSharedMemory;
    cudaOccResult occ_result;
    cudaOccError status = cudaOccMaxActiveBlocksPerMultiprocessor(
          &occ_result, &occProps[deviceId], &occFuncAttr, &occDeviceState,
          blockSize, dynamicSmemSize);
    if (status == CUDA_OCC_SUCCESS) {
      occupancy = occ_result.activeBlocksPerMultiprocessor * blockSize /
          (float) occProps[deviceId].maxThreadsPerMultiprocessor;
    }
  }
  return occupancy;
}

} // namespace KINETO_NAMESPACE