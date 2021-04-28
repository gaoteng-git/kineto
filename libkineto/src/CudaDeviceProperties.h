/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cuda_occupancy.h>
#include <vector>

namespace KINETO_NAMESPACE {

std::vector<cudaOccDeviceProp> getOccDeviceProp();

void initOccDeviceProps();

float getKernelOccupancy(uint32_t deviceId, uint16_t registersPerThread, 
                         int32_t staticSharedMemory, int32_t dynamicSharedMemory,
                         int32_t blockX, int32_t blockY, int32_t blockZ);

} // namespace KINETO_NAMESPACE