/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <vector>

namespace KINETO_NAMESPACE {

#ifdef HAS_CUPTI
std::vector<cudaOccDeviceProp> getOccDeviceProp();

void initOccDeviceProps();
#endif

float getKernelOccupancy(CUpti_ActivityKernel4* kernel);

} // namespace KINETO_NAMESPACE