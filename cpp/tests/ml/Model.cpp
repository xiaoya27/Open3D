// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/ml/Model.h"

#include "open3d/core/Tensor.h"
#include "tests/Tests.h"

namespace open3d {
namespace tests {

TEST(Model, Constructor) {
    ml::Model model;
    // Should construct without throwing
}

TEST(Model, IsPyTorchRuntimeEnabled) {
    bool enabled = ml::IsPyTorchRuntimeEnabled();
#if OPEN3D_BUILD_PYTORCH_OPS
    EXPECT_TRUE(enabled);
#else
    EXPECT_FALSE(enabled);
#endif
}

TEST(Model, ForwardWithoutLoad) {
    ml::Model model;
    std::vector<core::Tensor> inputs;
    // Forward without LoadModel should throw/log error
    EXPECT_ANY_THROW(model.Forward(inputs));
}

TEST(Model, LoadModelEmptyPath) {
    ml::Model model;
    // LoadModel with empty path should throw
    EXPECT_ANY_THROW(model.LoadModel(""));
}

}  // namespace tests
}  // namespace open3d
