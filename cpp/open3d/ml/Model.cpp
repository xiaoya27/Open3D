// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/ml/Model.h"

#include <cstdlib>
#include <mutex>
#include <sstream>
#include <utility>
#include <vector>

#include "open3d/utility/Logging.h"

#if OPEN3D_BUILD_PYTORCH_OPS
#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#endif

namespace open3d {
namespace ml {
namespace {

#if OPEN3D_BUILD_PYTORCH_OPS

class LibtorchLoader {
public:
    void EnsureLoaded();

    void* GetSymbol(const char* name) const;

    bool IsLoaded() const { return handle_ != nullptr; }

    const std::string& LoadedPath() const { return loaded_path_; }

private:
#if defined(_WIN32)
    using HandleType = HMODULE;
#else
    using HandleType = void*;
#endif

    static std::vector<std::string> CandidateLibraries();

    HandleType handle_ = nullptr;
    std::string loaded_path_;
    std::once_flag load_once_;
};

#if defined(_WIN32)
void LibtorchLoader::EnsureLoaded() {
    std::call_once(load_once_, [this]() {
        utility::LogError(
                "Dynamic libtorch loading on Windows is not implemented yet.");
    });
}

void* LibtorchLoader::GetSymbol(const char* name) const {
    utility::LogError(
            "Dynamic libtorch loading on Windows is not implemented yet "
            "(missing symbol {}).",
            name);
    return nullptr;
}
#else   // POSIX platforms

std::vector<std::string> LibtorchLoader::CandidateLibraries() {
    std::vector<std::string> candidates;
    if (const char* env = std::getenv("OPEN3D_LIBTORCH_PATH")) {
        candidates.emplace_back(env);
    }
    candidates.emplace_back("libtorch_cpu.so");
    candidates.emplace_back("libtorch_cpu.so.1");
    candidates.emplace_back("libtorch.so");
    return candidates;
}

void LibtorchLoader::EnsureLoaded() {
    std::call_once(load_once_, [this]() {
        auto candidates = CandidateLibraries();
        std::vector<std::pair<std::string, std::string>> failures;
        for (const auto& candidate : candidates) {
            dlerror();  // Clear any stale state.
            void* handle = dlopen(candidate.c_str(), RTLD_LAZY | RTLD_LOCAL);
            if (handle) {
                handle_ = handle;
                loaded_path_ = candidate;
                break;
            }
            if (const char* err = dlerror()) {
                failures.emplace_back(candidate, std::string(err));
            } else {
                failures.emplace_back(candidate,
                                      "dlopen returned nullptr without error");
            }
        }

        if (!handle_) {
            std::stringstream msg;
            msg << "Unable to locate libtorch shared library. "
                   "Set OPEN3D_LIBTORCH_PATH or update LD_LIBRARY_PATH. Tried:";
            for (const auto& failure : failures) {
                msg << "\n  - " << failure.first << ": " << failure.second;
            }
            utility::LogError("{}", msg.str());
        } else {
            utility::LogInfo("libtorch loaded from {}", loaded_path_);
        }
    });
}

void* LibtorchLoader::GetSymbol(const char* name) const {
    if (!handle_) {
        utility::LogError(
                "Attempted to resolve '{}' before libtorch was loaded.", name);
    }
    dlerror();
    void* fn = dlsym(handle_, name);
    if (const char* err = dlerror()) {
        utility::LogError("dlsym('{}') failed: {}", name, err);
    }
    return fn;
}
#endif  // defined(_WIN32)

#endif  // OPEN3D_BUILD_PYTORCH_OPS

}  // namespace

struct Model::Impl {
#if OPEN3D_BUILD_PYTORCH_OPS
    LibtorchLoader loader_;
    std::string artifact_path_;
    bool model_loaded_ = false;
#else
    bool model_loaded_ = false;
#endif
};

Model::Model() : impl_(std::make_unique<Impl>()) {}

Model::~Model() = default;

bool IsPyTorchRuntimeEnabled() {
#if OPEN3D_BUILD_PYTORCH_OPS
    return true;
#else
    return false;
#endif
}

void Model::LoadModel(const std::string& artifact_path) {
    if (artifact_path.empty()) {
        utility::LogError("artifact_path cannot be empty.");
    }

#if !OPEN3D_BUILD_PYTORCH_OPS
    utility::LogError(
            "Open3D was built without BUILD_PYTORCH_OPS=ON. "
            "Reconfigure the project to enable open3d::ml::Model.");
    return;
#else
    impl_->loader_.EnsureLoaded();
    impl_->artifact_path_ = artifact_path;
    impl_->model_loaded_ = true;
#endif
}

std::vector<core::Tensor> Model::Forward(
        const std::vector<core::Tensor>& inputs) const {
#if !OPEN3D_BUILD_PYTORCH_OPS
    utility::LogError(
            "Open3D was built without BUILD_PYTORCH_OPS=ON. "
            "Reconfigure the project to enable open3d::ml::Model.");
    return {};
#else
    if (!impl_->model_loaded_) {
        utility::LogError("Call LoadModel() before Forward().");
    }
    (void)inputs;
    utility::LogError(
            "Model::Forward is not implemented yet. This is a placeholder "
            "until the AOTInductor runtime is integrated.");
    return {};
#endif
}

}  // namespace ml
}  // namespace open3d
