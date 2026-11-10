// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdlib>
#include <string>

namespace gz_gpu_ouster_lidar::testutil {

/// Sets GZ_OUSTER_BACKEND for one test and restores the previous value (or
/// unsets it) afterwards, so the choice does not leak into later tests in the
/// same binary. Pass nullptr to run with the variable unset.
class BackendEnvGuard {
public:
    explicit BackendEnvGuard(const char * value)
    {
        const char * prev = std::getenv(kVar);
        had_prev_ = prev != nullptr;
        if (had_prev_) prev_ = prev;
        if (value) {
            ::setenv(kVar, value, 1);
        } else {
            ::unsetenv(kVar);
        }
    }
    ~BackendEnvGuard()
    {
        if (had_prev_) {
            ::setenv(kVar, prev_.c_str(), 1);
        } else {
            ::unsetenv(kVar);
        }
    }
    BackendEnvGuard(const BackendEnvGuard &) = delete;
    BackendEnvGuard & operator=(const BackendEnvGuard &) = delete;

private:
    static constexpr const char * kVar = "GZ_OUSTER_BACKEND";
    bool had_prev_ = false;
    std::string prev_;
};

}  // namespace gz_gpu_ouster_lidar::testutil
