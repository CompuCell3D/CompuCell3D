////
//// Created by m on 12/14/2025.
////

#ifndef COMPUCELL3D_RANDOMUTILS_H
#define COMPUCELL3D_RANDOMUTILS_H

#include <algorithm>
#include <random>
#include <cstdint>

namespace CompuCell3D {

// ------------------------------------------------------------
// C++17 feature detection (MSVC + GCC + Clang)
// ------------------------------------------------------------
#if (defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || (__cplusplus >= 201703L)
#   define CC3D_HAS_STD_SHUFFLE 1
#else
#   define CC3D_HAS_STD_SHUFFLE 0
#endif

// ------------------------------------------------------------
// Internal RNG access (thread-local, seedable)
// ------------------------------------------------------------
    inline std::mt19937 &rng() {
        static thread_local std::mt19937
        engine{std::random_device{}()};
        return engine;
    }

// ------------------------------------------------------------
// Optional deterministic seeding
// ------------------------------------------------------------
    inline void set_random_seed(std::uint32_t seed) {
        rng().seed(seed);
    }

// ------------------------------------------------------------
// Generic container randomization
// ------------------------------------------------------------
    template<typename Container>
    inline void randomize_container(Container &c) {
#if CC3D_HAS_STD_SHUFFLE
        std::shuffle(c.begin(), c.end(), rng());
#else
        std::random_shuffle(c.begin(), c.end());
#endif
    }

} // namespace CompuCell3D

#endif // COMPUCELL3D_RANDOMUTILS_H

