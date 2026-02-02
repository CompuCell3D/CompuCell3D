#pragma once
#include <cstdint>
#include <cstddef>

namespace CompuCell3D {

    using cc3d_dim_t = int32_t;
    // Signed long integer - platform independent
    using cc3d_long_t = std::int64_t;

    // Unsigned integer used for indexing / sizes
    using cc3d_index_t = std::size_t;

}
