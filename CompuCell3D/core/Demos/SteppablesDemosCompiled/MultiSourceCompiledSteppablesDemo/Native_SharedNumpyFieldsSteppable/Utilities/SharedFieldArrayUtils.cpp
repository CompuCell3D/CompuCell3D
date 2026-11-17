#include "SharedFieldArrayUtils_1.h"

#include <algorithm>
#include <iostream>

void printScalarFieldHead(const cc3d::kernel::SharedScalarField &field,
                          int rowCount,
                          int valuesPerRow) {
    const auto dimensions = field.dim();
    const int rows = std::max(0, std::min(rowCount, dimensions.y));
    const int columns = std::max(0, std::min(valuesPerRow, dimensions.x));

    std::cout << "C++ scalar field head (rows=" << rows
              << ", values_per_row=" << columns << ")\n";
    for (int y = 0; y < rows; ++y) {
        std::cout << "  row " << y << ":";
        for (int x = 0; x < columns; ++x) {
            std::cout << ' ' << field.get(x, y, 0);
        }
        std::cout << '\n';
    }
}
