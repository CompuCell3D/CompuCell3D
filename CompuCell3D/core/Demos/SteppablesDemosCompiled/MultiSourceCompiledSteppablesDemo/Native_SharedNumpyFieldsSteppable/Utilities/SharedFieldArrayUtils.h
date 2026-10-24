#pragma once

#include <cc3d/kernel.h>

void printScalarFieldHead(const cc3d::kernel::SharedScalarField &field,
                          int rowCount,
                          int valuesPerRow);
