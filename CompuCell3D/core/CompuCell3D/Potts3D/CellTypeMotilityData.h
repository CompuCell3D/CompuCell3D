
#ifndef CELLTYPEMOTILITYDATA_H
#define CELLTYPEMOTILITYDATA_H

#include <vector>
#include <string>

namespace CompuCell3D {

    class CellTypeMotilityData {
    public:
        CellTypeMotilityData() : motility(0.0) {}

        std::string typeName;
        float motility;

    };

};
#endif
