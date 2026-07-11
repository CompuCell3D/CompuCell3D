#ifndef CC3D_YAML_CONFORMANCE_LOADER_H
#define CC3D_YAML_CONFORMANCE_LOADER_H

#include <string>
#include <vector>

namespace CompuCell3D {

    struct ConformancePoint {
        int x;
        int y;
        int z;
    };

    struct ConformanceLattice {
        ConformancePoint dim;
        std::string boundaryX;
        std::string boundaryY;
        std::string boundaryZ;
        std::string neighborhoodKind;
        unsigned int neighborhoodOrder;
    };

    struct ConformanceCell {
        int id;
        std::string type;
        double targetVolume;
        double lambdaVolume;
        std::vector<ConformancePoint> pixels;
    };

    struct ConformanceExpectedValue {
        double value;
        double tolerance;
    };

    struct ConformanceVolumeQuery {
        std::string id;
        ConformancePoint source;
        ConformancePoint target;
        ConformanceExpectedValue expected;
    };

    struct ConformanceCase {
        std::string path;
        int version;
        std::string name;
        std::string domain;
        std::string mediumType;
        ConformanceLattice lattice;
        std::vector<ConformanceCell> cells;
        std::vector<ConformanceVolumeQuery> volumeQueries;
    };

    ConformanceCase loadConformanceCase(const std::string &path);
}

#endif
