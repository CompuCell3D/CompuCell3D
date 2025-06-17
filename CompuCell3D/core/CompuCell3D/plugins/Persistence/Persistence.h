#ifndef PERSISTENCE_H
#define PERSISTENCE_H

/**
@author T.J. Sego
*/
#include <Utils/Coordinates3D.h>
#include "PersistenceDLLSpecifier.h"

#include <vector>

namespace CompuCell3D {

    class PERSISTENCE_EXPORT PersistenceData {
    public:
        PersistenceData() : 
            persistenceAngles{0.0, 0.0, 0.0},
            magnitude{0.0}
        {}
        PersistenceData(const PersistenceData& _other) {
            persistenceAngles = Coordinates3D<double>(_other.persistenceAngles.X(), _other.persistenceAngles.Y(), _other.persistenceAngles.Z());
            magnitude = _other.magnitude;
        }

        // Z, X, Z2 Euler angles rotating unit vector [1, 0, 0]
        Coordinates3D<double> persistenceAngles;
        double magnitude;

    };
};
#endif
