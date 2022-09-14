
#ifndef ENERGYFUNCTION_H
#define ENERGYFUNCTION_H

#include "Potts3D.h"
#include <string>

namespace CompuCell3D {

    /**
    * The Potts3D energy function interface.
    */

    class Point3D;

    class CellG;

    class EnergyFunction {
    protected:

    public:
        EnergyFunction() {}

        virtual ~EnergyFunction() {}

        /**
        * @return The energy change for this function at point pt.
        */
        virtual double localEnergy(const Point3D &pt) { return 0.0; };

        /**
        * @param pt The point of change.
        * @param newCell The new spin.
        *
        * @return The energy change of changing point pt to newCell.
        */

        virtual double changeEnergy(const Point3D &pt, const CellG *newCell, const CellG *oldCell) {
            return 0.0;
        }

        virtual std::string toString() {
            return std::string("EnergyFunction");
        }
    };
};
#endif
