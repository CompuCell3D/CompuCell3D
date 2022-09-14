#ifndef CELL_H
#define CELL_H

#include <vector>
#include <CompuCell3D/DerivedProperty.h>
#include <CompuCell3D/ExtraMembers.h>

#ifndef PyObject_HEAD
struct _object; //forward declare
typedef _object PyObject; //type redefinition
#endif

namespace CompuCell3D {

    /**
     * A Potts3D cell.
     */

    class CellG {
    public:
        typedef unsigned char CellType_t;

        CellG();

        long volume;
        float targetVolume;
        float lambdaVolume;
        double surface;
        float targetSurface;
        float angle;
        float lambdaSurface;
        double clusterSurface;
        float targetClusterSurface;
        float lambdaClusterSurface;
        unsigned char type;
        unsigned char subtype;
        double xCM, yCM, zCM; // numerator of center of mass expression (components)
        double xCOM, yCOM, zCOM; // numerator of center of mass expression (components)
        double xCOMPrev, yCOMPrev, zCOMPrev; // previous center of mass
        double iXX, iXY, iXZ, iYY, iYZ, iZZ; // tensor of inertia components
        float lX, lY, lZ; //orientation vector components - set by MomentsOfInertia Plugin - read only
        float ecc; // cell eccentricity
        float lambdaVecX, lambdaVecY, lambdaVecZ; // external potential lambda vector components
        unsigned char flag;
        float averageConcentration;
        long id;
        long clusterId;
        double fluctAmpl;
        double lambdaMotility;
        double biasVecX;
        double biasVecY;
        double biasVecZ;
        bool connectivityOn;
        //std::vector<double> test_biasV = std::vector<double>(3);
        ExtraMembersGroup *extraAttribPtr;

        PyObject *pyAttrib;

        // Derived properties

    public:

        // Function defining the value of derived property: pressure
        float getPressure();

        // Function defining the value of derived property: surface tension
        float getSurfaceTension();

        // Function defining the value of derived property: cluster surface tension
        float getClusterSurfaceTension();

        // Internal pressure
        DerivedProperty<CellG, float, &CellG::getPressure> pressure;
        // Surface tension
        DerivedProperty<CellG, float, &CellG::getSurfaceTension> surfaceTension;
        // Cluster surface tension
        DerivedProperty<CellG, float, &CellG::getClusterSurfaceTension> clusterSurfaceTension;

    };


    class Cell {
    };

    class CellPtr {
    public:
        Cell *cellPtr;
    };
};
#endif
