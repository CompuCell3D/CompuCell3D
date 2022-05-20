
#ifndef CENTEROFMASSPLUGIN_H
#define CENTEROFMASSPLUGIN_H

#include <CompuCell3D/CC3D.h>

#define roundf(a) ((fmod(a,1)<0.5)?floor(a):ceil(a))


#include "CenterOfMassDLLSpecifier.h"

namespace CompuCell3D {
    class Potts3D;

    class ParseData;

    class BoundaryStrategy;


    class CENTEROFMASS_EXPORT CenterOfMassPlugin : public Plugin, public CellGChangeWatcher {
        Potts3D *potts;

    private:
        Point3D boundaryConditionIndicator;
        Dim3D fieldDim;
        BoundaryStrategy *boundaryStrategy;
        //determine allowed ranges of COM position - it COM is outside those values
        // we will shift COM (by multiples of lattice sizes +1,-1*lattice size) to inside the allowed area
        Coordinates3D<double> allowedAreaMin;
        Coordinates3D<double> allowedAreaMax;


    public:
        CenterOfMassPlugin();

        virtual ~CenterOfMassPlugin();

        void getCenterOfMass(CellG *cell, float cm[3]) const {
            if (!cell) throw CC3DException("getCenterOfMass() Cell cannot be NULL!");

            unsigned int volume = cell->volume;
            if (!volume) throw CC3DException("getCenterOfMass() Cell volume is 0!");


            cm[0] = cell->xCM / (float) volume;
            cm[1] = cell->yCM / (float) volume;
            cm[2] = cell->zCM / (float) volume;

        }

        void getCenterOfMass(CellG *cell, float &_x, float &_y, float &_z) const {
            if (!cell) throw CC3DException("getCenterOfMass() Cell cannot be NULL!");

            unsigned int volume = cell->volume;
            if (!volume) throw CC3DException("getCenterOfMass() Cell volume is 0!");


            _x = cell->xCM / (float) volume;
            _y = cell->yCM / (float) volume;
            _z = cell->zCM / (float) volume;

        }


        Point3D getCenterOfMass(CellG *cell) const {

            float floatCM[3];
            getCenterOfMass(cell, floatCM);

            return Point3D((unsigned short) roundf(floatCM[0]),
                           (unsigned short) roundf(floatCM[1]),
                           (unsigned short) roundf(floatCM[2]));

        }

        // BCGChangeWatcher interface
        void field3DCheck(const Point3D &pt, CellG *newCell,
                          CellG *oldCell);

        virtual void handleEvent(CC3DEvent &_event);

        // SimObject interface
        virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData = 0);

        // BCGChangeWatcher interface
        virtual void field3DChange(const Point3D &pt, CellG *newCell,
                                   CellG *oldCell);

        virtual std::string toString();

        virtual std::string steerableName();
    };
};
#endif
