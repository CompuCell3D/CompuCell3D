#ifndef LENGTHCONSTRAINTLOCALFLEXPLUGIN_H
#define LENGTHCONSTRAINTLOCALFLEXPLUGIN_H

#include <CompuCell3D/CC3D.h>
#include "LengthConstraintLocalFlexData.h"
#include "LengthConstraintLocalFlexDLLSpecifier.h"


class CC3DXMLElement;
namespace CompuCell3D {

    class Potts3D;

    class CellG;

    class BoundaryStrategy;


    class LENGTHCONSTRAINTLOCALFLEX_EXPORT LengthConstraintLocalFlexPlugin : public Plugin, public EnergyFunction {


        Potts3D *potts;
        BasicClassAccessor <LengthConstraintLocalFlexData> lengthConstraintLocalFlexDataAccessor;
        BoundaryStrategy *boundaryStrategy;

    public:


        typedef double (LengthConstraintLocalFlexPlugin::*changeEnergyFcnPtr_t)(const Point3D &pt, const CellG *newCell,
                                                                                const CellG *oldCell);

        LengthConstraintLocalFlexPlugin();

        virtual ~LengthConstraintLocalFlexPlugin();

        // SimObject interface
        virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData = 0);

        virtual std::string toString();


        BasicClassAccessor <LengthConstraintLocalFlexData> *
        getLengthConstraintLocalFlexDataPtr() { return &lengthConstraintLocalFlexDataAccessor; }

        void setLengthConstraintData(CellG *_cell, double _lambdaLength, double _targetLength);

        double getTargetLength(CellG *_cell);

        double getLambdaLength(CellG *_cell);


        //Energy Function interface
        virtual double changeEnergy(const Point3D &pt, const CellG *newCell,
                                    const CellG *oldCell);

        virtual double changeEnergy_xy(const Point3D &pt, const CellG *newCell,
                                       const CellG *oldCell);

        virtual double changeEnergy_xz(const Point3D &pt, const CellG *newCell,
                                       const CellG *oldCell);

        virtual double changeEnergy_yz(const Point3D &pt, const CellG *newCell,
                                       const CellG *oldCell);

        changeEnergyFcnPtr_t changeEnergyFcnPtr;


    };
};
#endif
