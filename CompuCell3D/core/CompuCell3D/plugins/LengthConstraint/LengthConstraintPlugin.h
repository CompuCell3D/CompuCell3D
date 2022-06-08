#ifndef LENGTHCONSTRAINTPLUGIN_H
#define LENGTHCONSTRAINTPLUGIN_H

#include <CompuCell3D/CC3D.h>

#include "LengthConstraintData.h"

#include "LengthConstraintDLLSpecifier.h"

class CC3DXMLElement;

namespace CompuCell3D {
    class Potts3D;

    class CellG;

    class BoundaryStrategy;

    class LENGTHCONSTRAINT_EXPORT LengthEnergyParam {
    public:
        LengthEnergyParam() : targetLength(0), lambdaLength(0.0), minorTargetLength(0.0) {}

        LengthEnergyParam(std::string _cellTypeName, double _targetLength, double _lambdaLength) :
                cellTypeName(_cellTypeName), targetLength(_targetLength), lambdaLength(_lambdaLength) {}

        double targetLength;
        double minorTargetLength;
        float lambdaLength;
        std::string cellTypeName;

    };

    class LENGTHCONSTRAINT_EXPORT LengthConstraintPlugin : public Plugin, public EnergyFunction {

        CC3DXMLElement *xmlData;

        Potts3D *potts;
        //energy function parse data

        ExtraMembersGroupAccessor <LengthConstraintData> lengthConstraintDataAccessor;

        std::unordered_map<unsigned char, LengthEnergyParam> lengthEnergyParamMap;
        std::vector <std::string> typeNameVec;//temporary vector for storage type names
        BoundaryStrategy *boundaryStrategy;

        double _get_non_nan_energy(double energy);

        double spring_energy(double lam, double x, double x0);

    public:

        typedef double (LengthConstraintPlugin::*changeEnergyFcnPtr_t)(const Point3D &pt, const CellG *newCell,
                                                                       const CellG *oldCell);

        LengthConstraintPlugin();

        virtual ~LengthConstraintPlugin();

        // Plugin interface
        virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData = 0);

        virtual void extraInit(Simulator *simulator);

        virtual std::string toString();

        ExtraMembersGroupAccessor <LengthConstraintData> *
        getLengthConstraintDataPtr() { return &lengthConstraintDataAccessor; }

        void setLengthConstraintData(CellG *_cell, double _lambdaLength = 0.0, double _targetLength = 0.0,
                                     double _minorTargetLength = 0.0);

        double getLambdaLength(CellG *_cell);

        double getTargetLength(CellG *_cell);

        double getMinorTargetLength(CellG *_cell);


        //EnergyFunction interface
        virtual double changeEnergy(const Point3D &pt, const CellG *newCell, const CellG *oldCell);

        double changeEnergy_xz(const Point3D &pt, const CellG *newCell, const CellG *oldCell);

        double changeEnergy_xy(const Point3D &pt, const CellG *newCell, const CellG *oldCell);


        double changeEnergy_yz(const Point3D &pt, const CellG *newCell, const CellG *oldCell);


        double changeEnergy_3D(const Point3D &pt, const CellG *newCell, const CellG *oldCell);

        changeEnergyFcnPtr_t changeEnergyFcnPtr;

        //SteerableObject interface
        virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag = false);

        virtual std::string steerableName();

    };
};
#endif
