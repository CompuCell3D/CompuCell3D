#ifndef CELLORIENTATIONPLUGIN_H
#define CELLORIENTATIONPLUGIN_H

#include <CompuCell3D/CC3D.h>
#include <CompuCell3D/plugins/PolarizationVector/PolarizationVector.h>

#include "CellOrientationDLLSpecifier.h"

namespace CompuCell3D {
    class CellOrientationEnergy;

    class PolarizationVector;


    class Point3D;

    class Potts3D;

    class Simulator;

    class PolarizationVector;

    class LambdaCellOrientation;

    class BoundaryStrategy;


    class BoundaryStrategy;

    template<class T>
    class Field3D;

    class CELLORIENTATION_EXPORT LambdaCellOrientation {
    public:
        LambdaCellOrientation() : lambdaVal(0.0) {}

        double lambdaVal;
    };

    class CELLORIENTATION_EXPORT CellOrientationPlugin : public Plugin, public EnergyFunction {


        Field3D<CellG *> *cellFieldG;


        ExtraMembersGroupAccessor <LambdaCellOrientation> lambdaCellOrientationAccessor;

        //EnergyFunction data

        Potts3D *potts;
        double lambdaCellOrientation;
        Simulator *simulator;
        Dim3D fieldDim;
        ExtraMembersGroupAccessor <PolarizationVector> *polarizationVectorAccessorPtr;
        BoundaryStrategy *boundaryStrategy;

        bool lambdaFlexFlag;


    public:
        CellOrientationPlugin();

        virtual ~CellOrientationPlugin();

        // SimObject interface
        virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData = 0);

        virtual void extraInit(Simulator *simulator);


        typedef double (CellOrientationPlugin::*changeEnergy_t)(const Point3D &pt, const CellG *newCell,
                                                                const CellG *oldCell);

        CellOrientationPlugin::changeEnergy_t changeEnergyFcnPtr;

        //EnergyFunctionInterface
        virtual double changeEnergy(const Point3D &pt, const CellG *newCell,
                                    const CellG *oldCell);

        double changeEnergyCOMBased(const Point3D &pt, const CellG *newCell, const CellG *oldCell);

        double changeEnergyPixelBased(const Point3D &pt, const CellG *newCell, const CellG *oldCell);


        ExtraMembersGroupAccessor <PolarizationVector> *
        getPolarizationVectorAccessorPtr() { return polarizationVectorAccessorPtr; }

        ExtraMembersGroupAccessor <LambdaCellOrientation> *
        getLambdaCellOrientationAccessorPtr() { return &lambdaCellOrientationAccessor; }

        void setLambdaCellOrientation(CellG *_cell, double _lambda);

        double getLambdaCellOrientation(CellG *_cell);

        //Steerable interface
        virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag = false);

        virtual std::string steerableName();

        virtual std::string toString();


    };
};
#endif
