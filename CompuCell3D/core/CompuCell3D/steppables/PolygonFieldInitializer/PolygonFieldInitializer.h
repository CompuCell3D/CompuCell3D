#ifndef POLYGONFIELDINITIALIZER_H
#define POLYGONFIELDINITIALIZER_H

#include <CompuCell3D/CC3D.h>


#include "PolygonFieldInitializerDLLSpecifier.h"

namespace CompuCell3D {
    class Potts3D;

    class Simulator;

    // class Dim3D;

    class Point3D;

    class POLYGONFIELDINITIALIZER_EXPORT PolygonFieldInitializerData {
    public:
        PolygonFieldInitializerData() :
                width(1), gap(0), randomize(false) {}

        //Here, every ith point srcPoints connects to the ith point of dstPoints to form an edge
        std::vector <Point3D> srcPoints; //source points
        std::vector <Point3D> dstPoints; //destination points
        
        std::vector <std::string> typeNames;
        std::string typeNamesString;
        int width;
        int gap;
        bool randomize;

        void Gap(int _gap) { gap = _gap; }

        void Width(int _width) { width = _width; }

        void SrcPoints(std::vector <Point3D> _srcPoints) { srcPoints = _srcPoints; }

        void DstPoints(std::vector <Point3D> _dstPoints) { dstPoints = _dstPoints; }

        void Types(std::string _type) {
            typeNames.push_back(_type);
        }
    };

    class POLYGONFIELDINITIALIZER_EXPORT PolygonFieldInitializer : public Steppable {
    protected:
        Potts3D *potts;
        Simulator *sim;

        // Dim3D polygonDim;
        // bool cellSortInit;
        std::vector <PolygonFieldInitializerData> initDataVec;

        void layOutCells(const PolygonFieldInitializerData &_initData);

        unsigned char initCellType(const PolygonFieldInitializerData &_initData);


    public:
        CC3DXMLElement *moduleXMLDataPtr;

        PolygonFieldInitializer();

        virtual ~PolygonFieldInitializer() {}

        void setPotts(Potts3D *potts) { this->potts = potts; }

        double distance(double, double, double, double, double, double);

        // Dim3D getPolygonDim() { return polygonDim; }

        // SimObject interface
        virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData = 0);

        // Begin Steppable interface
        virtual void start();

        virtual void step(const unsigned int currentStep) {}

        virtual void finish() {}

        // End Steppable interface
        Dim3D getPolygonDimensions(const Dim3D &dim, int size);

        void initializeCellTypesCellSort();

        virtual std::string steerableName();

        virtual std::string toString();
    };
};
#endif
