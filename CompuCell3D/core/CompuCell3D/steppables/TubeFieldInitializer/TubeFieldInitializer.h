#ifndef TUBEFIELDINITIALIZER_H
#define TUBEFIELDINITIALIZER_H

#include <CompuCell3D/CC3D.h>


#include "TubeFieldInitializerDLLSpecifier.h"

namespace CompuCell3D {
    class Potts3D;

    class Simulator;

    // class Dim3D;

    class Point3D;

    const int DEFAULT_NUM_SLICES = 8; //arbitrary
    
    enum CellShapeEnum {
        CUBE = 0, 
        WEDGE = 1
    };

    class TUBEFIELDINITIALIZER_EXPORT TubeFieldInitializerData {
    public:
        TubeFieldInitializerData() :
                width(1), gap(0), numSlices(DEFAULT_NUM_SLICES), innerRadius(0), outerRadius(0), cellShape(WEDGE), randomize(false) {}
        
        Point3D fromPoint;
        Point3D toPoint;
        std::vector <std::string> typeNames;
        std::string typeNamesString;
        int width;
        int gap;
        int numSlices;
        int innerRadius;
        int outerRadius;
        bool randomize;
        CellShapeEnum cellShape;


        void Gap(int _gap) { gap = _gap; }

        void Width(int _width) { width = _width; }

        void NumSlices(int _numSlices) { numSlices = _numSlices; }

        void FromPoint(Point3D _fromPoint) { fromPoint = _fromPoint; }

        void ToPoint(Point3D _toPoint) { toPoint = _toPoint; }

        void InnerRadius(int _innerRadius) { innerRadius = _innerRadius; }

        void OuterRadius(int _outerRadius) { outerRadius = _outerRadius; }

        void CellShape(CellShapeEnum _cellShape) { cellShape = _cellShape; }

        void Types(std::string _type) {
            typeNames.push_back(_type);
        }
    };

    class TUBEFIELDINITIALIZER_EXPORT TubeFieldInitializer : public Steppable {
    protected:
        Potts3D *potts;
        Simulator *sim;

        // Dim3D tubeDim;
        // bool cellSortInit;
        std::vector <TubeFieldInitializerData> initDataVec;

        void layOutCellsCube(const TubeFieldInitializerData &_initData);
        
        void layOutCellsWedge(const TubeFieldInitializerData &_initData);

        unsigned char initCellType(const TubeFieldInitializerData &_initData);


    public:
        CC3DXMLElement *moduleXMLDataPtr;

        TubeFieldInitializer();

        virtual ~TubeFieldInitializer() {}

        void setPotts(Potts3D *potts) { this->potts = potts; }



        Point3D crossProduct(Point3D p1, Point3D p2);

        std::vector<double> crossProductVec(const std::vector<double>& v1, const std::vector<double>& v2);

        double magnitude(Point3D p);

        double distanceToLine(Point3D line_point1, Point3D line_point2, Point3D point);



        // SimObject interface
        virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData = 0);

        // Begin Steppable interface
        virtual void start();

        virtual void step(const unsigned int currentStep) {}

        virtual void finish() {}
        // End Steppable interface

        Dim3D getTubeDimensions(const Dim3D &dim, int size);

        double distance(double, double, double, double);

        void initializeCellTypesCellSort();

        virtual std::string steerableName();

        virtual std::string toString();
    };
};
#endif
