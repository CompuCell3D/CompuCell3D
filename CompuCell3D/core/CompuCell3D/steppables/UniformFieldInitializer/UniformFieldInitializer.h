

#ifndef UNIFORMFIELDINITIALIZER_H
#define UNIFORMFIELDINITIALIZER_H

#include <CompuCell3D/CC3D.h>
#include "UniformFieldInitializerDLLSpecifier.h"

namespace CompuCell3D {
    class Potts3D;

    class Simulator;

    class CellInventory;

    class UNIFORMFIELDINITIALIZER_EXPORT UniformFieldInitializerData {

    public:
        UniformFieldInitializerData() :
                boxMin(Dim3D(0, 0, 0)), boxMax(Dim3D(0, 0, 0)), width(1), gap(0), randomize(false) {}

        Dim3D boxMin;
        Dim3D boxMax;
        std::vector <std::string> typeNames;
        std::string typeNamesString;
        int width;
        int gap;
        bool randomize;

        void BoxMin(Dim3D &_boxMin) { boxMin = _boxMin; }

        void BoxMax(Dim3D &_boxMax) { boxMax = _boxMax; }

        void Gap(int _gap) { gap = _gap; }

        void Width(int _width) { width = _width; }

        void Types(std::string _type) {
            typeNames.push_back(_type);
        }
    };


    class UNIFORMFIELDINITIALIZER_EXPORT UniformFieldInitializer : public Steppable {
        Potts3D *potts;
        Simulator *sim;
        CellInventory *cellInventoryPtr;

        void layOutCells(const UniformFieldInitializerData &_initData);

        unsigned char initCellType(const UniformFieldInitializerData &_initData);

        std::vector <UniformFieldInitializerData> initDataVec;

    public:

        UniformFieldInitializer();

        virtual ~UniformFieldInitializer() {};

        void setPotts(Potts3D *potts) { this->potts = potts; }

        void initializeCellTypes();

        // SimObject interface
        virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData = 0);

        // Begin Steppable interface
        virtual void start();

        virtual void step(const unsigned int currentStep) {}

        virtual void finish() {}

        // End Steppable interface
        virtual std::string steerableName();

        virtual std::string toString();


    };
};
#endif
