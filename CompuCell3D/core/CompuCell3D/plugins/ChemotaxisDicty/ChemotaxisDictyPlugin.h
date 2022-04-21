#ifndef CHEMOTAXISDICTYPLUGIN_H
#define CHEMOTAXISDICTYPLUGIN_H

#include <CompuCell3D/CC3D.h>
#include "ChemotaxisDictyDLLSpecifier.h"

class CC3DXMLElement;

namespace CompuCell3D {


    template<class T>
    class Field3D;

    template<class T>
    class Field3DImpl;


    class Potts3D;

    class Simulator;

    class SimpleClock;


    class CHEMOTAXISDICTY_EXPORT ChemotaxisDictyPlugin
            : public Plugin, public CellGChangeWatcher, public EnergyFunction {

        Simulator *sim;
        Field3D<float> *concentrationField;
        //EnergyFunction Data
        Field3D<float> *field;


        Potts3D *potts;
        ExtraMembersGroupAccessor <SimpleClock> *simpleClockAccessorPtr;

        double lambda;

        std::string chemicalFieldSource;
        std::string chemicalFieldName;
        // bool chemotax;
        bool gotChemicalField;

        std::vector<unsigned char> nonChemotacticTypeVector;
        CC3DXMLElement *xmlData;


    public:
        ChemotaxisDictyPlugin();

        virtual ~ChemotaxisDictyPlugin();

        //plugin interface
        virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData = 0);

        virtual void extraInit(Simulator *_simulator);


        ///CellChangeWatcher interface
        virtual void field3DChange(const Point3D &pt, CellG *newCell,
                                   CellG *oldCell);


        virtual void step() {}


        //energyFunction interface
        virtual double changeEnergy(const Point3D &pt, const CellG *newCell,
                                    const CellG *oldCell);

        //steerable interface
        virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag = false);

        virtual std::string steerableName();

        virtual std::string toString();


        //EnergyFunction methods
        double getConcentration(const Point3D &pt);

        double getLambda() { return lambda; }


        Field3D<float> *getField() { return (Field3D<float> *) field; }

        void initializeField();


    };
};
#endif
