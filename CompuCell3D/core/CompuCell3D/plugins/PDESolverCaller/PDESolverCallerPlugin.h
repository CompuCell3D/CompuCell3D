#ifndef PDESOLVERCALLERPLUGIN_H
#define PDESOLVERCALLERPLUGIN_H

#include <CompuCell3D/CC3D.h>


#include "PDESolverCallerDLLSpecifier.h"

class CC3DXMLElement;
namespace CompuCell3D {

    class Potts3D;

    class CellG;

    class Steppable;

    class Simulator;


    class PDESOLVERCALLER_EXPORT SolverData {
    public:
        SolverData() : extraTimesPerMC(0) {}

        SolverData(std::string _solverName, unsigned int _extraTimesPerMC) :
                solverName(_solverName),
                extraTimesPerMC(_extraTimesPerMC) {}

        std::string solverName;
        unsigned int extraTimesPerMC;

    };

    class PDESOLVERCALLER_EXPORT PDESolverCallerPlugin : public Plugin, public FixedStepper {
        Potts3D *potts;
        Simulator *sim;
        CC3DXMLElement *xmlData;

        std::vector <SolverData> solverDataVec;


        std::vector<Steppable *> solverPtrVec;

    public:
        PDESolverCallerPlugin();

        virtual ~PDESolverCallerPlugin();

        ///SimObject interface
        virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData = 0);

        virtual void extraInit(Simulator *simulator);

        // Stepper interface
        virtual void step();


        //SteerableObject interface
        virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag = false);

        virtual std::string steerableName();

        virtual std::string toString();

    };
};
#endif

