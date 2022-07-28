#ifndef CUSTOMACCEPTANCEFUNCTION_H
#define CUSTOMACCEPTANCEFUNCTION_H


#include <PublicUtilities/ParallelUtilsOpenMP.h>
#include <muParser/ExpressionEvaluator/ExpressionEvaluator.h>

#include "AcceptanceFunction.h"


#include <math.h>

class CC3DXMLElement;

namespace CompuCell3D {


    class ParallelUtilsOpenMP;

    class Simulator;

    class CustomAcceptanceFunction : public AcceptanceFunction {
        ExpressionEvaluatorDepot eed;
        Simulator *simulator;
        ParallelUtilsOpenMP *pUtils;

    public:
        CustomAcceptanceFunction() : simulator(0), pUtils(0) {}

        //AcceptanceFunction interface
        virtual double accept(const double temp, const double change);


        virtual void setOffset(double _offset) {};

        virtual void setK(double _k) {};

        void initialize(Simulator *_sim);

        void update(CC3DXMLElement *_xmlData, bool _fullInitFlag = false);


    };
};
#endif
