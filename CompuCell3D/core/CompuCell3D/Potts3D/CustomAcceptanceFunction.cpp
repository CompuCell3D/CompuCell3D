#include <CompuCell3D/Simulator.h>
#include "CustomAcceptanceFunction.h"
#include <iostream>

using namespace CompuCell3D;
using namespace std;

double CustomAcceptanceFunction::accept(const double temp, const double change) {

    int currentWorkNodeNumber = pUtils->getCurrentWorkNodeNumber();
    ExpressionEvaluator &ev = eed[currentWorkNodeNumber];
    double acceptance = 0.0;

    ev[0] = temp;
    ev[1] = change;

    acceptance = ev.eval();

    return acceptance;
}

void CustomAcceptanceFunction::initialize(Simulator *_sim) {
    if (eed.size()) {
        //this means initialization already happened
        return;
    }
    simulator = _sim;
    pUtils = simulator->getParallelUtils();
    unsigned int maxNumberOfWorkNodes = pUtils->getMaxNumberOfWorkNodesPotts();
    eed.allocateSize(maxNumberOfWorkNodes);
    vector <string> variableNames;

    variableNames.push_back("T");
    variableNames.push_back("DeltaE");

    eed.addVariables(variableNames.begin(), variableNames.end());


    eed.initializeUsingParseData();

}

void CustomAcceptanceFunction::update(CC3DXMLElement *_xmlData, bool _fullInitFlag) {
    eed.getParseData(_xmlData);
    if (_fullInitFlag) {
        eed.initializeUsingParseData();
    }
}
