#ifndef CLASSREGISTRY_H
#define CLASSREGISTRY_H

#include <CompuCell3D/CompuCellLibDLLSpecifier.h>
#include <CompuCell3D/CC3DExceptions.h>
#include "Steppable.h"

#include <map>
#include <list>
#include <string>
#include <vector>

namespace CompuCell3D {
    class Simulator;



    class COMPUCELLLIB_EXPORT ClassRegistry

    : public Steppable {

    typedef std::list<Steppable *> ActiveSteppers_t;
    ActiveSteppers_t activeSteppers;

    typedef std::map<std::string, Steppable *> ActiveSteppersMap_t;
    ActiveSteppersMap_t activeSteppersMap;


    Simulator *simulator;

    std::vector<ParseData *> steppableParseDataVector;


    public:
    ClassRegistry(Simulator
    *simulator);
    virtual ~

    ClassRegistry() {}

    Steppable *getStepper(std::string id) ;

    void addStepper(std::string _type, Steppable *_steppable);

    // Begin Steppable interface
    virtual void extraInit(Simulator * simulator);

    virtual void start();

    virtual void step(const unsigned int currentStep);

    virtual void finish();
    // End Steppable interface

    virtual void initModules(Simulator * _sim);
};
};
#endif
