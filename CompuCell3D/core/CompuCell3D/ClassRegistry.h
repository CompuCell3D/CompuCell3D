/*************************************************************************
 *    CompuCell - A software framework for multimodel simulations of     *
 * biocomplexity problems Copyright (C) 2003 University of Notre Dame,   *
 *                             Indiana                                   *
 *                                                                       *
 * This program is free software; IF YOU AGREE TO CITE USE OF CompuCell  *
 *  IN ALL RELATED RESEARCH PUBLICATIONS according to the terms of the   *
 *  CompuCell GNU General Public License RIDER you can redistribute it   *
 * and/or modify it under the terms of the GNU General Public License as *
 *  published by the Free Software Foundation; either version 2 of the   *
 *         License, or (at your option) any later version.               *
 *                                                                       *
 * This program is distributed in the hope that it will be useful, but   *
 *      WITHOUT ANY WARRANTY; without even the implied warranty of       *
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU    *
 *             General Public License for more details.                  *
 *                                                                       *
 *  You should have received a copy of the GNU General Public License    *
 *     along with this program; if not, write to the Free Software       *
 *      Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.        *
 *************************************************************************/

#ifndef CLASSREGISTRY_H
#define CLASSREGISTRY_H

//#include <CompuCell3D/dllDeclarationSpecifier.h>
#include <CompuCell3D/CompuCellLibDLLSpecifier.h>
#include "Steppable.h"





#include <BasicUtils/BasicClassRegistry.h>
#include <BasicUtils/BasicSmartPointer.h>

#include <map>
#include <list>
#include <string>
#include <vector>

namespace CompuCell3D {
  class Simulator;
  

  class COMPUCELLLIB_EXPORT ClassRegistry : public Steppable {
    BasicClassRegistry<Steppable> steppableRegistry;

//     typedef std::list<BasicSmartPointer<Steppable> > ActiveSteppers_t;
    typedef std::list<Steppable *> ActiveSteppers_t;
    ActiveSteppers_t activeSteppers;

//     typedef std::map<std::string, BasicSmartPointer<Steppable> > ActiveSteppersMap_t;
    typedef std::map<std::string, Steppable *> ActiveSteppersMap_t;
    ActiveSteppersMap_t activeSteppersMap;


    Simulator *simulator;

    std::vector<ParseData *> steppableParseDataVector;


  public:
    ClassRegistry(Simulator *simulator);
    virtual ~ClassRegistry() {}

//     void registerSteppable(std::string id,
// 			  BasicClassFactoryBase<SteppableG> *steppable)
//     {steppableRegistry.registerFactory(steppable, id);}

    
    Steppable *getStepper(std::string id);
    
     void addStepper(std::string _type, Steppable *_steppable);
    // Begin Steppable interface
    virtual void extraInit(Simulator *simulator);
    
    virtual void start();
    virtual void step(const unsigned int currentStep);
    virtual void finish();    
    // End Steppable interface

    virtual void initModules(Simulator *_sim);
  };
};
#endif
