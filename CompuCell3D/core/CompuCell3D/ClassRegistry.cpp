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


#include "Simulator.h"




//#include <XMLCereal/XMLPullParser.h>
//#include <XMLCereal/XMLSerializer.h>

#include <BasicUtils/BasicClassFactory.h>
#include <BasicUtils/BasicSmartPointer.h>
#include <BasicUtils/BasicException.h>
#include <BasicUtils/BasicString.h>

#include "ClassRegistry.h"


#include <string>
using namespace CompuCell3D;
using namespace std;

ClassRegistry::ClassRegistry(Simulator *simulator) : simulator(simulator) {
}



Steppable *ClassRegistry::getStepper(string id) {
//   BasicSmartPointer<Steppable> stepper = activeSteppersMap[id];
  Steppable* stepper = activeSteppersMap[id];
//   ASSERT_OR_THROW(string("Stepper '") + id + "' not found!", stepper.get());
//   cerr<<"REQUESTING STEPPER: "<<id<<endl;
  ASSERT_OR_THROW(string("Stepper '") + id + "' not found!", stepper);
//   return stepper.get();
  return stepper;
}

void ClassRegistry::extraInit(Simulator * simulator) {
  ActiveSteppers_t::iterator it;
  for (it = activeSteppers.begin(); it != activeSteppers.end(); it++)
    (*it)->extraInit(simulator);
}


void ClassRegistry::start() {
  ActiveSteppers_t::iterator it;
  for (it = activeSteppers.begin(); it != activeSteppers.end(); it++)
    (*it)->start();
}

void ClassRegistry::step(const unsigned int currentStep) {
  ActiveSteppers_t::iterator it;
  for (it = activeSteppers.begin(); it != activeSteppers.end(); it++) {
     if ((*it)->frequency && (currentStep % (*it)->frequency) == 0)
      (*it)->step(currentStep);
  }
}

void ClassRegistry::finish() {
  ActiveSteppers_t::iterator it;
  for (it = activeSteppers.begin(); it != activeSteppers.end(); it++)
    (*it)->finish();
}


//void ClassRegistry::readXML(XMLPullParser &in) {
//  in.skip(TEXT);
//  
//  while (in.check(START_ELEMENT)) {
//	if (in.getName() == "Steppable") {
//      string type = in.getAttribute("Type").value;
////       cerr<<"\n\n\nGot STEPPABLE NO will try to load it\n\n\n"<<endl;
//      Steppable *steppable=0;
//      steppable = simulator->steppableManager.get(type);
////       BasicSmartPointer<Steppable> steppable = steppableRegistry.create(type);
////       cerr<<"\t\t\t\t\t\t\t\tSteppable "<<type<<" loaded"<<endl;
//
//
//      if (in.findAttribute("Frequency") != -1)
//	steppable->frequency =
//	  BasicString::parseUInteger(in.getAttribute("Frequency").value);
//
////       cerr<<"GOT HERE"<<endl;
////       cerr<<"steppable="<<steppable<<endl;
//
//       //steppable->init(simulator);
//       in.match(START_ELEMENT);
//       steppable->readXML(in);
//       in.match(END_ELEMENT);
//       steppableParseDataVector.push_back(steppable->getParseData());
//
//      activeSteppers.push_back(steppable);
//      activeSteppersMap[type] = steppable;
//
//    } else {
//      THROW(string("Unexpected element '") + in.getName() + "'!");
//    }
//
//    in.skip(TEXT);
//  }
//
//   
//
//
//}

void ClassRegistry::addStepper(std::string _type, Steppable *_steppable){
      activeSteppers.push_back(_steppable);
      activeSteppersMap[_type] = _steppable;

}

void ClassRegistry::initModules(Simulator *_sim){

   //std::vector<ParseData *> & steppableParseDataVectorRef= _sim->ps.steppableParseDataVector;
	std::vector<CC3DXMLElement *> steppableCC3DXMLElementVectorRef = _sim->ps.steppableCC3DXMLElementVector;



   PluginManager<Steppable> &steppableManagerRef=Simulator::steppableManager;

   cerr<<" INSIDE INIT MODULES:"<<endl;
   //cerr<<"steppableParseDataVectorRef.size()="<<steppableParseDataVectorRef.size()<<endl;
   for (int i=0; i <steppableCC3DXMLElementVectorRef.size(); ++i){
      //string type=steppableParseDataVectorRef[i]->moduleName;
		string type=steppableCC3DXMLElementVectorRef[i]->getAttribute("Type");

      Steppable *steppable = steppableManagerRef.get(type);

      cerr<<"CLASS REGISTRY INITIALIZING "<<type<<endl;

//       if(plugin->getParseData()){//plugin has been initialized/read through XML
//          plugin->init(this,plugin->getParseData());
//       }else{//plugin parse data has been initialized externally
//          plugin->init(this, ps.pluginParseDataVector[i]);
//       }
      
      //steppable->init(_sim, steppableParseDataVectorRef[i]);
		steppable->init(_sim, steppableCC3DXMLElementVectorRef[i]);
      addStepper(type,steppable);
//       _sim->registerSteerableObject(steppable);
   
   }

   for (ActiveSteppers_t::iterator litr = activeSteppers.begin() ; litr != activeSteppers.end() ; ++litr){
      cerr<<"HAVE THIS STEPPER : "<<(*litr)->getParseData()->moduleName<<endl;;
   }

//    for (ActiveSteppers_t::iterator litr = activeSteppers.begin() ; litr != activeSteppers.end() ; ++litr){
//       (*litr)->init(simulator,(*litr)->getParseData());
//    }

}

//void ClassRegistry::writeXML(XMLSerializer &out) {
//}
