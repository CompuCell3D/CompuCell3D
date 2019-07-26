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


#define CompuCellLibShared_EXPORTS	// if you dont define this DLL import/export macro  from CompuCellLib you will get error "definition of dllimport static data member not allowed"
									//this is because you define static members in the Simulator class and witohut this macro they will be redefined here as import symbols which is not allowed

#include "Simulator.h"
using namespace CompuCell3D;

#include <BasicUtils/BasicException.h>
#include <BasicUtils/BasicSmartPointer.h>

// #include <XMLCereal/XMLPullParser.h>

// #include <XercesUtils/XercesStr.h>

// #include <xercesc/util/PlatformUtils.hpp>
// XERCES_CPP_NAMESPACE_USE;

#include <iostream>
#include <string>
#include <fstream>
using namespace std;

#include <stdlib.h>

//#include <config.h>
#include <BasicUtils/BasicRandomNumberGenerator.h>

//void Syntax(const string name) {
//  cerr << "Syntax: " << name << " <config>" << endl;
//  exit(1);
//}
//
////the reason to declare BoundaryStrategy* BoundaryStrategy::singleton; here is because 
////Simulator.h includes Potts.h which includes WatchableField3D.h which includes Field3Dimpl.h which includes BoundaryStrategy.h
////BoundaryStrategy* BoundaryStrategy::singleton;
//
//
PluginManager<Plugin> Simulator::pluginManager;
PluginManager<Steppable> Simulator::steppableManager;
BasicPluginManager<PluginBase> Simulator::pluginBaseManager;
//
//int main(int argc, char *argv[]) {
//
//     // Create Simulator
//    Simulator sim;
//
//	cerr<<"THIS IS COMPUCELL3D"<<endl;
//
//  try {
//    // Command line
//    if (argc < 2) Syntax(argv[0]);
//	
//    // Load Plugin Libaries
//    // Libaries in COMPUCELL3D_PLUGIN_PATH can override the
//    // DEFAULT_PLUGIN_PATH
////      char *steppablePath = getenv("COMPUCELL3D_STEPPABLE_PATH");
////      cerr<<"steppablePath="<<steppablePath<<endl;
////      if (steppablePath) Simulator::steppableManager.loadLibraries(steppablePath);
////
//     //char *pluginPath = getenv("COMPUCELL3D_PLUGIN_PATH");
//     //cerr<<"pluginPath="<<pluginPath<<endl;
//     //     Simulator::pluginManager.loadLibrary(string("PluginA") + BasicPluginManager<Plugin>::libExtension);
////      Simulator::steppableManager.loadLibrary(string(pluginPath)+string("\\")+string("CC3DCellTypePlugin") + BasicPluginManager<Plugin>::libExtension);
////      Simulator::steppableManager.loadLibrary(string("CC3DCellTypePlugin") + BasicPluginManager<Plugin>::libExtension);
//
//  //   Simulator::pluginManager.loadLibrary(string("CC3DBasicUtils") + BasicPluginManager<Plugin>::libExtension);
//  //   Simulator::pluginBaseManager.loadLibrary(string("PluginType") + BasicPluginManager<Plugin>::libExtension);
//	 //Simulator::pluginManager.loadLibrary(string("CC3DCellType") + BasicPluginManager<Plugin>::libExtension);
//
//	 
//	 //      if (pluginPath) Simulator::pluginManager.loadLibraries(pluginPath);
////     string pluginPath(".");
////	 if (pluginPath.size()) Simulator::pluginManager.loadLibraries(pluginPath);
//
//
//
////exit(0);
//
//      char *steppablePath = getenv("COMPUCELL3D_STEPPABLE_PATH");
//      cerr<<"steppablePath="<<steppablePath<<endl;
//      if (steppablePath) Simulator::steppableManager.loadLibraries(steppablePath);
//
//     char *pluginPath = getenv("COMPUCELL3D_PLUGIN_PATH");
//     cerr<<"pluginPath="<<pluginPath<<endl;
//	 if (pluginPath) Simulator::pluginManager.loadLibraries(pluginPath);
//    
//#ifdef DEFAULT_PLUGIN_PATH
//   Simulator::steppableManager.loadLibraries(DEFAULT_PLUGIN_PATH);
////     Simulator::pluginManager.loadLibraries(DEFAULT_PLUGIN_PATH);
//
//#endif
//
//    BasicPluginManager<Steppable>::infos_t *infosG = 
//      &Simulator::steppableManager.getPluginInfos();
//
//    if (!infosG->empty()) {
//      cerr << "Found the following Steppables:" << endl;
//      BasicPluginManager<Steppable>::infos_t::iterator it; 
//      for (it = infosG->begin(); it != infosG->end(); it++)
//	cerr << "  " << *(*it) << endl;
//      cerr << endl;
//    }
//
//
//
//
//    BasicPluginManager<Plugin>::infos_t *infos = 
//      &Simulator::pluginManager.getPluginInfos();
//
//    if (!infos->empty()) {
//      cerr << "Found the following plugins:" << endl;
//      BasicPluginManager<Plugin>::infos_t::iterator it; 
//      for (it = infos->begin(); it != infos->end(); it++)
//	cerr << "  " << *(*it) << endl;
//      cerr << endl;
//    }
//
//
//
//
//
//    // Initialize Xerces
//    XMLPlatformUtils::Initialize();
//    
//    // Create parser
//    BasicSmartPointer<XMLPullParser> parser = XMLPullParser::createInstance();
//
//    // Parse config
//    try {
//      // Start
//      parser->initParse(argv[1]);
//      parser->match(XMLEventTypes::START_DOCUMENT, -XMLEventTypes::TEXT);
//
//      // Parse
//      parser->assertName("CompuCell3D");
//      parser->match(XMLEventTypes::START_ELEMENT);
//      sim.readXML(*parser);
//      parser->match(XMLEventTypes::END_ELEMENT, -XMLEventTypes::TEXT);
//
//      // End
//      parser->match(XMLEventTypes::END_DOCUMENT);
//
//    } catch (BasicException e) {
//      throw BasicException("While parsing configuration!",
//			   parser->getLocation(), e);
//    }
//
////     BasicRandomNumberGenerator *rand = BasicRandomNumberGenerator::getInstance();
////     cerr<<"rand->getSeed()="<<rand->getSeed()<<endl;
////     for(int i  = 0 ; i < 10000 ; ++i){
////       cerr<<"randomNumber "<<i<<" = "<<rand->getRatio()<<endl;
////     }
////     exit(0);
//    sim.extraInit();///additional initialization after all plugins and steppables have been loaded and preinitialized
//    // Run simulation
//    sim.start();
////     sim.getPotts()->unregisterEnergyFunction("Volume");
//    
//    
//    for (unsigned int i = 1; i <= sim.getNumSteps(); i++)
//      sim.step(i);
//    sim.finish();
//
//    return 0;
//
//  } catch (const XMLException &e) {
//    cerr << "ERROR: " << XercesStr(e.getMessage()) << endl;
//
//  } catch (const BasicException &e) {
//    cerr << "ERROR: " << e << endl;
//  }
//
//  return 1;
//}


//PluginManager<Plugin> Simulator::pluginManager;
//PluginManager<Steppable> Simulator::steppableManager;
//BasicPluginManager<PluginBase> Simulator::pluginBaseManager;


void Syntax(const string name) {
  cerr << "Syntax: " << name << " <config>" << endl;
  exit(1);
}

int main(int argc, char *argv[]) {
	cerr << "mainSimpleDemo" << endl;
	char * pluginPath = "d:/Program Files/cc3d_py3/lib/CompuCell3DPlugins/";
	char * cellTypePluginPath = "d:/Program Files/cc3d_py3/lib/CompuCell3DPlugins/CC3DCellType.dll";
	Simulator::pluginManager.loadLibraries(pluginPath);

	Simulator::pluginManager.loadLibrary(cellTypePluginPath);

   // If we'd want to redirect cerr & cerr to a file (also see at end)
//  std::ofstream logFile("cc3d-out.txt");
//  std::streambuf *outbuf = std::cerr.rdbuf(logFile.rdbuf());
//  std::streambuf *errbuf = std::cerr.rdbuf(logFile.rdbuf());

//  try {
//    // Command line
//    if (argc < 2) Syntax(argv[0]);
//	
//    // Load Plugin Libaries
//    // Libaries in COMPUCELL3D_PLUGIN_PATH can override the
//    // DEFAULT_PLUGIN_PATH
//    char *steppablePath = getenv("COMPUCELL3D_STEPPABLE_PATH");
//    cerr<<"steppablePath="<<steppablePath<<endl;
//    if (steppablePath) Simulator::steppableManager.loadLibraries(steppablePath);
//
//    char *pluginPath = getenv("COMPUCELL3D_PLUGIN_PATH");
//    cerr<<"pluginPath="<<pluginPath<<endl;
//    if (pluginPath) Simulator::pluginManager.loadLibraries(pluginPath);
//
////     if (pluginPath) Simulator::pluginManager.loadLibraries(pluginPath);
//    
//    
//    
//   
//    
//#ifdef DEFAULT_PLUGIN_PATH
//   Simulator::steppableManager.loadLibraries(DEFAULT_PLUGIN_PATH);
////     Simulator::pluginManager.loadLibraries(DEFAULT_PLUGIN_PATH);
//
//#endif
//
//    BasicPluginManager<Steppable>::infos_t *infosG = 
//      &Simulator::steppableManager.getPluginInfos();
//
//    if (!infosG->empty()) {
//      cerr << "Found the following Steppables:" << endl;
//      BasicPluginManager<Steppable>::infos_t::iterator it; 
//      for (it = infosG->begin(); it != infosG->end(); it++)
//	cerr << "  " << *(*it) << endl;
//      cerr << endl;
//    }
//
//
//
//
//    BasicPluginManager<Plugin>::infos_t *infos = 
//      &Simulator::pluginManager.getPluginInfos();
//
//    if (!infos->empty()) {
//      cerr << "Found the following plugins:" << endl;
//      BasicPluginManager<Plugin>::infos_t::iterator it; 
//      for (it = infos->begin(); it != infos->end(); it++)
//	cerr << "  " << *(*it) << endl;
//      cerr << endl;
//    }
//
//
//
//    // Create Simulator
//    Simulator sim;
//
//
//    // Initialize Xerces
//    XMLPlatformUtils::Initialize();
//    
//    // Create parser
//    BasicSmartPointer<XMLPullParser> parser = XMLPullParser::createInstance();
//
//    // Parse config
//    try {
//      // Start
//      parser->initParse(argv[1]);
//      parser->match(XMLEventTypes::START_DOCUMENT, -XMLEventTypes::TEXT);
//
//      // Parse
//      parser->assertName("CompuCell3D");
//      parser->match(XMLEventTypes::START_ELEMENT);
//      cerr<<"BEFORE READXML"<<endl;
//      sim.readXML(*parser);
//      parser->match(XMLEventTypes::END_ELEMENT, -XMLEventTypes::TEXT);
//
//      // End
//      parser->match(XMLEventTypes::END_DOCUMENT);
//		cerr<<"FINISHED INITIALIZING"<<endl;
//      sim.initializeCC3D();
//
//    } catch (BasicException e) {
//      throw BasicException("While parsing configuration!",
//			   parser->getLocation(), e);
//    }
//
////     BasicRandomNumberGenerator *rand = BasicRandomNumberGenerator::getInstance();
////     cerr<<"rand->getSeed()="<<rand->getSeed()<<endl;
////     for(int i  = 0 ; i < 10000 ; ++i){
////       cerr<<"randomNumber "<<i<<" = "<<rand->getRatio()<<endl;
////     }
////     exit(0);
//    sim.extraInit();///additional initialization after all plugins and steppables have been loaded and preinitialized
//    // Run simulation
//    sim.start();
////     sim.getPotts()->unregisterEnergyFunction("Volume");
//    
//    cerr<<"sim.getNumSteps()="<<sim.getNumSteps()<<endl;
//    for (unsigned int i = 1; i <= sim.getNumSteps(); i++)
//      sim.step(i);
//	 cerr<<"got here before finish"<<endl;
//    sim.finish();
//	 cerr<<"got here after finish"<<endl;
//    return 0;
//
//  } catch (const XMLException &e) {
//    cerr << "ERROR: " << XercesStr(e.getMessage()) << endl;
//
//  } catch (const BasicException &e) {
//    cerr << "ERROR: " << e << endl;
//  }
//  
////  // restore the buffers
////  std::cerr.rdbuf(outbuf);
////  std::cerr.rdbuf(errbuf);

  return 1;
}
