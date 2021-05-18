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

#define CompuCellLibShared_EXPORTS // if you dont define this DLL import/export macro  from CompuCellLib you will get error "definition of dllimport static data member not allowed"
//this is because you define static members in the Simulator class and witohut this macro they will be redefined here as import symbols which is not allowed

#include "Simulator.h"
#include "CC3DExceptions.h"
using namespace CompuCell3D;

#include <BasicUtils/BasicSmartPointer.h>

#include <iostream>
#include <string>
#include <fstream>
using namespace std;

#include <stdlib.h>

#include <XMLUtils/XMLParserExpat.h>

#if defined(_WIN32)
#include <windows.h>
#endif

PluginManager<Plugin> Simulator::pluginManager;
PluginManager<Steppable> Simulator::steppableManager;
BasicPluginManager<PluginBase> Simulator::pluginBaseManager;

void Syntax(const string name)
{
    cerr << "Syntax: " << name << " <config>" << endl;
    exit(1);
}

using namespace std;

int main(int argc, char* argv[])
{
    cerr << "Welcome to CC3D command line edition" << endl;

    // If we'd want to redirect cerr & cerr to a file (also see at end)
    //  std::ofstream logFile("cc3d-out.txt");
    //  std::streambuf *outbuf = std::cerr.rdbuf(logFile.rdbuf());
    //  std::streambuf *errbuf = std::cerr.rdbuf(logFile.rdbuf());

    try {
        // Command line
        if (argc < 2)
            Syntax(argv[0]);

        // Load Plugin Libaries
        // Libaries in COMPUCELL3D_PLUGIN_PATH can override the
        // DEFAULT_PLUGIN_PATH
        char* steppablePath = getenv("COMPUCELL3D_STEPPABLE_PATH");
        cerr << "steppablePath=" << steppablePath << endl;
        if (steppablePath)
            Simulator::steppableManager.loadLibraries(steppablePath);

        char* pluginPath = getenv("COMPUCELL3D_PLUGIN_PATH");
        cerr << "pluginPath=" << pluginPath << endl;
        if (pluginPath)
            Simulator::pluginManager.loadLibraries(pluginPath);

#ifdef DEFAULT_PLUGIN_PATH
        Simulator::steppableManager.loadLibraries(DEFAULT_PLUGIN_PATH);
//     Simulator::pluginManager.loadLibraries(DEFAULT_PLUGIN_PATH);
#endif

        BasicPluginManager<Steppable>::infos_t* infosG = &Simulator::steppableManager.getPluginInfos();

        if (!infosG->empty()) {
            cerr << "Found the following Steppables:" << endl;
            BasicPluginManager<Steppable>::infos_t::iterator it;
            for (it = infosG->begin(); it != infosG->end(); it++)
                cerr << "  " << *(*it) << endl;
            cerr << endl;
        }

        BasicPluginManager<Plugin>::infos_t* infos = &Simulator::pluginManager.getPluginInfos();

        if (!infos->empty()) {
            cerr << "Found the following plugins:" << endl;
            BasicPluginManager<Plugin>::infos_t::iterator it;
            for (it = infos->begin(); it != infos->end(); it++)
                cerr << "  " << *(*it) << endl;
            cerr << endl;
        }
    }
    catch (const CC3DException& e) {
        cerr << "ERROR: " << e << endl;
    }

    XMLParserExpat xmlParser;
    if (argc < 2) {
        cerr << "SPECIFY XML FILE" << endl;
        exit(0);
    }
    xmlParser.setFileName(string(argv[1]));
    xmlParser.parse();

    // Create Simulator
    Simulator sim;

    //extracting plugin elements from the XML file
    CC3DXMLElementList pluginDataList = xmlParser.rootElement->getElements("Plugin");
    for (int i = 0; i < pluginDataList.size(); ++i) {
        cerr << "Plugin: " << pluginDataList[i]->getAttribute("Name") << endl;
        sim.ps.addPluginDataCC3D(pluginDataList[i]);
    }

    //extracting steppable elements from the XML file
    CC3DXMLElementList steppableDataList = xmlParser.rootElement->getElements("Steppable");
    for (int i = 0; i < steppableDataList.size(); ++i) {
        cerr << "Steppable: " << steppableDataList[i]->getAttribute("Type") << endl;
        sim.ps.addSteppableDataCC3D(steppableDataList[i]);
    }

    // extracting Potts section
    CC3DXMLElementList pottsDataList = xmlParser.rootElement->getElements("Potts");
    if (pottsDataList.size() != 1) throw CC3DException("You must have exactly 1 definition of the Potts section");
    sim.ps.addPottsDataCC3D(pottsDataList[0]);

    //    extracting Metadata section
    CC3DXMLElementList metadataDataList = xmlParser.rootElement->getElements("Metadata");
    if (metadataDataList.size() == 1) {
        sim.ps.addMetadataDataCC3D(metadataDataList[0]);
    }
    else {
        cerr << "Not using Metadata" << endl;
    }

    sim.initializeCC3D();

    sim.extraInit(); ///additional initialization after all plugins and steppables have been loaded and preinitialized
// Run simulation

#if defined(_WIN32)
    volatile DWORD dwStart;
    dwStart = GetTickCount();
#endif

    sim.start();

    for (unsigned int i = 1; i <= sim.getNumSteps(); i++) {
        sim.step(i);
    }
    sim.finish();

#if defined(_WIN32)
    cerr << "SIMULATION TOOK " << GetTickCount() - dwStart << " miliseconds to complete" << endl;
    dwStart = GetTickCount();
#endif

    return 0;
}