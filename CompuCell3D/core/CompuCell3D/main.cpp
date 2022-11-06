
#define CompuCellLibShared_EXPORTS
// if you dont define this DLL import/export macro
// from CompuCellLib you will get error "definition of dllimport static data member not allowed"
//this is because you define static members in the Simulator class and without
// this macro they will be redefined here as import symbols which is not allowed

#include "Simulator.h"
#include "CC3DExceptions.h"

using namespace CompuCell3D;

#include <iostream>
#include <string>
#include <fstream>

using namespace std;

#include <stdlib.h>

#include <XMLUtils/XMLParserExpat.h>
#include <Logger/CC3DLogger.h>
#if defined(_WIN32)
#include <windows.h>
#endif

PluginManager<Plugin> Simulator::pluginManager;
PluginManager<Steppable> Simulator::steppableManager;
PluginManager<PluginBase> Simulator::pluginBaseManager;

void Syntax(const string name) {
    CC3D_Log(LOG_DEBUG) << "Syntax: " << name << " <config>";
    exit(1);
}

using namespace std;

int main(int argc, char* argv[]) {
    CC3D_Log(LOG_DEBUG) << "Welcome to CC3D command line edition";

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
        char *steppablePath = getenv("COMPUCELL3D_STEPPABLE_PATH");
        CC3D_Log(LOG_DEBUG) << "steppablePath=" << steppablePath;
        if (steppablePath)
            Simulator::steppableManager.loadLibraries(steppablePath);

        char *pluginPath = getenv("COMPUCELL3D_PLUGIN_PATH");
        CC3D_Log(LOG_DEBUG) << "pluginPath=" << pluginPath;
        if (pluginPath)
            Simulator::pluginManager.loadLibraries(pluginPath);

#ifdef DEFAULT_PLUGIN_PATH
        Simulator::steppableManager.loadLibraries(DEFAULT_PLUGIN_PATH);
//     Simulator::pluginManager.loadLibraries(DEFAULT_PLUGIN_PATH);
#endif

        PluginManager<Steppable>::infos_t *infosG = &Simulator::steppableManager.getPluginInfos();

        if (!infosG->empty()) {
            CC3D_Log(LOG_DEBUG) << "Found the following Steppables:";
            PluginManager<Steppable>::infos_t::iterator it;
            for (it = infosG->begin(); it != infosG->end(); it++)
                CC3D_Log(LOG_DEBUG) << "  " << *(*it);
        }

        PluginManager<Plugin>::infos_t *infos = &Simulator::pluginManager.getPluginInfos();

        if (!infos->empty()) {
            CC3D_Log(LOG_DEBUG) << "Found the following plugins:";
            PluginManager<Plugin>::infos_t::iterator it;
            for (it = infos->begin(); it != infos->end(); it++)
                CC3D_Log(LOG_DEBUG) << "  " << *(*it);
        }
    }
    catch (const CC3DException &e) {
        CC3D_Log(LOG_DEBUG) <<  "ERROR: " << e;
    }

    XMLParserExpat xmlParser;
    if (argc < 2) {
        CC3D_Log(LOG_DEBUG) << "SPECIFY XML FILE";
        exit(0);
    }
    xmlParser.setFileName(string(argv[1]));
    xmlParser.parse();

    // Create Simulator
    Simulator sim;

    //extracting plugin elements from the XML file
    CC3DXMLElementList pluginDataList = xmlParser.rootElement->getElements("Plugin");
    for (int i = 0; i < pluginDataList.size(); ++i) {
        CC3D_Log(LOG_DEBUG) << "Plugin: " << pluginDataList[i]->getAttribute("Name");
        sim.ps.addPluginDataCC3D(pluginDataList[i]);
    }

    //extracting steppable elements from the XML file
    CC3DXMLElementList steppableDataList = xmlParser.rootElement->getElements("Steppable");
    for (int i = 0; i < steppableDataList.size(); ++i) {
        CC3D_Log(LOG_DEBUG) << "Steppable: " << steppableDataList[i]->getAttribute("Type");
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
    } else {
        CC3D_Log(LOG_DEBUG) << "Not using Metadata";
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
    CC3D_Log(LOG_DEBUG) << "SIMULATION TOOK " << GetTickCount() - dwStart << " miliseconds to complete";
    dwStart = GetTickCount();
#endif

    return 0;
}