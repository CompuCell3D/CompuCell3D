#ifndef CC3D_TEST_SIMULATOR_H
#define CC3D_TEST_SIMULATOR_H

#include <CompuCell3D/Plugin.h>
#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/Field3D/WatchableField3D.h>
#include <XMLUtils/CC3DXMLElement.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace CompuCell3D {

    class CC3DTestSimulator {
    public:
        CC3DTestSimulator();
        ~CC3DTestSimulator();

        CC3DXMLElement *addPottsData(const Dim3D &dim, unsigned int neighborOrder = 1, double temperature = 10.0,
                                     unsigned int steps = 1);
        CC3DXMLElement *addPluginData(const std::string &name);
        CC3DXMLElement *addCellTypePluginData(const std::vector<std::pair<std::string, unsigned char>> &types);

        void initializeSimulator();

        Simulator *getSimulator() { return simulator.get(); }
        Potts3D *getPotts() { return simulator->getPotts(); }
        WatchableField3D<CellG *> *getCellField() {
            return static_cast<WatchableField3D<CellG *> *>(simulator->getPotts()->getCellFieldG());
        }

        CellG *createCell(const Point3D &seed, unsigned char type, const std::vector<Point3D> &additionalPixels = {});

        template<class PluginType>
        PluginType *getPlugin(const std::string &name) {
            bool alreadyRegistered = false;
            Plugin *plugin = Simulator::pluginManager.get(name, &alreadyRegistered);
            return dynamic_cast<PluginType *>(plugin);
        }

    private:
        CC3DXMLElement *createElement(const std::string &name,
                                      const std::vector<std::pair<std::string, std::string>> &attributes = {},
                                      const std::string &cdata = "");
        static std::string toString(int value);
        static std::string toString(unsigned int value);
        static std::string toString(double value);

        std::unique_ptr<Simulator> simulator;
        std::vector<std::unique_ptr<CC3DXMLElement>> ownedElements;
    };
}

#endif
