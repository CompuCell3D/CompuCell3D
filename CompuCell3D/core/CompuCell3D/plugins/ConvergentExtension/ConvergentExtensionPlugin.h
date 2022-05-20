#ifndef CONVERGENTEXTENSIONPLUGIN_H
#define CONVERGENTEXTENSIONPLUGIN_H

#include <CompuCell3D/CC3D.h>
#include "ConvergentExtensionDLLSpecifier.h"

class CC3DXMLElement;

namespace CompuCell3D {
    class Potts3D;

    class Automaton;

    class BoundaryStrategy;

    class CONVERGENTEXTENSION_EXPORT ConvergentExtensionPlugin : public Plugin, public EnergyFunction {

        Potts3D *potts;

        std::set<unsigned char> interactingTypes;
        std::unordered_map<unsigned char, double> alphaConvExtMap;
        std::map<std::string, double> typeNameAlphaConvExtMap;

        double depth;

        Automaton *automaton;

        unsigned int maxNeighborIndex;
        BoundaryStrategy *boundaryStrategy;
        CC3DXMLElement *xmlData;

    public:
        ConvergentExtensionPlugin();

        virtual ~ConvergentExtensionPlugin();

        //Plugin interface
        virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData);

        virtual void extraInit(Simulator *simulator);


        //EnergyFunction Interface
        virtual double changeEnergy(const Point3D &pt, const CellG *newCell, const CellG *oldCell);

        //Steerable interface
        virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag = false);

        virtual std::string steerableName();

        virtual std::string toString();


    };
};
#endif
