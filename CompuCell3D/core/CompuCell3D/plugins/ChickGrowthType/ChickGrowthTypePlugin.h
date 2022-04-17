

#ifndef CHICKGROWTHTYPEPLUGIN_H
#define CHICKGROWTHTYPEPLUGIN_H

#include <CompuCell3D/Plugin.h>

#include <CompuCell3D/Potts3D/CellChangeWatcher.h>
#include <CompuCell3D/Automaton/Automaton.h>
#include <string>

namespace CompuCell3D {
  class Potts3D;
  class Cell;

  class ChickGrowthTypePlugin : public Plugin, public Automaton {
    Simulator* sim;
    Potts3D* potts;
    std::string fieldSource;
    std::string fieldName;
    float threshold;
  public:
    ChickGrowthTypePlugin();
    virtual ~ChickGrowthTypePlugin();

    // SimObject interface
    virtual void init(Simulator *simulator);

    unsigned char getCellType(const CellG *cell) const;
    std::string getTypeName(const char type) const;
    unsigned char getTypeId(const std::string typeName) const;

    float getThreshold() {return threshold;}
    float getConcentration(Point3D pt);

    // Begin XMLSerializable interface
    virtual void readXML(XMLPullParser &in);
    virtual void writeXML(XMLSerializer &out);
    // End XMLSerializable interface
  };
};
#endif
