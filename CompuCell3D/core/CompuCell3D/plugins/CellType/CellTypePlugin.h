#ifndef CELLTYPEPLUGIN_H
#define CELLTYPEPLUGIN_H

#include <CompuCell3D/CC3D.h>

#include "CellTypeG.h"
#include "CellTypeDLLSpecifier.h"


class CC3DXMLElement;

namespace CompuCell3D {
    class Potts3D;

    class CellG;


    class CELLTYPE_EXPORT CellTypePlugin : public Plugin, public Automaton {
        Potts3D *potts;

        std::map<unsigned char, std::string> typeNameMap;
        std::map<std::string, unsigned char> nameTypeMap;
        unsigned char maxTypeId;

    public:

        CellTypePlugin();

        virtual ~CellTypePlugin();

        ///SimObject interface
        virtual void init(Simulator *simulator, ParseData *_pd = 0);

        virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData);

        virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag = false);

        std::map<unsigned char, std::string> &getTypeNameMap() { return typeNameMap; }

        ///Automaton Interface
        virtual unsigned char getCellType(const CellG *) const;

        virtual std::string getTypeName(const char type) const;

        virtual unsigned char getTypeId(const std::string typeName) const;

        virtual unsigned char getMaxTypeId() const;

        virtual const std::vector<unsigned char> getTypeIds() const;

        virtual std::string toString();

        //Steerable object interface
        virtual std::string steerableName();

        virtual void update(ParseData *_pd, bool _fullInitFlag = false);

    };

    inline unsigned char CellTypePlugin::getCellType(const CellG *cell) const { return cell ? cell->type : 0; };

    inline unsigned char CellTypePlugin::getMaxTypeId() const { return typeNameMap.empty() ? 0 : maxTypeId; };

    inline const std::vector<unsigned char> CellTypePlugin::getTypeIds() const {
        std::vector<unsigned char> o;
        for (auto &x: nameTypeMap) o.push_back(x.second);
        return o;
    }

};
#endif

