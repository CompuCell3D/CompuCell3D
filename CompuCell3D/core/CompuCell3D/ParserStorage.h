#ifndef PARSER_H
#define PARSER_H

#include <CompuCell3D/ParseData.h>
#include <vector>
#include <string>
#include <XMLUtils/CC3DXMLElement.h>

class CC3DXMLElement;

namespace CompuCell3D {

    class ParserStorage {
    public:

        ParserStorage() : pottsCC3DXMLElement(0), updatePottsCC3DXMLElement(0), metadataCC3DXMLElement(0),
                          updateMetadataCC3DXMLElement(0) {}

        CC3DXMLElementList steppableCC3DXMLElementVector;
        CC3DXMLElementList pluginCC3DXMLElementVector;
        CC3DXMLElement *pottsCC3DXMLElement;
        CC3DXMLElement *metadataCC3DXMLElement;

        CC3DXMLElementList updateSteppableCC3DXMLElementVector;
        CC3DXMLElementList updatePluginCC3DXMLElementVector;
        CC3DXMLElement *updatePottsCC3DXMLElement;
        CC3DXMLElement *updateMetadataCC3DXMLElement;

        void addPottsDataCC3D(CC3DXMLElement *_element) { pottsCC3DXMLElement = _element; }

        void addMetadataDataCC3D(CC3DXMLElement *_element) { metadataCC3DXMLElement = _element; }

        void addPluginDataCC3D(CC3DXMLElement *_element) { pluginCC3DXMLElementVector.push_back(_element); }

        void addSteppableDataCC3D(CC3DXMLElement *_element) { steppableCC3DXMLElementVector.push_back(_element); }


    };


};
#endif
