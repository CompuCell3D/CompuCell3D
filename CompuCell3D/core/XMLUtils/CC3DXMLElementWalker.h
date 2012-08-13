#ifndef CC3DXMLELEMENTWALKER_H
#define CC3DXMLELEMENTWALKER_H


class CC3DXMLElement;

#include "XMLUtilsDLLSpecifier.h"
class XMLUTILS_EXPORT CC3DXMLElementWalker{
public:
	CC3DXMLElementWalker();
	void iterateCC3DXMLElement(CC3DXMLElement* _element);
};

#endif


