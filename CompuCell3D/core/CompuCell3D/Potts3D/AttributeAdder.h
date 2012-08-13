#ifndef ATTRIBUTEADDER_H
#define ATTRIBUTEADDER_H
namespace CompuCell3D{

class CellG;

class AttributeAdder{
    public:
	AttributeAdder(){}
	virtual ~AttributeAdder(){}
	virtual void addAttribute(CellG *){};
	virtual void destroyAttribute(CellG *){};

};

};

#endif
