// This macro performs Python-equivalent implementation of the C++ DerivedProperty
// for a wrapped class
// className: the name of the class with a DerivedProperty member; explicitly including the namespace of the class helps
// propertyName: the name of the DerivedProperty member, exactly as it is defined in the class definition
// accessorName: the name of the function that defines the value of the DerivedProperty; do not include its namespace
%define DERIVEDPROPERTYEXTENSORPY(className, propertyName, accessorName)
%extend className {
	%pythoncode %{
    def derived_property_get ## propertyName(self):
        return self. ## accessorName()

    def derived_property_set ## propertyName(self, _val):
        raise AttributeError('Assignment of derived property propertyName is illegal.')

    propertyName = property(derived_property_get ## propertyName, derived_property_set ## propertyName)

    %}
}
%enddef

// CellG DerivedProperties

DERIVEDPROPERTYEXTENSORPY(CompuCell3D::CellG, pressure, getPressure)
DERIVEDPROPERTYEXTENSORPY(CompuCell3D::CellG, surfaceTension, getSurfaceTension)
DERIVEDPROPERTYEXTENSORPY(CompuCell3D::CellG, clusterSurfaceTension, getClusterSurfaceTension)

// Link DerivedProperties
DERIVEDPROPERTYEXTENSORPY(CompuCell3D::FocalPointPlasticityLinkBase, length, getDistance)
DERIVEDPROPERTYEXTENSORPY(CompuCell3D::FocalPointPlasticityLinkBase, tension, getTension)

DERIVEDPROPERTYEXTENSORPY(CompuCell3D::FocalPointPlasticityLink, cellPair, getCellPair)
DERIVEDPROPERTYEXTENSORPY(CompuCell3D::FocalPointPlasticityInternalLink, cellPair, getCellPair)
DERIVEDPROPERTYEXTENSORPY(CompuCell3D::FocalPointPlasticityAnchor, cell, getObj0)