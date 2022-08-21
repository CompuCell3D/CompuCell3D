#ifndef DERIVEDPROPERTY_H
#define DERIVEDPROPERTY_H

namespace CompuCell3D {

    /**
    DerivedProperty: Derived properties that mimic the behavior of Python properties in C++
    Written by T.J. Sego, Ph.D.
    9/12/2020

    The DerivedProperty returns the value of the property on demand according to a function
    that defines the DerivedProperty. All requisite information to calculate the current
    value of a DerivedProperty must be intrinsic to the instance of the parent class.

    For CC3D Python support, follow the procedure of existing implementations
    using the SWIG macro DERIVEDPROPERTYEXTENSORPY in CompuCell3D/core/pyinterface/CompuCellPython/DerivedProperty.i
    */

    template<typename ParentType, typename PropertyType, PropertyType(ParentType::*PropertyFunction)()>
    class DerivedProperty {

        // Parent object with this derived property as a member
        ParentType *obj;

    public:

        DerivedProperty() {}

        DerivedProperty(ParentType *_obj) :
                obj(_obj) {}

        ~DerivedProperty() { obj = 0; }

        // Pretend to be a value
        operator PropertyType() const { return (obj->*PropertyFunction)(); }

    };

}

#endif