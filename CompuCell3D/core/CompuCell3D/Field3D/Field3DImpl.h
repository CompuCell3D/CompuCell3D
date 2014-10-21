/*************************************************************************
 *    CompuCell - A software framework for multimodel simulations of     *
 * biocomplexity problems Copyright (C) 2003 University of Notre Dame,   *
 *                             Indiana                                   *
 *                                                                       *
 * This program is free software; IF YOU AGREE TO CITE USE OF CompuCell  *
 *  IN ALL RELATED RESEARCH PUBLICATIONS according to the terms of the   *
 *  CompuCell GNU General Public License RIDER you can redistribute it   *
 * and/or modify it under the terms of the GNU General Public License as *
 *  published by the Free Software Foundation; either version 2 of the   *
 *         License, or (at your option) any later version.               *
 *                                                                       *
 * This program is distributed in the hope that it will be useful, but   *
 *      WITHOUT ANY WARRANTY; without even the implied warranty of       *
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU    *
 *             General Public License for more details.                  *
 *                                                                       *
 *  You should have received a copy of the GNU General Public License    *
 *     along with this program; if not, write to the Free Software       *
 *      Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.        *
 *************************************************************************/

#ifndef FIELD3DIMPL_H
#define FIELD3DIMPL_H

#include <math.h>
#include <CompuCell3D/Boundary/BoundaryStrategy.h>
#include <BasicUtils/BasicException.h>

#include "Dim3D.h"
#include "Field3D.h"


namespace CompuCell3D {
//indexing macro
#define PT2IDX(pt) (pt.x + ((pt.y + (pt.z * dim.y)) * dim.x))

  template<class T>
  class Field3D;

  /** 
   * Default implementation of the Field3D interface.
   *
   */
  template<class T>
  class Field3DImpl : public Field3D<T> {
  protected:
    Dim3D dim;
    T *field;
    T initialValue;
    long len;
public:
  
    /**
     * @param dim The field dimensions
     * @param initialValue The initial value of all data elements in the field.
     */
    Field3DImpl(const Dim3D dim, const T &initialValue) :
      dim(dim),field(0), initialValue(initialValue) {

      ASSERT_OR_THROW("Field3D cannot have a 0 dimension!!!",
		      dim.x != 0 && dim.y != 0 && dim.z != 0);

      // Check that the dimensions are not too large.
      ASSERT_OR_THROW("Field3D dimensions too large!!!",
		      log((double)dim.x)/log(2.0) + log((double)dim.y)/log(2.0) + log((double)dim.z)/log(2.0) <=
		      sizeof(int) * 8);

      // Allocate and initialize the field
      len = dim.x * dim.y * dim.z;
      field = new T[len];
      for (unsigned int i = 0; i < len; i++)
	field[i] = initialValue;
    }

    virtual ~Field3DImpl() {
        if(field){
            delete [] field;
            field=0;
        }
            
    }

    virtual void set(const Point3D &pt, const T value) {
      ASSERT_OR_THROW("set() point out of range!", isValid(pt));
      field[PT2IDX(pt)] = value;
    }

	virtual void resizeAndShift(const Dim3D theDim,  Dim3D shiftVec=Dim3D()){
       T* field2 = new T[theDim.x*theDim.y*theDim.z];
       //first initialize the lattice with initial value 
       for(long int i = 0 ; i <  theDim.x*theDim.y*theDim.z ; ++i) 
            field2[i]=initialValue;
       
       //then  copy old field 
       for (int x = 0; x < theDim.x; x++)
           for (int y = 0; y < theDim.y; y++)
               for (int z = 0; z < theDim.z; z++)
                   //if ((x-shiftVec.x < dim.x) && (y-shiftVec.y < dim.y) && (z < dim.z))
				   //field2[(x+shiftVec.x)+(((y+shiftVec.y)+((z+shiftVec.z)*theDim.y))*theDim.x)] = getQuick(Point3D(x,y,z));
				   if ((x-shiftVec.x>=0) && (x-shiftVec.x<dim.x) && (y-shiftVec.y>=0) && (y-shiftVec.y<dim.y) && (z-shiftVec.z>=0) && (z-shiftVec.z<dim.z)	){
						field2[x+((y+(z*theDim.y))*theDim.x)] = getQuick(Point3D(x-shiftVec.x,y-shiftVec.y,z-shiftVec.z));
				   }
                       
                       
       delete []  field;
       field = field2;
       dim = theDim;
		
	
       //Set dimension for the Boundary Strategy
       BoundaryStrategy::getInstance()->setDim(dim);		

	}
    virtual void setDim(const Dim3D theDim,  Dim3D shiftVec=Dim3D())
    {
		this->resizeAndShift(theDim);
    }

   T getQuick(const Point3D &pt) const {

      //return field[PT2IDX(pt)];
       return (isValid(pt) ? field[PT2IDX(pt)] : initialValue);
    }

   void setQuick(const Point3D &pt,const T _value)  {

      field[PT2IDX(pt)]=_value;
    }


    virtual T get(const Point3D &pt) const {
      
      return (isValid(pt) ? field[PT2IDX(pt)] : initialValue);
    }

    virtual T getByIndex(long _offset) const {
      
      return ( ((0<=_offset)&&(_offset<len)) ? field[_offset] : initialValue);
    }

    virtual void setByIndex(long _offset, const T _value) {
      if((0<=_offset)&&(_offset<len))
         field[_offset]=_value;
    }

    
    
    virtual Dim3D getDim() const {return dim;}

    virtual bool isValid(const Point3D &pt) const {
      return (0 <= pt.x && pt.x < dim.x &&
	      0 <= pt.y && pt.y < dim.y &&
	      0 <= pt.z && pt.z < dim.z);
    }
  };    
};
#endif
