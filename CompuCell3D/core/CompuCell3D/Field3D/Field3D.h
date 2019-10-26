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

#ifndef FIELD3D_H
#define FIELD3D_H


// It would be better to hide the implementation (i.e. not include it here),
// but createInstance needs it and because Field3D is templated 
// createInstance must be in the header file.
#include "Field3DImpl.h"
#include "Point3D.h"
#include "Dim3D.h"

#include "NeighborFinder.h"

#include <fstream>

#include <cmath>

#include <algorithm> //need for debian compilation

#include <BasicUtils/BasicException.h>

#include <CompuCell3D/Boundary/BoundaryStrategy.h>

namespace CompuCell3D {
  // Forward declaration
  template <class T> class Field3DImpl;

  // Forward declare class
  template <class T> class Field3D;

  // Forward declare friend functions
  template <class T>
  std::ofstream &operator<<(std::ofstream &, const Field3D<T> &);
  template <class T>
  std::ifstream &operator>>(std::ifstream &, Field3D<T> &);

  /** 
   * A 3 dimensional descrete field.
   *
   * You should only use dynamically allocated instances of this interface.
   * This will make sure the underlying implementation is not lost in copying.
   */
  template <class T>
  class Field3D {
  public:
    /// Used by Field3DIO functions to match data types.
    static const char typeStr[4];    

    /** 
     * Create an instance of the default Field3D implementation.  In this
     * case Field3DImpl.
     * 
     * @param dim The dimensions of the field.
     * @param initialValue The initial value of all data elements in the field.
     * 
     * @return A pointer to the newly allocated Field.
     */
    static Field3D<T> *createInstance(const Dim3D dim, const T &initialValue) {
      return new Field3DImpl<T>(dim, initialValue);
    }

    /** 
     * Set a field element. If the value was already set it will be
     * overwritten.  If pt is out of range then a BasicException will
     * be thrown.
     * 
     * @param pt The coordinate of the element.
     * @param value The new value.
     */
    virtual void set(const Point3D &pt, const T value) = 0;

    /** 
     * If pt is out of range a BasicException will be thrown.
     *
     * @param pt The coordinates of the field element.
     * 
     * @return The value of the element at pt.
     */
    virtual T get(const Point3D &pt) const = 0;

    /** 
     * If _offset is out of range either a BasicException will be thrown
     * or a function will return default value for the field element
     * 
     *
     * @param _offset offset of the field element in the internal field array.
     * 
     * @return The value of the element at _offset.
     */
    virtual T getByIndex(long _offset) const {return T();}//will have to make it abstract

    /** 
     * If _offset is out of range either nothing is done
     *
     * 
     *
     * @param _offset offset of the field element in the internal field array.
     * 
     * @return The value of the element at _offset.
     */
    virtual void setByIndex(long _offset, const T value) {}//will have to make it abstract


        
    /** 
     * If pt is out of range a BasicException will be thrown.
     *
     * @param pt The coordinates of the field element.
     * 
     * @return The value of the element at pt.
     */
    
    /** 
     * If pt is out of range a BasicException will be thrown.
     * 
     * @param pt The coordinates of the field element.
     * 
     * @return The value of the element at pt.
     */
    virtual T operator[](const Point3D &pt) const {return get(pt);}

    /** 
     * @return The dimensions of this field.
     */
    virtual Dim3D getDim() const = 0;

    /** 
     * @param pt A coordinate in 3D space.
     * 
     * @return True if the coordinate is in this field false otherwise.
     */
    virtual bool isValid(const Point3D &pt) const = 0;

    /**
     * Change the dimensions of the field dynamically
     * 
     * @param theDim New dimensions.
     */
    virtual void setDim(const Dim3D theDim) {}
	virtual void resizeAndShift(const Dim3D theDim,const Dim3D shiftVec) {}
    
    virtual void clearSecData(){}
    //virtual std::vector<std::vector<int> > &  getDoNotDiffuseVec();
    //virtual std::vector<std::vector<float> > &  getSecretionData();
    

    /** 
     * Get the next Neighbor points in the sequence.  To start
     * a new sequence of Neighbor points set token to zero.
     * Neighbor points which are outside of the field are automatically
     * filtered out.  If you call this function too many times it will
     * cause an infinite loop when the neighbor depth is beyond all
     * points in the field.
     * 
     * @param pt The current source point.
     * @param token Keeps track of which Neighbor is next.
     * @param distance Is set on return to the distance of the neighbor.
     * 
     * @return The next neighbor point.
     */
    virtual Point3D getNeighbor(const Point3D &pt, unsigned int &token,
				double &distance,
				bool checkBounds = true) const {

     return BoundaryStrategy::getInstance()->getNeighbor(pt, token, distance);
    
    }

   bool isLittleEndian() {
       unsigned int tmp=1;
       return (0 != *(reinterpret_cast<const char*>(&tmp)));
    }   

   template <typename U> inline
   void swapBytes(U& t) {
      char* res = reinterpret_cast<char*>(&t);
      std::reverse(res, res+sizeof(U));
   }

    // Friend declarations for template fstream friend operators.
//     friend std::ofstream &operator<< <>(std::ofstream &, const Field3D<T> &);
//     friend std::ifstream &operator>> <>(std::ifstream &, Field3D<T> &);
  };
    
  template<class T> const char Field3D<T>::typeStr[4] ="c3d";
  
};
#endif
