/*******************************************************************\

              Copyright (C) 2003 Joseph Coffland

    This program is free software; you can redistribute it and/or
     modify it under the terms of the GNU General Public License
    as published by the Free Software Foundation; either version 2
        of the License, or (at your option) any later version.

   This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
             GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
     along with this program; if not, write to the Free Software
      Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA
                           02111-1307, USA.

            For information regarding this software email
                   jcofflan@users.sourceforge.net

\*******************************************************************/

#include "BasicException.h" 

#ifndef BASICSMARTPOINTER_H
#define BASICSMARTPOINTER_H

#include "BasicCounter.h"

#include <stdlib.h>

// Forward Delcarations
class BasicException;

#include <BasicUtils/BasicException.h>

typedef enum {SP_NEW, SP_ARRAY, SP_MALLOC} sp_alloc_t;

/** 
 * This class is an implementation of a smart pointer.  IT IS NOT
 * THREAD SAFE.  This means that you cannot use BasicSmartPointers
 * to the same data in different threads.  Even if you make a new
 * copy of the smart pointer.  This is because they will still share
 * the same BasicCounter.  However, BasicSmartPointer
 * is faster because it is not thread safe.
 *
 * You can use a BasicSmartPointer on a dynamic instance of a class A as 
 * follows.
 *
 * BasicSmartPointer<A> aPtr = new A;
 *
 * Since aPtr is static when it goes out of scope it will be destructed.
 * When this happens the reference counter will be decremented and if
 * it is zero the dynamic instance of A will be destructed.
 *
 * You can make a copy of a smart pointer as follows.
 *
 * BasicSmartPointer<A> anotherPtr = aPtr;
 *
 * Now the dynamic instance of A will not be destructed until both
 * aPtr and anotherPtr go out of scope.
 *
 * See http://ootips.org/yonat/4dev/smart-pointers.html for more information
 * about smart pointers and why to use them.
 */
template <class T, sp_alloc_t alloc_t = SP_NEW>
class BasicSmartPointer {
  /// A pointer to the reference counter.
  BasicCounter *refCounter;

  /// The actual pointer.
  T *ptr;

public:
  /** 
   * Create a NULL smart pointer
   *
   */
  BasicSmartPointer() : refCounter(0), ptr(0) {}

  /** 
   * The copy constructor.  If the smart pointer being copied
   * is not NULL the object reference count will be incremented.
   * 
   * @param smartPtr The pointer to copy.
   */
  BasicSmartPointer(const BasicSmartPointer<T, alloc_t> &smartPtr) :
    refCounter(0), ptr(0) {
    *this = smartPtr;
  }

  /** 
   * Create a smart pointer from a pointer value.  If ptr is
   * non-NULL the reference count will be set to one.
   * NOTE: It is an error to create more than one smart pointer 
   * through this constructor for a single pointer value because 
   * this will cause the object
   * to which it points to be deallocated more than once.  To create
   * a copy of a smart pointer either use the copy constructor or
   * the smart pointer to smart pointer assignment function.
   * 
   * @param ptr The pointer to point to.
   */
  BasicSmartPointer(T *ptr) : refCounter(0), ptr(0) {
    if (ptr) {
      refCounter = new BasicCounter(1);
      this->ptr = ptr;
    }
  }

  /** 
   * Destroy this smart pointer.  If this smart pointer is set to
   * a non-NULL value and there are no other references to the
   * object to which it points the object will be deleted.
   */
  ~BasicSmartPointer() {release();}

  /** 
   * Compare two smart pointers for equality.  Two smart pointers 
   * are equal if and only if their internal pointers are equal.
   * 
   * @param smartPtr The pointer to compare with.
   * 
   * @return True if the smart pointers are equal. False otherwise.
   */
  const bool operator==(const BasicSmartPointer<T, alloc_t> &smartPtr) {
    return ptr == smartPtr.ptr;
  }

  /** 
   * Compare this smart pointer to an actual pointer value.
   * 
   * @param ptr The pointer to compare with.
   * 
   * @return True if this smart pointers internal pointer equals ptr.
   *         False otherwise.
   */
  const bool operator==(const T *ptr) {
    return this->ptr == ptr;
  }

  /** 
   * Asign this smart pointer to another.  If the passed smart pointer
   * is non-NULL a reference will be added.
   * 
   * @param smartPtr The pointer to copy.
   * 
   * @return A reference to this smart pointer.
   */
  BasicSmartPointer<T, alloc_t> &
  operator=(const BasicSmartPointer<T, alloc_t> &smartPtr) {
    if (*this == smartPtr) return *this;

    release();

    refCounter = smartPtr.refCounter;
    if (refCounter) refCounter->inc();
    ptr = smartPtr.ptr;

    return *this;
  }

  /** 
   * Dereference this smart pointer.
   * A BasicException will be thrown if this smart pointer is NULL.
   *
   * @return A reference to the object pointed to by this smart pointer.
   */
  T *operator->() const {checkPtr(); return get();}

  /** 
   * Dereference this smart pointer.
   * A BasicException will be thrown if this smart pointer is NULL.
   *
   * @return A reference to the object pointed to by this smart pointer.
   */
  T &operator*() const {checkPtr(); return *get();}

  /** 
   * Dereference this smart pointer with an array index.
   * A BasicException will be thrown if this smart pointer is NULL.
   *
   * @return A reference to an object in the array pointed to by 
   * this smart pointer.
   */
  T &operator[](const long x) const {checkPtr(); return get()[x];}

  /** 
   * Access this smart pointers internal object pointer;
   * 
   * @return The value of the internal object pointer;
   */
  T *get() const {return ptr;}

  /// Not operator
  bool operator!() const {return ptr == 0;}

  /** 
   * Release this smart pointer's reference.  
   * If the reference count is one the object to which this
   * smart pointer points will be deleted.
   */
  void release() {
    if (refCounter && !refCounter->dec()) {
      delete refCounter;
      switch (alloc_t) {
      case SP_NEW: delete ptr; break;
      case SP_ARRAY: delete [] ptr; break;
      case SP_MALLOC: free(ptr); break;
      }
    }

    refCounter = 0;
    ptr = 0;
  }

  /** 
   * Assume responsibility for this pointer.  If the reference
   * counter is more than one a BasicException will be thrown.
   * When successful this smart pointer will be NULL.
   *
   * This function can be useful for moving the pointer to
   * a base class smart pointer.
   * 
   * @return The value of the internal pointer.
   */
  T *adopt() {
    if (refCounter && refCounter->getCount() > 1)
      throw BasicException(std::string("BasicSmartPointer: Cannot adopt a") +
			   "pointer with multiple references!");

    if (refCounter) {
      delete refCounter;
      refCounter = 0;
    }
    T *tmp = ptr;
    ptr = 0;

    return tmp;
  }

  /** 
   * Get the number of references to the object pointed to by
   * this smart pointer.  The reference count will be zero
   * if this is a NULL smart pointer other wise it will be
   * one or more.
   * 
   * @return The reference count.
   */
  long getRefCount() const {return refCounter ? refCounter->getCount() : 0;}

  /** 
   * Check for NULL pointer value;
   * 
   * @return True if the internal pointer is NULL. False otherwise.
   */
  bool isNull() const {return get() == 0;}

protected:
  void checkPtr() const {
    if (!ptr)
      throw 
	BasicException("BasicSmartPointer: Can't dereference a NULL pointer!");
  }
};
#endif
