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
#ifndef BASICARRAY_H
#define BASICARRAY_H

#include "BasicException.h"

#include <stdlib.h>
#include <string.h>
#include <vector>

#include <iostream>

/** 
 * A generic implementation of a dynamically allocated array.
 * 
 * WARNING class constructors and destructors are not called.  Allocated 
 * memory is initially set to zero.  This works well with arrays of pointers 
 * or basic data types, but will not work for classes such as std::string 
 * which requires the constructor be called.
 */
template <class T>
class BasicArray :public std::vector<T>{
//   unsigned int capacity;D
//   unsigned int size;
//   std::vector<T> vec;
//   T *data;

 public:
  
  /** 
   * Create an empty array.
   *
   * @param initialCapacity Initially allocate this many spaces.
   */
//	BasicArray() /*: data(0)*/ {
// 	using namespace std;
// 	cerr<<"DEFAULT CONSTRUCTOR"<<endl;
// 		std::vector<T>::clear();
// 		cerr<<"vec.size()="<<std::vector<T>::size()<<" getSize()="<<getSize()<<endl;
// 	}

  BasicArray(unsigned int initialCapacity=0) /*: data(0)*/ {
//     reset(initialCapacity);
	std::vector<T>::clear();
// 	using namespace std;
	   if(initialCapacity>0)
			std::vector<T>::assign(initialCapacity,T());

// 		cerr<<"size vector="<<std::vector<T>::size()<<" getSize()="<<getSize()<<endl;
  }

  /** 
   * Destruct the array and all the data in it.
   */  
  ~BasicArray() {/*if (data) free(data);*/}


  /** 
   * @return true if size is zero, false otherwise.
   */
  bool isEmpty() const {return std::vector<T>::size() == 0;}  

  /** 
   * @return The current size of the array.
   */
  unsigned int getSize() const {return std::vector<T>::size();}

  /** 
   * setSize() will increase the array capacity if
   * the new size is larger than the current capacity.
   * @see increase.  If size is less than the current
   * any array elements beyond the new size will no
   * longer be accessable.  setSize() does not deallocated
   * any memory.
   * 
   * @param size The new array size.
   */
//   void setSize(const unsigned int size) {
//     increase(size);
//     this->size = size;
//   }

  /** 
   * Will always be >= to getSize().
   * 
   * @return The number of slots currently allocated.
   */
  unsigned int getCapacity() const {return std::vector<T>::capacity();}


  /** 
   * Modifying or deallocating this buffer may adversely effect
   * the contents of the array.
   * 
   * @return A pointer to the internal buffer.
   */
//  T *getBuffer() {
//    return &((*this)[0]);
//  }

  /** 
   * Release the internal buffer. It is then the callers 
   * responsibility to deallocate the returned pointer if
   * not NULL.
   * 
   * @return A pointer to the internal buffer.
   */
//   T *adopt() {
//     T *tmp = data;
// 
//     data = 0;
//     capacity = 0;
//     size = 0;
// 
//     return tmp;
//   }


  /** 
   * Set size to zero and deallocate memory.
   * 
   * @param initialCapacity New capacity.
   */
//   void reset(unsigned int initialCapacity = 0) {
//     if (data) {
//       free(data);
//       data = 0;
//     }
// 
//     capacity = 0;
//     size = 0;
//     increase(initialCapacity);
//   }

  /** 
   * Set size to zero.  Does not effect capacity.
   * Memory is not initialized to zero!
   */
  void clear() {
    std::vector<T>::clear();
  }

  /** 
   * Increase array capacity if minCapacity is greater than
   * the current capacity.  The capacity will be increased
   * to the larger of minCapacity or multiple times the current capacity.
   * The default value for multiple is 2.0.
   * The advantage of this liberal memory allocation comes from
   * the unfortunate side effect that when a memory block is
   * reallocated it may be copied to a new location.
   * Doubling the capacity reduces the amortized time complexity
   * of adding n elements one at a time from O(n^2) to O(n*log(n)).  The
   * disadvantage is that up to twice as much memory may be used.  If you
   * know the maximum size of the array ahead of time you can set
   * the capacity to that size in the constructor.  You may also
   * pass a multiple of zero causing increase to allocate no more than
   * minCapacity.
   *
   * If memory allocation fails when capacity is greater than
   * minCapacity then allocation is retried with capacity equal
   * to minCapacity.  If allocation fails again an std::bad_alloc()
   * exception will be thrown.
   * 
   * @param minCapacity The minimum number of array slots needed.
   * @param multiple The minimum capacity multiplier.
   */
//   void increase(const unsigned int minCapacity,
// 		const unsigned int multiple = 2) {
//     if (minCapacity <= capacity) return;
//     if (minCapacity < multiple * capacity) capacity *= multiple;
//     else capacity = minCapacity;
//     
//     void *newMem = realloc(data, capacity * sizeof(T));
//     if (!newMem) {
//       if (capacity > minCapacity) capacity = minCapacity;
//       newMem = realloc(data, capacity * sizeof(T));
//       
//       if (!newMem) throw std::bad_alloc();
//     }
// 
//     data = (T *)newMem;
//     memset(&data[size], 0, sizeof(T) * (capacity - size));
//   }

  /** 
   * Calling this function assures that the array capacity is equal
   * to its size by releasing any extra allocated memory.  shrink()
   * should normally only be used after the last put() call and
   * before a call to adopt().  However, when the difference
   * between capacity and size is small the recovered memory will
   * often not be usable because of memory fragmentation.  As a
   * rule of thumb do not bother with shrink() when array sizes
   * are small or when you have only a few array instances.  It
   * is generally more efficient to set the array capacity
   * in the constructor. @see increase.
   */
//   void shrink() {
//     if (capacity > size) {
//       data = (T *)realloc(data, size * sizeof(T));
//       capacity = size;
//     }
//   }

  /** 
   * Add the array element to the end of the array.  The size of the array will
   * be increased by one.  See the notes on the increase() function.
   *
   * @param x A reference to the new array element
   * 
   * @return The index of the new array element.
   */
  unsigned int put(const T &x) {
	 using namespace std;
	 std::vector<T>::push_back(x);	
    return std::vector<T>::size()-1;
  }

  /** 
   * Add an array element.  If there is all ready an element at this index it 
   * will be overwritten.  If the index is beyond the current size of the 
   * array the array size will be automatically increased.
   * 
   * @param i The index of the array element.
   * @param x The new array element.
   */
//   void put(const unsigned int i, const T &x) {
//     put(i, &x, 1);
//   }

  /** 
   * Add an array of elements to the end of the array.
   * 
   * @param x A pointer to the elements to be added.
   * @param count The number of elements to be added.
   * 
   * @return The starting position of the added elements.
   */
//   unsigned int put(const T *x, const int count) {
//     put(getSize(), x, count);
//     return getSize() - count;
//   }

  /** 
   * Add an array of elements to the array.  Elements from
   * i to count - 1 will be overwritten.  If i + count is greater
   * than the current size the array will automatically be increased.
   * 
   * @param i The starting index.
   * @param x A pointer to the elements to be added.
   * @param count The number of elements to copy into the array.
   */
//   void put(const unsigned int i, const T *x, const int count) {
// 	 using namespace std;
// 	 cerr<<"i="<<i<<endl;
// 	 cerr<<"count="<<count<<endl;
// 	 cerr<<"sizeof(T)="<<sizeof(T)<<endl;
// 	 cerr<<"x="<<x<<endl;
// 	 void *dest=alloc(i, count);
// 	 cerr<<"dest="<<dest<<endl;
// 	 
//     memcpy(alloc(i, count), x, sizeof(T) * count);
// // 	 memmove(dest, x, sizeof(T) * count);
//   }

  /** 
   * Allocate space for @param count elements at the end of
   * the array. @see alloc(const unsigned int, const unsigned int).
   * 
   * @return A void * to the newly allocated memory.
   */
//   void *alloc(const unsigned int count) {
//     return alloc(getSize(), count);
//   }

  /** 
   * Allocate space for @param count elements at position
   * @param i.  This space may all ready be occupied by other
   * array elements and writing to this memory may destroy their
   * contents.  If @param i + @param count is geater than
   * the array size then the array size will be increased.
   * You should normally use the put and get methods to
   * access the array contents.
   * 
   * @return A void * to the newly allocated memory.
   */
//   void *alloc(const unsigned int i, const unsigned int count) {
//     increase(i + count);
//     if (size < i + count) size = i + count;
//     return &data[i];
//   }

  /** 
   * If the index is out of range a BasicException will be thrown.
   * Array indexing starts from zero.
   * 
   * @param i The index of the array element.
   * 
   * @return A reference to the array element at index i.
   */
  T &get(const unsigned int i) {
// 	 using namespace std;
//     ASSERT_OR_THROW("BasicArray index out of range!", i < std::vector<T>::size());
// 	 cerr<<"inside GET"<<endl;
    return const_cast<T&>(((*this)[i]));
  }

 BasicArray<T> & operator=(const BasicArray<T> & rhs){
	if(this==&rhs) return *this;
	static_cast<std::vector<T>&>(*this)=rhs;
	return *this;
 }	
  /** 
   * Allows you to access a BasicArray like a normal array.
   * See get().
   */
//   T &operator[](const unsigned int i) const {
// 		using namespace std;
// 		cerr<<"inside GET"<<endl;
// 		return const_cast<T&>(((*((std::vector<T>)this))[i]));
// 	}
};

#endif // BASICARRAY_H
