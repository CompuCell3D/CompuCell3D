/*******************************************************************\

              Copyright (C) 2004 Joseph Coffland

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
#ifndef BASICRANDOMNUMBERGENERATOR_H
#define BASICRANDOMNUMBERGENERATOR_H

#include <stdlib.h>

#define MBIG 1000000000
#define MSEED 161803398
#define MZ 0
#define FAC (1.0/MBIG)

/** 
 * A C++ interface to system random number generators.
 */
class BasicRandomNumberGenerator {
  static BasicRandomNumberGenerator *singleton;

  unsigned int seed;
  int bitBuf;
  unsigned int bits;
  int idum;

public:
  BasicRandomNumberGenerator() :  bits(0)
   {
      idum=-1;
      setSeed(1);
   }

  /** 
   * Get a pointer to an instance of BasicRandomNumberGenerator.
   * Creates an instance of the default implementation.
   * 
   * @return A pointer to an instance of BasicRandomNumberGenerator.
   */
  static BasicRandomNumberGenerator *getInstance();

  /** 
   * Set the random seed.  Using the same seed will generate the
   * same sequence of random numbers.
   */
  void setSeed(const unsigned int seed);

  /** 
   * @return The seed used to initialize this sequence of random numbers.
   */  
  unsigned int getSeed() {return seed;}

  /** 
   * Returns one bit from the random buffer.
   * 
   * @return A random boolean value.
   */
  bool getBool();

  /** 
   * @param min Minimum value inclusive.
   * @param max Maximum value inclusive.
   * 
   * @return A random integer in the range [min, max].
   */
  long getInteger(const long min = 0, const long max = RAND_MAX);

  /** 
   * @return A random double in the range [0, 1).
   */
  double getRatio();

  /** 
   * @return seed
   * used to initialize randome number generator based on local time
   */

  int Randomize(void);
protected:
  virtual ~BasicRandomNumberGenerator() {}
};
///////////////////////////////////////////////
class BasicRandomNumberGeneratorNonStatic {
  

  unsigned int seed;
  int bitBuf;
  unsigned int bits;
  int idum;

public:
  BasicRandomNumberGeneratorNonStatic() :  bits(0)
   {
      idum=-1;
      setSeed(1);
	  iff=0;
   }

  /** 
   * Get a pointer to an instance of BasicRandomNumberGenerator.
   * Creates an instance of the default implementation.
   * 
   * @return A pointer to an instance of BasicRandomNumberGenerator.
   */
  BasicRandomNumberGeneratorNonStatic *getInstance();

  /** 
   * Set the random seed.  Using the same seed will generate the
   * same sequence of random numbers.
   */
  void setSeed(const unsigned int seed);

  /** 
   * @return The seed used to initialize this sequence of random numbers.
   */  
  unsigned int getSeed() {return seed;}

  /** 
   * Returns one bit from the random buffer.
   * 
   * @return A random boolean value.
   */
  bool getBool();

  /** 
   * @param min Minimum value inclusive.
   * @param max Maximum value inclusive.
   * 
   * @return A random integer in the range [min, max].
   */
  long getInteger(const long min = 0, const long max = RAND_MAX);

  /** 
   * @return A random double in the range [0, 1).
   */
  double getRatio();

  /** 
   * @return seed
   * used to initialize randome number generator based on local time
   */

  int Randomize(void);

  virtual ~BasicRandomNumberGeneratorNonStatic() {}

private:
	int inext,inextp;
	long ma[56];
	int iff;
};


#endif // BASICRANDOMNUMBERGENERATOR_H
