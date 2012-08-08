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
#include "BasicRandomNumberGenerator.h"
#include <sys/timeb.h>
#include <iostream>
using namespace std;

//This implementation is based on Roeland Merks code TST 0.1.1

BasicRandomNumberGenerator *BasicRandomNumberGenerator::singleton = NULL;

BasicRandomNumberGenerator *BasicRandomNumberGenerator::getInstance() {
  if (!singleton) singleton = new BasicRandomNumberGenerator();
  return singleton;
}

void BasicRandomNumberGenerator::setSeed(const unsigned int seed) {
//   this->seed = seed;
//   srand(seed);
  this->seed = seed;
  if (seed < 0) {
	  std::cerr << "Randomizing random generator, seed is ";
    int rseed=Randomize();
    std::cerr << rseed << "\n";
//     return rseed;
  } else {
    int i;
    idum = -seed;
    for (i=0; i <100; i++)
      getRatio();
//     return seed;
  }
}

bool BasicRandomNumberGenerator::getBool() {
  if (!bits) {
//     bitBuf = rand_r(&seed);
    bitBuf = getInteger(0, RAND_MAX);
    bits = sizeof(int) * 8;
  }

  bool value = bitBuf & 1;
  bits--;
  bitBuf = bitBuf >> 1;

  return value;
}

long BasicRandomNumberGenerator::getInteger(const long min, const long max) {
  return min + (int)(((max - min) + 1) * getRatio());
}


double BasicRandomNumberGenerator::getRatio() 
/* Knuth's substrative method, see Numerical Recipes */

{
//   return rand_r(&seed) / (RAND_MAX + 1.0);
  static int inext,inextp;
  static long ma[56];
  static int iff=0;
  long mj,mk;
  int i,ii,k;

  if (idum < 0 || iff == 0) {
    iff=1;
    //mj=MSEED-(idum < 0 ? -idum : idum); // original version
	mj=labs(MSEED-labs(idum)); //fix sent by Dan Lea
    mj %= MBIG;
    ma[55]=mj;
    mk=1;
    i=1;
    do {
      ii=(21*i) % 55;
      ma[ii]=mk;
      mk=mj-mk;
      if (mk < MZ) mk += MBIG;
      mj=ma[ii];
    } while ( ++i <= 54 );
    k=1;
    do {
      i=1;
      do {
        ma[i] -= ma[1+(i+30) % 55];
        if (ma[i] < MZ) ma[i] += MBIG;
      } while ( ++i <= 55 );
    } while ( ++k <= 4 );
    inext=0;
    inextp=31;
    idum=1;
  }
  if (++inext == 56) inext=1;
  if (++inextp == 56) inextp=1;
  mj=ma[inext]-ma[inextp];
  if (mj < MZ) mj += MBIG;
  ma[inext]=mj;
//    cerr<<"ratio="<<mj*FAC<<endl;
  return mj*FAC;
}

int BasicRandomNumberGenerator::Randomize(void) {
  
  // Set the seed according to the local time
  struct timeb t;
  int seed;

  ftime(&t);
  
  seed=abs((int)((t.time*t.millitm)%655337));
  setSeed(seed);
//   fprintf(stderr,"Random seed is %d\n",seed);
  return seed;
}


///////////////////////////////////////////


BasicRandomNumberGeneratorNonStatic *BasicRandomNumberGeneratorNonStatic::getInstance() {
  
  return this;
}

void BasicRandomNumberGeneratorNonStatic::setSeed(const unsigned int seed) {
//   this->seed = seed;
//   srand(seed);
  this->seed = seed;
  if (seed < 0) {
	  std::cerr << "Randomizing random generator, seed is ";
    int rseed=Randomize();
    std::cerr << rseed << "\n";
//     return rseed;
  } else {
    int i;
    idum = -seed;
    for (i=0; i <100; i++)
      getRatio();
//     return seed;
  }
}

bool BasicRandomNumberGeneratorNonStatic::getBool() {
  if (!bits) {
//     bitBuf = rand_r(&seed);
    bitBuf = getInteger(0, RAND_MAX);
    bits = sizeof(int) * 8;
  }

  bool value = bitBuf & 1;
  bits--;
  bitBuf = bitBuf >> 1;

  return value;
}

long BasicRandomNumberGeneratorNonStatic::getInteger(const long min, const long max) {
  return min + (int)(((max - min) + 1) * getRatio());
}


double BasicRandomNumberGeneratorNonStatic::getRatio() 
/* Knuth's substrative method, see Numerical Recipes */

{
//   return rand_r(&seed) / (RAND_MAX + 1.0);
  //static int inext,inextp;
  //static long ma[56];
  //static int iff=0;
  long mj,mk;
  int i,ii,k;

  if (idum < 0 || iff == 0) {
    iff=1;
    //mj=MSEED-(idum < 0 ? -idum : idum); // original version
	mj=labs(MSEED-labs(idum)); //fix sent by Dan Lea
    mj %= MBIG;
    ma[55]=mj;
    mk=1;
    i=1;
    do {
      ii=(21*i) % 55;
      ma[ii]=mk;
      mk=mj-mk;
      if (mk < MZ) mk += MBIG;
      mj=ma[ii];
    } while ( ++i <= 54 );
    k=1;
    do {
      i=1;
      do {
        ma[i] -= ma[1+(i+30) % 55];
        if (ma[i] < MZ) ma[i] += MBIG;
      } while ( ++i <= 55 );
    } while ( ++k <= 4 );
    inext=0;
    inextp=31;
    idum=1;
  }
  if (++inext == 56) inext=1;
  if (++inextp == 56) inextp=1;
  mj=ma[inext]-ma[inextp];
  if (mj < MZ) mj += MBIG;
  ma[inext]=mj;
//    cerr<<"ratio="<<mj*FAC<<endl;
  return mj*FAC;
}

int BasicRandomNumberGeneratorNonStatic::Randomize(void) {
  
  // Set the seed according to the local time
  struct timeb t;
  int seed;

  ftime(&t);
  
  seed=abs((int)((t.time*t.millitm)%655337));
  setSeed(seed);
//   fprintf(stderr,"Random seed is %d\n",seed);
  return seed;
}
