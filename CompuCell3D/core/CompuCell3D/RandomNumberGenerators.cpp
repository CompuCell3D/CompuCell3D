#include "RandomNumberGenerators.h"
#include <iostream>

using namespace CompuCell3D;
using namespace std;

bool RandomNumberGenerator::getBool() {
    if (!bits) {
        bitBuffer = getInteger(0, RAND_MAX);
        bits = sizeof(int) * 8;
    }
    bool out = bool(bitBuffer & 1);
    bits--;
    bitBuffer = bitBuffer >> 1;
    return out;
}

RandomNumberGeneratorMersenneT::RandomNumberGeneratorMersenneT() {
    setSeed(1);
    minVal = prng.min();
    valRange = double(prng.max() - minVal);
};

RandomNumberGeneratorMersenneT::RandomNumberGeneratorMersenneT(const unsigned int &seed) {
    setSeed(seed);
    minVal = prng.min();
    valRange = double(prng.max() - minVal);
}


void RandomNumberGeneratorMersenneT::setSeed(const unsigned int &_seed) {
    this->prng = std::mt19937(_seed);
    seed = _seed;
}

double RandomNumberGeneratorMersenneT::getRatio() {
    return double(prng() - minVal) / valRange;
}

RandomNumberGeneratorLegacy::RandomNumberGeneratorLegacy() {
    idum = -1;
    setSeed(1);
}

RandomNumberGeneratorLegacy::RandomNumberGeneratorLegacy(const unsigned int &seed) {
    idum = -(int) seed;
    setSeed(seed);
}

void RandomNumberGeneratorLegacy::setSeed(const unsigned int &_seed) {
    seed = _seed;

    if (seed < 0) {
        struct timeb t;
        ftime(&t);
        setSeed(abs(int((t.time * t.millitm) % 655337)));
    } else {
        idum = -(int) seed;
        for (unsigned int i = 0; i < 100; i++) getRatio();
    }
}

double RandomNumberGeneratorLegacy::getRatio() {
    static int inext, inextp;
    static long ma[56];
    static int iff = 0;
    long mj, mk;
    int i, j, k;

    if (idum < 0 || iff == 0) {
        iff = 1;
        mj = labs(CC3DPRNG_MSEED - labs(idum));
        mj %= CC3DPRNG_MBIG;
        ma[55] = mj;
        mk = 1;
        i = 1;
        do {
            j = (21 * i) % 55;
            ma[j] = mk;
            mk = mj - mk;
            if (mk < CC3DPRNG_MZ) mk += CC3DPRNG_MBIG;
            mj = ma[j];
        } while (++i <= 54);
        k = 1;
        do {
            i = 1;
            do {
                ma[i] -= ma[1 + (i + 30) % 55];
                if (ma[i] < CC3DPRNG_MZ) ma[i] += CC3DPRNG_MBIG;
            } while (++i <= 55);
        } while (++k <= 4);
        inext = 0;
        inextp = 31;
        idum = 1;
    }
    if (++inext == 56) inext = 1;
    if (++inextp == 56) inextp = 1;
    mj = ma[inext] - ma[inextp];
    if (mj < CC3DPRNG_MZ) mj += CC3DPRNG_MBIG;
    ma[inext] = mj;
    return mj * CC3DPRNG_FAC;
}

RandomNumberGenerator *RandomNumberGeneratorFactory::generateRandomNumberGenerator(const unsigned int &seed) {

    switch (type) {
        case MERSENNE_TWISTER:
            return new RandomNumberGeneratorMersenneT(seed);
        case LEGACY:
            return new RandomNumberGeneratorLegacy(seed);
    }
    throw CC3DException(std::string("Unknown random number generator type."));

}

RandomNumberGenerator *RandomNumberGeneratorFactory::getInstance(const unsigned int &seed) {
    if (!singleton) singleton = generateRandomNumberGenerator(seed);
    return singleton;
}
