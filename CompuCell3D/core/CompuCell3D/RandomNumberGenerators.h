#ifndef RANDOMNUMBERGENERATOR_H
#define RANDOMNUMBERGENERATOR_H

#include <random>
#include <string>
#include <sys/timeb.h>

#include "CC3DExceptions.h"

#define CC3DPRNG_MBIG 1000000000
#define CC3DPRNG_MSEED 161803398
#define CC3DPRNG_MZ 0
#define CC3DPRNG_FAC (1.0/CC3DPRNG_MBIG)

namespace CompuCell3D {

    /**
	Written by T.J. Sego, Ph.D.
	*/

    class RandomNumberGenerator {

        unsigned int bitBuffer;
        unsigned int bits;

    protected:

        unsigned int seed;

    public:

        RandomNumberGenerator() : seed(1), bits(0) {};
        RandomNumberGenerator(const unsigned int& seed): bits(0) {
            setSeed(seed);
        };
        virtual ~RandomNumberGenerator()=default;

        const unsigned int getSeed() const { return seed; }
        virtual void setSeed(const unsigned int& _seed) { seed = _seed; }

        bool getBool();
        long getInteger(const long& min = 0, const long& max = RAND_MAX) { return min + long(((max - min) + 1) * getRatio()); }

        virtual double getRatio() = 0;
        virtual std::string name() = 0;

    };

    class RandomNumberGeneratorMersenneT: public RandomNumberGenerator {

        std::mt19937 prng;
        unsigned int minVal;
        double valRange;

    public:
    
        RandomNumberGeneratorMersenneT();
        RandomNumberGeneratorMersenneT(const unsigned int& seed);

        void setSeed(const unsigned int& _seed);

        double getRatio();
        std::string name() { return "MersenneTwister"; }

    };

    // This implementation is based on Roeland Merks code TST 0.1.1
    class RandomNumberGeneratorLegacy: public RandomNumberGenerator {

        int idum;

    public:

        RandomNumberGeneratorLegacy();
        RandomNumberGeneratorLegacy(const unsigned int& seed);

        void setSeed(const unsigned int& _seed);

        // Knuth's substrative method, see Numerical Recipes
        double getRatio();
        
        std::string name() { return "Legacy"; }

    };

    // Random number generator factory
    class RandomNumberGeneratorFactory {

        RandomNumberGenerator* singleton;

    public:
    
        enum Type {
            DEFAULT,
            MERSENNE_TWISTER,
            LEGACY
        };

    private:

        Type type;

    public:

        RandomNumberGeneratorFactory(Type _type = DEFAULT) : type(_type), singleton(0) {
            if (type == DEFAULT) type = MERSENNE_TWISTER;
        }
        ~RandomNumberGeneratorFactory() {
            if (singleton) delete singleton;
            singleton = 0;
        }

        /**
         * @brief Generates a new random number generator. Client is responsible for deallocation. 
         * 
         * @param seed value with which to seed the random number generator
         * 
         * @return A new random number generator. 
         */
        RandomNumberGenerator* generateRandomNumberGenerator(const unsigned int& seed = 1);

        /**
         * @return Singleton of the factory. 
         */
        RandomNumberGenerator* getInstance(const unsigned int& seed = 1);

        std::string getName() {
            switch (type) {
                case MERSENNE_TWISTER:
                    return "MersenneTwister";
                case LEGACY:
                    return "Legacy";
                case DEFAULT:
                    return "MersenneTwister";
            }

            throw CC3DException(std::string("Unknown random number generator type."));
        }

    };

}

#endif //RANDOMNUMBERGENERATOR_H