#ifndef ALGORITHMS_CC3D_H
#define ALGORITHMS_CC3D_H

#include <algorithm>

template<typename Container_t, typename RandGen_t>
void shuffleSTLContainer(Container_t & _container, RandGen_t & _randGen){
    
    std::size_t containerSize = _container.size();
	 for (std::size_t i = containerSize-1; i>0; --i) {
		// swap (_container[i],_container[_randGen.getInteger(0, i+1)]);
        swap (_container[i],_container[_randGen.getInteger(0, i)]);
	 }
    
}

#endif