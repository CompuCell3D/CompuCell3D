#ifndef ALGO_H
#define ALGO_H

#include "CompuCell3D/Field3D/Dim3D.h"
#include <functional>

template<class Fn_t>
inline
void for_each(CompuCell3D::Dim3D dim, Fn_t fn) {
    for (int zi = 1; zi <= dim.z; ++zi) {
        for (int yi = 1; yi <= dim.y; ++yi) {
            for (int xi = 1; xi <= dim.x; ++xi) {
                if (!fn(xi, yi, zi, zi * (dim.x + 2) * (dim.y + 2) + yi * (dim.x + 2) + xi))
                    goto exit;
            }
        }
    }
    exit:;
}


template<class T>
inline
std::pair <size_t, T> max(viennacl::vector < T >
const &v){
std::vector <T> h_v(v.size());
viennacl::copy(v, h_v
);
std::vector<T>::const_iterator it = std::max_element(h_v.begin(), h_v.end());
return
std::make_pair(std::distance(h_v.cbegin(), it), *it
);
}

template<class T>
inline
std::pair <size_t, T> min(viennacl::vector < T >
const &v){
std::vector <T> h_v(v.size());
viennacl::copy(v, h_v
);
std::vector<T>::const_iterator it = std::min_element(h_v.begin(), h_v.end());
return
std::make_pair(std::distance(h_v.cbegin(), it), *it
);
}

template<class T>
inline
std::pair<bool, size_t> isFinite(viennacl::vector < T >
const &v){
std::vector <T> h_v(v.size());
viennacl::copy(v, h_v
);
auto notFiniteWrap = std::not1(std::ptr_fun(_finite));
auto itf = std::find_if(h_v.begin(), h_v.end(), notFiniteWrap);
return
std::make_pair(itf
==h_v.

end(), std::distance(h_v.begin(), itf)

);
}

#endif