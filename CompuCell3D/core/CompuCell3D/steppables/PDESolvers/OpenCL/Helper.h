#ifndef HELPER_H
#define HELPER_H

#include <CompuCell3D/Field3D/Dim3D.h>
#include <numeric>

inline
CompuCell3D::Dim3D d3From1d(int ind1d, CompuCell3D::Dim3D dim3d) {
    int z = ind1d / (dim3d.x * dim3d.y);
    ind1d -= z * dim3d.x * dim3d.y;
    int y = ind1d / dim3d.x;
    ind1d -= y * dim3d.x;
    return CompuCell3D::Dim3D(ind1d, y, z);
}

template<typename T>
inline
void analyze(const char *msg, viennacl::vector <T> const &v, CompuCell3D::Dim3D dim3d, int fields) {

    typedef std::vector <T> vct_t;

    int fieldLen = v.size() / fields;
    if (msg)
        std::cout << msg;
    for (int i = 0; i < fields; ++i) {
        typedef std::vector <T> vct_t;
        vct_t host_out(fieldLen);
        viennacl::copy(v.begin() + fieldLen * i, v.begin() + fieldLen * (i + 1), host_out.begin());
        //oclHelper.ReadBuffer(out, &host_out[0], totalLength);

        //std::pair<vct_t::const_iterator, vct_t::const_iterator> mnmx=std::minmax_element(host_out.begin(), host_out.end());
        //
        // change to compile with gcc 4.7.1 on Mac OS X 10.8.x:
        //    "typename" is required to inform the compiler that it'll be a type and not a member variable:
        typename vct_t::iterator mn = std::min_element(host_out.begin(), host_out.end());
        typename vct_t::iterator mx = std::max_element(host_out.begin(), host_out.end());
        CompuCell3D::Dim3D mnInd3d = d3From1d(std::distance(host_out.begin(), mn), dim3d);
        CompuCell3D::Dim3D mxInd3d = d3From1d(std::distance(host_out.begin(), mx), dim3d);

        double sum = std::accumulate(host_out.begin(), host_out.end(), 0.);
        std::cout << "\t";
        if (fields > 1)
            std::cout << "field #" << i << ": ";
        std::cout << "min element: " << *mn << "; " << mnInd3d << " max element: " << *mx << " " << mxInd3d << "; sum: "
                  << sum;
        if (fields > 1 && i != fields - 1)
            std::cout << std::endl;
    }
}

#endif