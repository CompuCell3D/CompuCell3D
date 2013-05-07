#ifndef rrDoubleMatrixH
#define rrDoubleMatrixH
#include <vector>
#include "rrObject.h"
#include "libstruct/matrix.h"
using std::vector;
using std::ostream;
namespace rr
{
typedef ls::DoubleMatrix DoubleMatrix;

//class RR_DECLSPEC DoubleMatrix : public rrObject
//{
//    protected:
//        unsigned            mRowCount;
//        unsigned            mColCount;
//        double*             mMatrix;
//        bool                mIsOwner;
//        const string*       mNamePtr;
//
//    public:
//                            DoubleMatrix(const unsigned& rows = 0, const unsigned& cols = 0, const string& name = "");
//                            DoubleMatrix(const DoubleMatrix& m);        // Copy constructor
//                           ~DoubleMatrix();                            // Destructor
//                            DoubleMatrix(double* ptrToArray, const unsigned& rowCount = 0, const unsigned& colCount = 0);
//
//        DoubleMatrix&       operator = (const DoubleMatrix & rhs);       // Assignment operator
//        double&             operator() (const unsigned& row, const unsigned& col);
//        double              operator() (const unsigned& row, const unsigned& col) const;
//
//        unsigned            RSize() const {return mRowCount;}
//        unsigned            CSize() const {return mColCount;}
//        bool                Allocate(unsigned rows, unsigned cols);
//        double*             GetPointer(){return mMatrix;}
//        void                SetNamePointer(const string* namePtr){mNamePtr = namePtr;}
//};

//DoubleMatrix RR_DECLSPEC GetDoubleMatrixFromPtr(double** *pointer, const int& nRows, const int& nCols);
//RR_DECLSPEC ostream& operator<<(ostream&, const DoubleMatrix& mat);
}
#endif
