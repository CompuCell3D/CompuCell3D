#ifndef DIFFUSABLE_VECTOR_RD_OPENCL_IMPL_H
#define DIFFUSABLE_VECTOR_RD_OPENCL_IMPL_H

#include <memory>
#include <Logger/CC3DLogger.h>
namespace CompuCell3D {

    template<typename T>
    class DiffusableVectorRDOpenCLImplFieldProxy : public Field3D<T> {
        T *m_chunkPtr;
        Dim3D const &m_dim;
    public:
        float getDirect(int x, int y, int z) const { return m_chunkPtr[z * m_dim.x * m_dim.y + y * m_dim.x + x]; }

        void setDirect(int x, int y, int z, float val) {
            m_chunkPtr[z * m_dim.x * m_dim.y + y * m_dim.x + x] = val;
        }

        float get(const Point3D &pt) const { return getDirect(pt.x, pt.y, pt.z); }

        void set(const Point3D &pt, const T val) { setDirect(pt.x, pt.y, pt.z, val); }

        Dim3D getDim() const { return m_dim; }

        Dim3D getInternalDim() const { return m_dim; }

        bool isValid(const Point3D &pt) const {
            return pt.x >= 0 && pt.x < m_dim.x &&
                   pt.y >= 0 && pt.y < m_dim.y &&
                   pt.z >= 0 && pt.z < m_dim.z;
        }

        void initializeFieldUsingEquation(std::string _expression) {
            Point3D pt;
            mu::Parser parser;
            double xVar, yVar, zVar; //variables used by parser
            try {
                parser.DefineVar("x", &xVar);
                parser.DefineVar("y", &yVar);
                parser.DefineVar("z", &zVar);

                parser.SetExpr(_expression);

                for (int x = 0; x < m_dim.x; ++x)
                    for (int y = 0; y < m_dim.y; ++y)
                        for (int z = 0; z < m_dim.z; ++z) {
                            pt.x = x;
                            pt.y = y;
                            pt.z = z;
                            //setting parser variables
                            xVar = x;
                            yVar = y;
                            zVar = z;
                            set(pt, static_cast<float>(parser.Eval()));
                        }

            } catch (mu::Parser::exception_type &e) {
                CC3D_Log(LOG_DEBUG) << e.GetMsg();
                ASSERT_OR_THROW(e.GetMsg(), 0);
            }

        }

    public:

        DiffusableVectorRDOpenCLImplFieldProxy(T *chunkPtr, Dim3D const &dim) : m_chunkPtr(chunkPtr), m_dim(dim) {}

    private:
        DiffusableVectorRDOpenCLImplFieldProxy(const DiffusableVectorRDOpenCLImplFieldProxy &);

        DiffusableVectorRDOpenCLImplFieldProxy &operator=(const DiffusableVectorRDOpenCLImplFieldProxy &);

/*	DiffusableVectorRDOpenCLImplFieldProxy(DiffusableVectorRDOpenCLImplFieldProxy const &other):
		m_chunkPtr(other.m_chunkPtr), m_dim(other.m_dim){}

	DiffusableVectorRDOpenCLImplFieldProxy& operator=(DiffusableVectorRDOpenCLImplFieldProxy other){
		swap(other);
		return *this;
	}

	void swap(DiffusableVectorRDOpenCLImplFieldProxy &other){
		std::swap(other.m_dim, m_dim);
		std::swap(other.m_chunkPtr, m_chunkPtr);
	}*/

    };


    template<typename Real_t>
    class DiffusableVectorRDOpenCLImpl {
        BoundaryStrategy const *m_boundaryStrategy;
        unsigned int m_maxNeighborIndex;
        std::vector <Real_t> m_array;
        Dim3D m_dim;
        std::vector <std::string> m_concentrationFieldNameVector;

        typedef DiffusableVectorRDOpenCLImplFieldProxy<Real_t> *unique_proxy_ptr;
        std::vector <unique_proxy_ptr> m_proxies;
    public:

        //typedef DiffusableVectorRDOpenCLImplFieldProxy<Real_t>* PtrToFieldCont_t;
        ~DiffusableVectorRDOpenCLImpl();

        DiffusableVectorRDOpenCLImpl() : m_boundaryStrategy(NULL), m_maxNeighborIndex(0) {}

        DiffusableVectorRDOpenCLImplFieldProxy<Real_t> *getConcentrationField(int idx);

        BoundaryStrategy const *getBoundaryStrategy() const { return m_boundaryStrategy; }

        unsigned int getMaxNeighborIndex() const { return m_maxNeighborIndex; }

        void
        initializeFieldUsingEquation(DiffusableVectorRDOpenCLImplFieldProxy<Real_t> *_field, std::string _expression);

        void allocateDiffusableFieldVector(unsigned int numberOfFields, Dim3D fieldDim);

        void setConcentrationFieldName(int n, std::string const &name);

        std::string getConcentrationFieldName(int n) const;

        Dim3D getDim() const { return m_dim; }

    protected:
        Real_t *getPtr() { return &m_array[0]; }

        Real_t const *getPtr() const { return &m_array[0]; }


    private:
        //it could be implemented, just don't need them now
        DiffusableVectorRDOpenCLImpl(DiffusableVectorRDOpenCLImpl const &);

        DiffusableVectorRDOpenCLImpl &operator=(DiffusableVectorRDOpenCLImpl const &);
    };

    template<typename Real_t>
    DiffusableVectorRDOpenCLImpl<Real_t>::~DiffusableVectorRDOpenCLImpl() {
        for (size_t i = 0; i < m_proxies.size(); ++i) {
            delete m_proxies[i];
        }
    }


    template<typename Real_t>
    DiffusableVectorRDOpenCLImplFieldProxy<Real_t> *
    DiffusableVectorRDOpenCLImpl<Real_t>::getConcentrationField(int idx) {
        return m_proxies[idx];
    }

    template<typename Real_t>
    void DiffusableVectorRDOpenCLImpl<Real_t>::initializeFieldUsingEquation(
            DiffusableVectorRDOpenCLImplFieldProxy<Real_t> *field, std::string expression) {
        field->initializeFieldUsingEquation(expression);
    }

    template<typename Real_t>
    void
    DiffusableVectorRDOpenCLImpl<Real_t>::allocateDiffusableFieldVector(unsigned int numberOfFields, Dim3D fieldDim) {
        m_dim = fieldDim;
        CC3D_Log(LOG_TRACE) << "***************************fieldDim************************"<<fieldDim;
        std::vector<Real_t>(fieldDim.x * fieldDim.y * fieldDim.z * numberOfFields).swap(m_array);
        m_concentrationFieldNameVector.resize(numberOfFields);

        m_boundaryStrategy = BoundaryStrategy::getInstance();
        m_maxNeighborIndex = m_boundaryStrategy->getMaxNeighborIndexFromNeighborOrderNoGen(
                1);//for nearest neighbors only

        std::vector<unique_proxy_ptr>().swap(m_proxies);
        m_proxies.reserve(numberOfFields);

        for (size_t i = 0; i < numberOfFields; ++i) {
            size_t shift = m_dim.x * m_dim.y * m_dim.z * i;
            m_proxies.push_back(
                    unique_proxy_ptr(new DiffusableVectorRDOpenCLImplFieldProxy<Real_t>(&m_array[shift], m_dim)));
        }
    }

    template<typename Real_t>
    void DiffusableVectorRDOpenCLImpl<Real_t>::setConcentrationFieldName(int n, std::string const &name) {
        m_concentrationFieldNameVector[n] = name;
    }

    template<typename Real_t>
    std::string DiffusableVectorRDOpenCLImpl<Real_t>::getConcentrationFieldName(int n) const {
        return m_concentrationFieldNameVector[n];
    }

}//namespace CompuCell3D
#endif
