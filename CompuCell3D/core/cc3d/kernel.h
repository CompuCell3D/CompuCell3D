#ifndef CC3D_KERNEL_H
#define CC3D_KERNEL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define CC3D_KERNEL_ABI_VERSION 5u

#ifdef __cplusplus
namespace cc3d {
namespace kernel {
class CellIterator;
struct CellSentinel;
class OptionalCell;
class ScalarField;
struct Dim3;
}
}

extern "C" {
#endif

typedef void *CC3DCellIteratorHandle;
typedef void *CC3DFieldHandle;

typedef struct CC3DCellViewV1 {
    const long *id;
    const long *clusterId;
    const long *volume;
    const double *surface;
    const double *clusterSurface;
    const unsigned char *type;
    unsigned char *subtype;
    float *targetVolume;
    float *lambdaVolume;
    float *targetSurface;
    float *angle;
    float *lambdaSurface;
    float *targetClusterSurface;
    float *lambdaClusterSurface;
    const double *xCM;
    const double *yCM;
    const double *zCM;
    const double *xCOM;
    const double *yCOM;
    const double *zCOM;
    const double *xCOMPrev;
    const double *yCOMPrev;
    const double *zCOMPrev;
    const double *iXX;
    const double *iXY;
    const double *iXZ;
    const double *iYY;
    const double *iYZ;
    const double *iZZ;
    const float *lX;
    const float *lY;
    const float *lZ;
    const float *ecc;
    float *lambdaVecX;
    float *lambdaVecY;
    float *lambdaVecZ;
    unsigned char *flag;
    float *averageConcentration;
    double *fluctAmpl;
    double *lambdaMotility;
    double *biasVecX;
    double *biasVecY;
    double *biasVecZ;
    bool *connectivityOn;
} CC3DCellViewV1;

typedef struct CC3DCellsAPIV1 {
    void *userdata;
    CC3DCellIteratorHandle (*begin_fn)(void *userdata);
    uint8_t (*valid_fn)(void *userdata, CC3DCellIteratorHandle iterator);
    CC3DCellViewV1 (*get_fn)(void *userdata, CC3DCellIteratorHandle iterator);
    void (*next_fn)(void *userdata, CC3DCellIteratorHandle iterator);
    void (*destroy_fn)(void *userdata, CC3DCellIteratorHandle iterator);
#ifdef __cplusplus
    inline ::cc3d::kernel::CellIterator begin() const;
    inline ::cc3d::kernel::CellSentinel end() const;
#endif
} CC3DCellsAPIV1;

typedef struct CC3DFieldDimV1 {
    int x;
    int y;
    int z;
} CC3DFieldDimV1;

typedef struct CC3DOptionalCellViewV1 {
    uint8_t hasCell;
    CC3DCellViewV1 cell;
} CC3DOptionalCellViewV1;

typedef struct CC3DScalarFieldsAPIV1 {
    void *userdata;
    CC3DFieldHandle (*find_fn)(void *userdata, const char *name);
    CC3DFieldDimV1 (*dim_fn)(void *userdata, CC3DFieldHandle field);
    float (*get_fn)(void *userdata, CC3DFieldHandle field, int x, int y, int z);
    void (*set_fn)(void *userdata, CC3DFieldHandle field, int x, int y, int z, float value);
#ifdef __cplusplus
    inline ::cc3d::kernel::ScalarField operator[](const char *name) const;
#endif
} CC3DScalarFieldsAPIV1;

typedef struct CC3DCellFieldAPIV1 {
    void *userdata;
    CC3DFieldDimV1 (*dim_fn)(void *userdata);
    CC3DOptionalCellViewV1 (*get_fn)(void *userdata, int x, int y, int z);
#ifdef __cplusplus
    inline ::cc3d::kernel::Dim3 dim() const;
    inline ::cc3d::kernel::OptionalCell operator()(int x, int y, int z) const;
#endif
} CC3DCellFieldAPIV1;

typedef struct CC3DKernelContext {
    uint32_t abiVersion;
    uint64_t mcs;
    CC3DCellsAPIV1 cells;
    CC3DScalarFieldsAPIV1 scalarFields;
    CC3DCellFieldAPIV1 cellField;
} CC3DKernelContext;

typedef struct CC3DSteppableV1 {
    uint32_t abiVersion;
    void *(*create_fn)();
    void (*start_fn)(CC3DKernelContext *ctx, void *state);
    void (*step_fn)(CC3DKernelContext *ctx, void *state);
    void (*finish_fn)(CC3DKernelContext *ctx, void *state);
    void (*destroy_fn)(void *state);
} CC3DSteppableV1;

typedef const CC3DSteppableV1 *(*CC3DGetSteppableV1Fn)();

#ifdef __cplusplus
}

namespace cc3d {
namespace kernel {

struct CellSentinel {};

struct Dim3 {
    int x;
    int y;
    int z;
};

template<typename T>
class ReadOnlyProperty {
public:
    ReadOnlyProperty() : ptr_(nullptr) {}
    explicit ReadOnlyProperty(const T *ptr) : ptr_(ptr) {}

    operator T() const { return ptr_ ? *ptr_ : T(); }

private:
    const T *ptr_;
};

template<typename T>
class Property {
public:
    Property() : ptr_(nullptr) {}
    explicit Property(T *ptr) : ptr_(ptr) {}

    operator T() const { return ptr_ ? *ptr_ : T(); }

    Property &operator=(T value) {
        if (ptr_) {
            *ptr_ = value;
        }
        return *this;
    }

    Property &operator+=(T value) {
        if (ptr_) {
            *ptr_ += value;
        }
        return *this;
    }

    Property &operator-=(T value) {
        if (ptr_) {
            *ptr_ -= value;
        }
        return *this;
    }

private:
    T *ptr_;
};

class Cell {
public:
    explicit Cell(const CC3DCellViewV1 &view)
        : id(view.id),
          clusterId(view.clusterId),
          volume(view.volume),
          surface(view.surface),
          clusterSurface(view.clusterSurface),
          type(view.type),
          subtype(view.subtype),
          targetVolume(view.targetVolume),
          lambdaVolume(view.lambdaVolume),
          targetSurface(view.targetSurface),
          angle(view.angle),
          lambdaSurface(view.lambdaSurface),
          targetClusterSurface(view.targetClusterSurface),
          lambdaClusterSurface(view.lambdaClusterSurface),
          xCM(view.xCM),
          yCM(view.yCM),
          zCM(view.zCM),
          xCOM(view.xCOM),
          yCOM(view.yCOM),
          zCOM(view.zCOM),
          xCOMPrev(view.xCOMPrev),
          yCOMPrev(view.yCOMPrev),
          zCOMPrev(view.zCOMPrev),
          iXX(view.iXX),
          iXY(view.iXY),
          iXZ(view.iXZ),
          iYY(view.iYY),
          iYZ(view.iYZ),
          iZZ(view.iZZ),
          lX(view.lX),
          lY(view.lY),
          lZ(view.lZ),
          ecc(view.ecc),
          lambdaVecX(view.lambdaVecX),
          lambdaVecY(view.lambdaVecY),
          lambdaVecZ(view.lambdaVecZ),
          flag(view.flag),
          averageConcentration(view.averageConcentration),
          fluctAmpl(view.fluctAmpl),
          lambdaMotility(view.lambdaMotility),
          biasVecX(view.biasVecX),
          biasVecY(view.biasVecY),
          biasVecZ(view.biasVecZ),
          connectivityOn(view.connectivityOn) {}

    ReadOnlyProperty<long> id;
    ReadOnlyProperty<long> clusterId;
    ReadOnlyProperty<long> volume;
    ReadOnlyProperty<double> surface;
    ReadOnlyProperty<double> clusterSurface;
    ReadOnlyProperty<unsigned char> type;
    Property<unsigned char> subtype;
    Property<float> targetVolume;
    Property<float> lambdaVolume;
    Property<float> targetSurface;
    Property<float> angle;
    Property<float> lambdaSurface;
    Property<float> targetClusterSurface;
    Property<float> lambdaClusterSurface;
    ReadOnlyProperty<double> xCM;
    ReadOnlyProperty<double> yCM;
    ReadOnlyProperty<double> zCM;
    ReadOnlyProperty<double> xCOM;
    ReadOnlyProperty<double> yCOM;
    ReadOnlyProperty<double> zCOM;
    ReadOnlyProperty<double> xCOMPrev;
    ReadOnlyProperty<double> yCOMPrev;
    ReadOnlyProperty<double> zCOMPrev;
    ReadOnlyProperty<double> iXX;
    ReadOnlyProperty<double> iXY;
    ReadOnlyProperty<double> iXZ;
    ReadOnlyProperty<double> iYY;
    ReadOnlyProperty<double> iYZ;
    ReadOnlyProperty<double> iZZ;
    ReadOnlyProperty<float> lX;
    ReadOnlyProperty<float> lY;
    ReadOnlyProperty<float> lZ;
    ReadOnlyProperty<float> ecc;
    Property<float> lambdaVecX;
    Property<float> lambdaVecY;
    Property<float> lambdaVecZ;
    Property<unsigned char> flag;
    Property<float> averageConcentration;
    Property<double> fluctAmpl;
    Property<double> lambdaMotility;
    Property<double> biasVecX;
    Property<double> biasVecY;
    Property<double> biasVecZ;
    Property<bool> connectivityOn;
};

class OptionalCell {
public:
    OptionalCell() : hasValue_(false), value_(CC3DCellViewV1{}) {}

    explicit OptionalCell(const CC3DOptionalCellViewV1 &view)
        : hasValue_(view.hasCell != 0), value_(view.cell) {}

    explicit operator bool() const { return hasValue_; }

    Cell &value() { return value_; }
    const Cell &value() const { return value_; }

private:
    bool hasValue_;
    Cell value_;
};

class CellIterator {
public:
    CellIterator() : api_(nullptr), handle_(nullptr) {}

    CellIterator(const CC3DCellsAPIV1 *api, CC3DCellIteratorHandle handle)
        : api_(api), handle_(handle) {}

    CellIterator(const CellIterator &) = delete;
    CellIterator &operator=(const CellIterator &) = delete;

    CellIterator(CellIterator &&other) noexcept
        : api_(other.api_), handle_(other.handle_) {
        other.api_ = nullptr;
        other.handle_ = nullptr;
    }

    CellIterator &operator=(CellIterator &&other) noexcept {
        if (this != &other) {
            cleanup();
            api_ = other.api_;
            handle_ = other.handle_;
            other.api_ = nullptr;
            other.handle_ = nullptr;
        }
        return *this;
    }

    ~CellIterator() { cleanup(); }

    Cell operator*() const { return Cell(api_->get_fn(api_->userdata, handle_)); }

    CellIterator &operator++() {
        api_->next_fn(api_->userdata, handle_);
        return *this;
    }

    bool operator!=(CellSentinel) const {
        return api_ && handle_ && api_->valid_fn(api_->userdata, handle_) != 0;
    }

private:
    void cleanup() {
        if (api_ && handle_ && api_->destroy_fn) {
            api_->destroy_fn(api_->userdata, handle_);
        }
        api_ = nullptr;
        handle_ = nullptr;
    }

    const CC3DCellsAPIV1 *api_;
    CC3DCellIteratorHandle handle_;
};

class ScalarFieldValue {
public:
    ScalarFieldValue(const CC3DScalarFieldsAPIV1 *api, CC3DFieldHandle field, int x, int y, int z)
        : api_(api), field_(field), x_(x), y_(y), z_(z) {}

    operator float() const {
        if (!api_ || !field_ || !api_->get_fn) {
            return 0.0f;
        }
        return api_->get_fn(api_->userdata, field_, x_, y_, z_);
    }

    ScalarFieldValue &operator=(float value) {
        if (api_ && field_ && api_->set_fn) {
            api_->set_fn(api_->userdata, field_, x_, y_, z_, value);
        }
        return *this;
    }

    ScalarFieldValue &operator+=(float value) {
        *this = static_cast<float>(*this) + value;
        return *this;
    }

    ScalarFieldValue &operator-=(float value) {
        *this = static_cast<float>(*this) - value;
        return *this;
    }

private:
    const CC3DScalarFieldsAPIV1 *api_;
    CC3DFieldHandle field_;
    int x_;
    int y_;
    int z_;
};

class ScalarField {
public:
    ScalarField() : api_(nullptr), field_(nullptr) {}
    ScalarField(const CC3DScalarFieldsAPIV1 *api, CC3DFieldHandle field)
        : api_(api), field_(field) {}

    explicit operator bool() const { return api_ && field_; }

    Dim3 dim() const {
        if (!api_ || !field_ || !api_->dim_fn) {
            return {0, 0, 0};
        }
        CC3DFieldDimV1 d = api_->dim_fn(api_->userdata, field_);
        return {d.x, d.y, d.z};
    }

    ScalarFieldValue operator()(int x, int y, int z) const {
        return ScalarFieldValue(api_, field_, x, y, z);
    }

private:
    const CC3DScalarFieldsAPIV1 *api_;
    CC3DFieldHandle field_;
};

} // namespace kernel
} // namespace cc3d

inline ::cc3d::kernel::CellIterator CC3DCellsAPIV1::begin() const {
    if (!begin_fn) {
        return ::cc3d::kernel::CellIterator();
    }
    return ::cc3d::kernel::CellIterator(this, begin_fn(userdata));
}

inline ::cc3d::kernel::CellSentinel CC3DCellsAPIV1::end() const {
    return ::cc3d::kernel::CellSentinel();
}

inline ::cc3d::kernel::ScalarField CC3DScalarFieldsAPIV1::operator[](const char *name) const {
    if (!find_fn || !name) {
        return ::cc3d::kernel::ScalarField();
    }
    return ::cc3d::kernel::ScalarField(this, find_fn(userdata, name));
}

inline ::cc3d::kernel::Dim3 CC3DCellFieldAPIV1::dim() const {
    if (!dim_fn) {
        return ::cc3d::kernel::Dim3{0, 0, 0};
    }
    CC3DFieldDimV1 d = dim_fn(userdata);
    return ::cc3d::kernel::Dim3{d.x, d.y, d.z};
}

inline ::cc3d::kernel::OptionalCell CC3DCellFieldAPIV1::operator()(int x, int y, int z) const {
    if (!get_fn) {
        return ::cc3d::kernel::OptionalCell();
    }
    return ::cc3d::kernel::OptionalCell(get_fn(userdata, x, y, z));
}
#endif

#endif
