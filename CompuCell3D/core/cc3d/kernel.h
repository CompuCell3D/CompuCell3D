#ifndef CC3D_KERNEL_H
#define CC3D_KERNEL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define CC3D_KERNEL_ABI_VERSION 9u

#if defined(_WIN32) || defined(__CYGWIN__)
#define CC3D_STEPPABLE_EXPORT __declspec(dllexport)
#else
#define CC3D_STEPPABLE_EXPORT
#endif

#ifdef __cplusplus
namespace cc3d {
namespace kernel {
class Cell;
class CellIterator;
struct CellSentinel;
class CellNeighbor;
class CellNeighborIterator;
class CellNeighborRange;
struct CellNeighborSentinel;
class PixelIterator;
class PixelRange;
struct PixelSentinel;
class OptionalCell;
class ScalarField;
class SBMLModel;
struct Dim3;
}
}

extern "C" {
#endif

typedef void *CC3DCellIteratorHandle;
typedef void *CC3DCellNeighborIteratorHandle;
typedef void *CC3DPixelIteratorHandle;
typedef void *CC3DFieldHandle;
typedef void *CC3DCellHandle;
typedef void *CC3DSBMLModelHandle;

typedef struct CC3DCellViewV1 {
    CC3DCellHandle handle;
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

typedef struct CC3DCellNeighborViewV1 {
    CC3DCellViewV1 neighbor;
    int commonSurfaceArea;
} CC3DCellNeighborViewV1;

typedef struct CC3DCellNeighborsAPIV1 {
    void *userdata;
    CC3DCellNeighborIteratorHandle (*begin_fn)(void *userdata, CC3DCellHandle cell);
    uint8_t (*valid_fn)(void *userdata, CC3DCellNeighborIteratorHandle iterator);
    CC3DCellNeighborViewV1 (*get_fn)(void *userdata, CC3DCellNeighborIteratorHandle iterator);
    void (*next_fn)(void *userdata, CC3DCellNeighborIteratorHandle iterator);
    void (*destroy_fn)(void *userdata, CC3DCellNeighborIteratorHandle iterator);
#ifdef __cplusplus
    inline ::cc3d::kernel::CellNeighborRange forCell(const ::cc3d::kernel::Cell &cell) const;
#endif
} CC3DCellNeighborsAPIV1;

typedef struct CC3DPixelViewV1 {
    int x;
    int y;
    int z;
} CC3DPixelViewV1;

typedef struct CC3DCellPixelsAPIV1 {
    void *userdata;
    CC3DPixelIteratorHandle (*begin_fn)(void *userdata, CC3DCellHandle cell);
    uint8_t (*valid_fn)(void *userdata, CC3DPixelIteratorHandle iterator);
    CC3DPixelViewV1 (*get_fn)(void *userdata, CC3DPixelIteratorHandle iterator);
    void (*next_fn)(void *userdata, CC3DPixelIteratorHandle iterator);
    void (*destroy_fn)(void *userdata, CC3DPixelIteratorHandle iterator);
#ifdef __cplusplus
    inline ::cc3d::kernel::PixelRange forCell(const ::cc3d::kernel::Cell &cell) const;
#endif
} CC3DCellPixelsAPIV1;

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
    void (*assign_fn)(void *userdata, int x_min, int x_max, int y_min, int y_max, int z_min, int z_max, CC3DCellHandle cell);
#ifdef __cplusplus
    inline ::cc3d::kernel::Dim3 dim() const;
    inline ::cc3d::kernel::OptionalCell operator()(int x, int y, int z) const;
    inline void assign(int x_min, int x_max, int y_min, int y_max, int z_min, int z_max, const ::cc3d::kernel::Cell &cell) const;
#endif
} CC3DCellFieldAPIV1;

typedef struct CC3DSBMLAPIV1 {
    void *userdata;
    CC3DSBMLModelHandle (*find_model_fn)(void *userdata, CC3DCellHandle cell, const char *model_name);
    void (*destroy_model_fn)(void *userdata, CC3DSBMLModelHandle model);
    uint8_t (*has_value_fn)(void *userdata, CC3DSBMLModelHandle model, const char *value_name);
    uint8_t (*get_value_fn)(void *userdata, CC3DSBMLModelHandle model, const char *value_name, double *value);
    uint8_t (*set_value_fn)(void *userdata, CC3DSBMLModelHandle model, const char *value_name, double value);
    uint8_t (*timestep_fn)(void *userdata, CC3DSBMLModelHandle model);
#ifdef __cplusplus
    inline ::cc3d::kernel::SBMLModel find(const ::cc3d::kernel::Cell &cell, const char *model_name) const;
#endif
} CC3DSBMLAPIV1;
typedef struct CC3DCellFactoryAPIV1 {
    void *userdata;
    CC3DCellViewV1 (*create_fn)(void *userdata, unsigned char cell_type);
#ifdef __cplusplus
    inline ::cc3d::kernel::Cell create(unsigned char cell_type) const;
#endif
} CC3DCellFactoryAPIV1;

typedef struct CC3DKernelContext {
    uint32_t abiVersion;
    uint64_t mcs;
    CC3DCellsAPIV1 cells;
    CC3DCellNeighborsAPIV1 neighbors;
    CC3DCellPixelsAPIV1 pixels;
    CC3DCellPixelsAPIV1 boundaryPixels;
    CC3DScalarFieldsAPIV1 scalarFields;
    CC3DCellFieldAPIV1 cellField;
    CC3DCellFactoryAPIV1 cellFactory;
    CC3DSBMLAPIV1 sbml;
#ifdef __cplusplus
    inline ::cc3d::kernel::Cell newCell(unsigned char cell_type) const;
#endif
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

struct CellNeighborSentinel {};

struct PixelSentinel {};

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
        : handle_(view.handle),
          id(view.id),
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

    CC3DCellHandle handle() const { return handle_; }

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

private:
    CC3DCellHandle handle_;
};

class CellNeighbor {
public:
    CellNeighbor() : neighbor_(CC3DCellViewV1{}), commonSurfaceArea(0) {}

    explicit CellNeighbor(const CC3DCellNeighborViewV1 &view)
        : neighbor_(view.neighbor), commonSurfaceArea(view.commonSurfaceArea) {}

    explicit operator bool() const { return neighbor_.handle() != nullptr; }

    Cell &neighbor() { return neighbor_; }
    const Cell &neighbor() const { return neighbor_; }

    int commonSurfaceArea;

private:
    Cell neighbor_;
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

class CellNeighborIterator {
public:
    CellNeighborIterator() : api_(nullptr), handle_(nullptr) {}

    CellNeighborIterator(const CC3DCellNeighborsAPIV1 *api, CC3DCellNeighborIteratorHandle handle)
        : api_(api), handle_(handle) {}

    CellNeighborIterator(const CellNeighborIterator &) = delete;
    CellNeighborIterator &operator=(const CellNeighborIterator &) = delete;

    CellNeighborIterator(CellNeighborIterator &&other) noexcept
        : api_(other.api_), handle_(other.handle_) {
        other.api_ = nullptr;
        other.handle_ = nullptr;
    }

    CellNeighborIterator &operator=(CellNeighborIterator &&other) noexcept {
        if (this != &other) {
            cleanup();
            api_ = other.api_;
            handle_ = other.handle_;
            other.api_ = nullptr;
            other.handle_ = nullptr;
        }
        return *this;
    }

    ~CellNeighborIterator() { cleanup(); }

    CellNeighbor operator*() const { return CellNeighbor(api_->get_fn(api_->userdata, handle_)); }

    CellNeighborIterator &operator++() {
        api_->next_fn(api_->userdata, handle_);
        return *this;
    }

    bool operator!=(CellNeighborSentinel) const {
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

    const CC3DCellNeighborsAPIV1 *api_;
    CC3DCellNeighborIteratorHandle handle_;
};

class CellNeighborRange {
public:
    CellNeighborRange() : api_(nullptr), cell_(nullptr) {}

    CellNeighborRange(const CC3DCellNeighborsAPIV1 *api, CC3DCellHandle cell)
        : api_(api), cell_(cell) {}

    CellNeighborIterator begin() const {
        if (!api_ || !api_->begin_fn || !cell_) {
            return CellNeighborIterator();
        }
        return CellNeighborIterator(api_, api_->begin_fn(api_->userdata, cell_));
    }

    CellNeighborSentinel end() const { return CellNeighborSentinel(); }

private:
    const CC3DCellNeighborsAPIV1 *api_;
    CC3DCellHandle cell_;
};


class Pixel {
public:
    Pixel() : x(0), y(0), z(0) {}
    explicit Pixel(const CC3DPixelViewV1 &view) : x(view.x), y(view.y), z(view.z) {}

    int x;
    int y;
    int z;
};

class PixelIterator {
public:
    PixelIterator() : api_(nullptr), handle_(nullptr) {}

    PixelIterator(const CC3DCellPixelsAPIV1 *api, CC3DPixelIteratorHandle handle)
        : api_(api), handle_(handle) {}

    PixelIterator(const PixelIterator &) = delete;
    PixelIterator &operator=(const PixelIterator &) = delete;

    PixelIterator(PixelIterator &&other) noexcept
        : api_(other.api_), handle_(other.handle_) {
        other.api_ = nullptr;
        other.handle_ = nullptr;
    }

    PixelIterator &operator=(PixelIterator &&other) noexcept {
        if (this != &other) {
            cleanup();
            api_ = other.api_;
            handle_ = other.handle_;
            other.api_ = nullptr;
            other.handle_ = nullptr;
        }
        return *this;
    }

    ~PixelIterator() { cleanup(); }

    Pixel operator*() const { return Pixel(api_->get_fn(api_->userdata, handle_)); }

    PixelIterator &operator++() {
        api_->next_fn(api_->userdata, handle_);
        return *this;
    }

    bool operator!=(PixelSentinel) const {
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

    const CC3DCellPixelsAPIV1 *api_;
    CC3DPixelIteratorHandle handle_;
};

class PixelRange {
public:
    PixelRange() : api_(nullptr), cell_(nullptr) {}

    PixelRange(const CC3DCellPixelsAPIV1 *api, CC3DCellHandle cell)
        : api_(api), cell_(cell) {}

    PixelIterator begin() const {
        if (!api_ || !api_->begin_fn || !cell_) {
            return PixelIterator();
        }
        return PixelIterator(api_, api_->begin_fn(api_->userdata, cell_));
    }

    PixelSentinel end() const { return PixelSentinel(); }

private:
    const CC3DCellPixelsAPIV1 *api_;
    CC3DCellHandle cell_;
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


class SBMLModel {
public:
    SBMLModel() : api_(nullptr), handle_(nullptr) {}

    SBMLModel(const CC3DSBMLAPIV1 *api, CC3DSBMLModelHandle handle)
        : api_(api), handle_(handle) {}

    SBMLModel(const SBMLModel &) = delete;
    SBMLModel &operator=(const SBMLModel &) = delete;

    SBMLModel(SBMLModel &&other) noexcept
        : api_(other.api_), handle_(other.handle_) {
        other.api_ = nullptr;
        other.handle_ = nullptr;
    }

    SBMLModel &operator=(SBMLModel &&other) noexcept {
        if (this != &other) {
            cleanup();
            api_ = other.api_;
            handle_ = other.handle_;
            other.api_ = nullptr;
            other.handle_ = nullptr;
        }
        return *this;
    }

    ~SBMLModel() { cleanup(); }

    explicit operator bool() const { return api_ && handle_; }

    bool has(const char *value_name) const {
        return api_ && handle_ && api_->has_value_fn && api_->has_value_fn(api_->userdata, handle_, value_name) != 0;
    }

    bool get(const char *value_name, double *value) const {
        return api_ && handle_ && value && api_->get_value_fn && api_->get_value_fn(api_->userdata, handle_, value_name, value) != 0;
    }

    double get(const char *value_name, double default_value = 0.0) const {
        double value = default_value;
        get(value_name, &value);
        return value;
    }

    bool set(const char *value_name, double value) const {
        return api_ && handle_ && api_->set_value_fn && api_->set_value_fn(api_->userdata, handle_, value_name, value) != 0;
    }

    bool timestep() const {
        return api_ && handle_ && api_->timestep_fn && api_->timestep_fn(api_->userdata, handle_) != 0;
    }

private:
    void cleanup() {
        if (api_ && handle_ && api_->destroy_model_fn) {
            api_->destroy_model_fn(api_->userdata, handle_);
        }
        api_ = nullptr;
        handle_ = nullptr;
    }

    const CC3DSBMLAPIV1 *api_;
    CC3DSBMLModelHandle handle_;
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

inline ::cc3d::kernel::CellNeighborRange CC3DCellNeighborsAPIV1::forCell(
    const ::cc3d::kernel::Cell &cell
) const {
    return ::cc3d::kernel::CellNeighborRange(this, cell.handle());
}


inline ::cc3d::kernel::PixelRange CC3DCellPixelsAPIV1::forCell(
    const ::cc3d::kernel::Cell &cell
) const {
    return ::cc3d::kernel::PixelRange(this, cell.handle());
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

inline void CC3DCellFieldAPIV1::assign(
    int x_min,
    int x_max,
    int y_min,
    int y_max,
    int z_min,
    int z_max,
    const ::cc3d::kernel::Cell &cell
) const {
    if (!assign_fn) {
        return;
    }
    assign_fn(userdata, x_min, x_max, y_min, y_max, z_min, z_max, cell.handle());
}

inline ::cc3d::kernel::SBMLModel CC3DSBMLAPIV1::find(
    const ::cc3d::kernel::Cell &cell,
    const char *model_name
) const {
    if (!find_model_fn || !model_name) {
        return ::cc3d::kernel::SBMLModel();
    }
    return ::cc3d::kernel::SBMLModel(this, find_model_fn(userdata, cell.handle(), model_name));
}
inline ::cc3d::kernel::Cell CC3DCellFactoryAPIV1::create(unsigned char cell_type) const {
    if (!create_fn) {
        return ::cc3d::kernel::Cell(CC3DCellViewV1{});
    }
    return ::cc3d::kernel::Cell(create_fn(userdata, cell_type));
}

inline ::cc3d::kernel::Cell CC3DKernelContext::newCell(unsigned char cell_type) const {
    return cellFactory.create(cell_type);
}
#endif

#endif
