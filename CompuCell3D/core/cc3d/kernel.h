#ifndef CC3D_KERNEL_H
#define CC3D_KERNEL_H

#include <stddef.h>
#include <stdint.h>

#define CC3D_KERNEL_ABI_VERSION 1u

#ifdef __cplusplus
namespace cc3d {
namespace kernel {
class CellIterator;
struct CellSentinel;
}
}

extern "C" {
#endif

typedef void *CC3DCellIteratorHandle;

typedef struct CC3DCellViewV1 {
    uint64_t id;
    const uint8_t *type;
    const long *volume;
    float *targetVolume;
    float *lambdaVolume;
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

typedef struct CC3DKernelContext {
    uint32_t abiVersion;
    uint64_t mcs;
    CC3DCellsAPIV1 cells;
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

template<typename T>
class ReadOnlyProperty {
public:
    ReadOnlyProperty() : ptr_(nullptr) {}
    explicit ReadOnlyProperty(const T *ptr) : ptr_(ptr) {}

    operator T() const { return *ptr_; }

private:
    const T *ptr_;
};

template<typename T>
class Property {
public:
    Property() : ptr_(nullptr) {}
    explicit Property(T *ptr) : ptr_(ptr) {}

    operator T() const { return *ptr_; }

    Property &operator=(T value) {
        *ptr_ = value;
        return *this;
    }

    Property &operator+=(T value) {
        *ptr_ += value;
        return *this;
    }

    Property &operator-=(T value) {
        *ptr_ -= value;
        return *this;
    }

private:
    T *ptr_;
};

class Cell {
public:
    explicit Cell(const CC3DCellViewV1 &view)
        : id(view.id),
          type(view.type),
          volume(view.volume),
          targetVolume(view.targetVolume),
          lambdaVolume(view.lambdaVolume) {}

    uint64_t id;
    ReadOnlyProperty<uint8_t> type;
    ReadOnlyProperty<long> volume;
    Property<float> targetVolume;
    Property<float> lambdaVolume;
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
#endif

#endif
