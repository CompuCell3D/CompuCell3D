
#include "MBDynBitset.h"
#include <iostream>
#include <vector>
#include <bitset>
#include <unistd.h>
#include <sys/types.h>
#include <assert.h>

//#include <sys/syscall.h>
//#define gettid() syscall(SYS_gettid)

#ifdef MB_COUNT
size_t MBDynBitset::new_cnt = 0;
size_t MBDynBitset::new_default_cnt = 0;
size_t MBDynBitset::copy_cnt = 0;
size_t MBDynBitset::true_copy_cnt = 0;
size_t MBDynBitset::assign_cnt = 0;
size_t MBDynBitset::assign_and_cnt = 0;
size_t MBDynBitset::alloc_cnt = 0;
size_t MBDynBitset::del_cnt = 0;
#endif

#ifdef USE_DYNAMIC_BITSET_STD_ALLOC
typedef uint8_t refcnt_t;
const size_t refcnt_size = sizeof(refcnt_t);

uint8_t* MBDynBitset::alloc(size_t num_bytes)
{
#ifdef MB_COUNT
  alloc_cnt++;
#endif
  uint8_t* data = new uint8_t[num_bytes+refcnt_size];
  refcnt_t* p_refcnt = (refcnt_t*)&data[num_bytes];
  *p_refcnt = 1;
  //std::cerr << "alloc " << (void*)data << " " << *p_refcnt << '\n';
  return data;
}

void MBDynBitset::incr_refcount(uint64_t* ldata, size_t num_bytes)
{
  //static refcnt_t max_refcnt = 0;
  uint8_t* data = (uint8_t*)ldata;
  refcnt_t* p_refcnt = (refcnt_t*)&data[num_bytes];
  (*p_refcnt)++;
  /*
  if (*p_refcnt > max_refcnt) {
    max_refcnt = *p_refcnt;
    std::cerr << "max_refcnt = " << (int)max_refcnt << '\n';
  }
  */
  //std::cerr << "incr_ref " << (void*)data << " " << *p_refcnt << '\n';
}

void MBDynBitset::decr_refcount(uint64_t* ldata, size_t num_bytes)
{
  uint8_t* data = (uint8_t*)ldata;
  refcnt_t* p_refcnt = (refcnt_t*)&data[num_bytes];
  (*p_refcnt)--;
}

void MBDynBitset::destroy(uint64_t* ldata, size_t num_bytes)
{
  if (ldata) {
    uint8_t* data = (uint8_t*)ldata;
    refcnt_t* p_refcnt = (refcnt_t*)&data[num_bytes];
    //std::cerr << "destroy " << (void*)data << " " << *p_refcnt << '\n';
    if (--(*p_refcnt) == 0) {
      delete[] data;
#ifdef MB_COUNT
      del_cnt++;
#endif
    }
  }
}

void MBDynBitset::init_pthread()
{
}

void MBDynBitset::end_pthread()
{
}

#else

class Mutex {
  pthread_mutex_t mutex;
public:
  Mutex() {
    pthread_mutex_init(&mutex, 0);
  }
  void lock() {
    pthread_mutex_lock(&mutex);
  }
  void unlock() {
    pthread_mutex_unlock(&mutex);
  }
};

#define MAXTHREADS 256

class MBDynBitsetAllocator {
  // try to have a dynamic BUCKET_SIZE, or better:
  // in API (+ .cfg?)
  // initial_bucket_size = 10;
  // coef_bucket_size = 2;

  // initial_bucket_size = 100000;
  // coef_bucket_size = 2;

  static const unsigned int BUCKET_SIZE = 1000000;
  typedef std::bitset<BUCKET_SIZE> bitmap_alloc_t;

  struct BucketHeader {
    size_t free_cell_cnt;
  };

  struct CellHeader {
    int which; // buffer_index
    int num;
    //unsigned char busy;
    unsigned char thread_index;
  };

  std::vector<bitmap_alloc_t> bitmap_alloc_v;
  std::vector<uint8_t*> buffer_v;
  std::vector<unsigned int> cellnum_v;
  unsigned char thread_index;
  size_t total_free_cell_cnt;
  size_t total_been_freed_cell_cnt;
  size_t cell_size;

  int init(size_t num_bytes) {
    if (total_free_cell_cnt == 0) {
      cell_size = num_bytes + sizeof(CellHeader);
      bitmap_alloc_v.push_back(bitmap_alloc_t());
      uint8_t* buffer = new uint8_t[sizeof(BucketHeader) + BUCKET_SIZE * cell_size];
      ((BucketHeader*)buffer)->free_cell_cnt = BUCKET_SIZE;
      buffer_v.push_back(buffer);
      cellnum_v.push_back(0);
      total_free_cell_cnt += BUCKET_SIZE;
      //std::cout << "push_back " << buffer_v.size() << " " << total_been_freed_cell_cnt << std::endl;
    }
    return buffer_v.size()-1;
  }

  MBDynBitsetAllocator(const MBDynBitsetAllocator&);

public:
  MBDynBitsetAllocator(unsigned char thread_index) : thread_index(thread_index), total_free_cell_cnt(0), total_been_freed_cell_cnt(0), cell_size(0) {
  }

  uint8_t* alloc(size_t num_bytes) {
    int which = init(num_bytes);
    int cell_num = cellnum_v[which];
    uint8_t* buffer = buffer_v[which];
    BucketHeader* bh = (BucketHeader*)buffer;
    uint8_t* alloc_buffer = buffer+sizeof(BucketHeader);
    bh->free_cell_cnt--;
    CellHeader* cell = (CellHeader*)&alloc_buffer[cell_num * cell_size];
    cell->which = which;
    cell->num = cell_num;
    cell->thread_index = thread_index;
    //cell->busy = 1;
    //std::cout << "alloc: which=" << which << ", cell_num=" << cell->num << '\n';
    cellnum_v[which] = cell_num+1;
    if (cell_num >= BUCKET_SIZE-1) {
      total_free_cell_cnt = 0; // for now
    }
    
    return ((uint8_t*)cell)+sizeof(CellHeader);
  }

  static unsigned char getThreadIndex(uint64_t* data) {
    CellHeader* cell = (CellHeader*)((uint8_t*)data-sizeof(CellHeader));
    return cell->thread_index;
  }

  void free(uint64_t* data) {
    CellHeader* cell = (CellHeader*)((uint8_t*)data-sizeof(CellHeader));
    int which = cell->which;
    //assert(which < buffer_v.size());
    //assert(cell->num < BUCKET_SIZE);
    //std::cout << "free: which=" << which << ", cell_num=" << cell->num << '\n';
    //cell->busy = 0;
    bitmap_alloc_v[which].set(cell->num);
    ((BucketHeader*)buffer_v[which])->free_cell_cnt++;
    total_been_freed_cell_cnt++;
  }

  static MBDynBitsetAllocator* allocators[MAXTHREADS];
  static unsigned char last_thread_index;
};

static Mutex MP;

static pthread_key_t PTHREAD_ALLOCATOR_KEY;
static pthread_once_t PTHREAD_KEY_ONCE = PTHREAD_ONCE_INIT;

unsigned char MBDynBitsetAllocator::last_thread_index = 0;
MBDynBitsetAllocator* MBDynBitsetAllocator::allocators[MAXTHREADS];

static void make_allocator_key()
{
  (void)pthread_key_create(&PTHREAD_ALLOCATOR_KEY, NULL);
}

void MBDynBitset::init_pthread()
{
  MP.lock();

  (void)pthread_once(&PTHREAD_KEY_ONCE, make_allocator_key);
  MBDynBitsetAllocator* p_mb = new MBDynBitsetAllocator(MBDynBitsetAllocator::last_thread_index);
  MBDynBitsetAllocator::allocators[MBDynBitsetAllocator::last_thread_index] = p_mb;
  pthread_setspecific(PTHREAD_ALLOCATOR_KEY, new int(MBDynBitsetAllocator::last_thread_index));
  MBDynBitsetAllocator::last_thread_index++;

  MP.unlock();
}

void MBDynBitset::end_pthread()
{
  unsigned char* p_thread_index = (unsigned char*)pthread_getspecific(PTHREAD_ALLOCATOR_KEY);
  delete p_thread_index;
}

uint8_t* MBDynBitset::alloc(size_t num_bytes)
{
#ifdef MB_COUNT
  alloc_cnt++;
#endif
  unsigned char thread_index = *(unsigned char*)pthread_getspecific(PTHREAD_ALLOCATOR_KEY);
  return MBDynBitsetAllocator::allocators[thread_index]->alloc(num_bytes);
}

void MBDynBitset::destroy(uint64_t* data, size_t num_bytes)
{
#ifdef MB_COUNT
  if (data) {
    del_cnt++;
  }
#endif
  if (data) {
    MBDynBitsetAllocator *p_mb = MBDynBitsetAllocator::allocators[MBDynBitsetAllocator::getThreadIndex(data)];
    p_mb->free(data);
  }
}

void MBDynBitset::incr_refcount(uint64_t* data, size_t num_bytes)
{
}

void MBDynBitset::decr_refcount(uint64_t* data, size_t num_bytes)
{
}

#endif

void MBDynBitset::stats()
{
#ifdef MB_COUNT
  std::cout << "\nnew_default_count: " << new_default_cnt << '\n';
  std::cout << "new_count: " << new_cnt << '\n';
  std::cout << "true_copy_count: " << true_copy_cnt << '\n';
  std::cout << "copy_count: " << copy_cnt << '\n';
  std::cout << "assign_count: " << assign_cnt << '\n';
  std::cout << "assign_and_count: " << assign_and_cnt << '\n';
  std::cout << "\nalloc_count: " << alloc_cnt << '\n';
  std::cout << "delete_count: " << del_cnt << "\n\n";
#endif
}

