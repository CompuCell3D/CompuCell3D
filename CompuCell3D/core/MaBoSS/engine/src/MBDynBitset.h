#include <stdlib.h>
#include <string.h>
#include <sstream>
#include <iostream>
#include "maboss-config.h"

//#define USE_MB_SHIFT
//#define MB_COUNT

class MBDynBitset {

  uint64_t* data;
  size_t num_bits;
  size_t num_bytes;
  size_t num_64;

  static size_t getNumBytes(size_t nbits) {
    if (nbits < 8) {
      return 1;
    }
    return 1 + (nbits - 1) / 8;
  }

  static size_t getNum64(size_t nbits) {
    return 1 + (nbits - 1) / 64;
  }

#ifdef MB_COUNT
  static size_t new_cnt;
  static size_t new_default_cnt;
  static size_t true_copy_cnt;
  static size_t copy_cnt;
  static size_t assign_cnt;
  static size_t assign_and_cnt;

  static size_t alloc_cnt;
  static size_t del_cnt;
#endif

  static uint8_t* alloc(size_t num_bytes);
  static void destroy(uint64_t* data, size_t num_bytes);
  static void decr_refcount(uint64_t* data, size_t num_bytes);
  static void incr_refcount(uint64_t* data, size_t num_bytes);

public:
  MBDynBitset() : data(0), num_bits(0), num_bytes(0), num_64(0) {
#ifdef MB_COUNT
    new_default_cnt++;
#endif
 }

  MBDynBitset(size_t nbits) {
    num_bytes = 0;
    num_64 = 0;
    resize(nbits);
#ifdef MB_COUNT
    new_cnt++;
#endif
  }

  //private:
  MBDynBitset(const MBDynBitset& bitset) {
#ifdef MB_COUNT
    copy_cnt++;
#endif
    num_bits = bitset.num_bits;
    num_bytes = bitset.num_bytes;
    num_64 = bitset.num_64;
    data = bitset.data;
    incr_refcount(data, num_bytes);
  }
  //public:

  MBDynBitset(const MBDynBitset& bitset, int real_copy) {
#ifdef MB_COUNT
    true_copy_cnt++;
#endif
    data = 0;
    *this = bitset;
  }

  MBDynBitset& operator=(const MBDynBitset& bitset) {
    if (this == &bitset) {
      return *this;
    }
#ifdef MB_COUNT
    if (data != 0) {
      assign_cnt++;
    }
#endif
    destroy(data, num_bytes);
    num_bits = bitset.num_bits;
    num_bytes = bitset.num_bytes;
    num_64 = bitset.num_64;
    if (num_bits) {
      data = (uint64_t*)alloc(num_bytes);
      /*if (num_64 == 1) {
	data[0] = bitset.data[0];
	} else*/ {
	memcpy(data, bitset.data, num_bytes);
      }
    } else {
      data = 0;
    }
    return *this;
  }


#if 0
  MBDynBitset& operator&=(const MBDynBitset &bitset) {
#ifdef MB_COUNT
    assign_and_cnt++;
#endif
    *this = (*this) & bitset;
    return *this;
  }
#endif

  void resize(size_t nbits) {
    num_bits = nbits;
    if (num_bytes == 0) {
      num_64 = getNum64(num_bits);
      num_bytes = num_64*sizeof(uint64_t);
      data = (uint64_t*)alloc(num_bytes);
      /*if (num_64 == 1) {
	data[0] = 0;
	} else*/ {
	memset(data, 0, num_bytes);
      }
    } else {
      std::cerr << "BAD 1\n";
      abort();
    }
  }

  bool test(size_t pos) const {
#ifdef USE_MB_SHIFT
    size_t byte_loc = pos >> 3;
    size_t offset = pos - (byte_loc * 8);
#else
    size_t byte_loc = pos / 8;
    size_t offset = pos % 8;
#endif
    uint8_t* b_data = (uint8_t*)data;
    return ((b_data[byte_loc] >> offset) & 0x1) == 1;
  }

  void set() {
    /*if (num_64 == 1) {
      data[0] = ~0ULL;
      } else*/ {
      for (size_t nn = 0; nn < num_64; ++nn) {
	data[nn] = ~0ULL;
      }
    }
  }

  void set(size_t pos, bool state) {
#ifdef USE_MB_SHIFT
    size_t byte_loc = pos >> 3;
    size_t offset = pos - (byte_loc * 8);
#else
    size_t byte_loc = pos / 8;
    size_t offset = pos % 8;
#endif
    uint8_t bitfield = uint8_t(1 << offset);

    uint8_t* b_data = (uint8_t*)data;

    if (state) {
      b_data[byte_loc] |= bitfield;
    } else {
      b_data[byte_loc] &= ~bitfield;
    }
  }

  void flip(size_t pos) {
#ifdef USE_MB_SHIFT
    size_t byte_loc = pos >> 3;
    size_t offset = pos - (byte_loc * 8);
#else
    size_t byte_loc = pos / 8;
    size_t offset = pos % 8;
#endif
    uint8_t bitfield = uint8_t(1 << offset);
    uint8_t* b_data = (uint8_t*)data;
    uint8_t value = (b_data[byte_loc] & bitfield);

    if (value) {
      b_data[byte_loc] &= ~bitfield;
    } else {
      b_data[byte_loc] |= bitfield;
    }
  }

  std::string toString() const {
    std::stringstream ostr;

    for (size_t pos = 0; pos < num_bits; pos++) {
      ostr << (test(num_bits - pos - 1) ? "1" : "0");
    }

    return ostr.str();
  }

  int operator==(const MBDynBitset& bitset) const {
    /*if (num_64 == 1) {
      return data[0] == bitset.data[0];
      } else*/ {
      for (size_t nn = 0; nn < num_64; nn++) {
	if (data[nn] != bitset.data[nn]) {
	  return 0;
	}
      }
    }
    return 1;
  }

  MBDynBitset operator&(const MBDynBitset& bitset) const {
    if (num_bits != bitset.num_bits) {
      std::cerr << "BAD 2\n";
      abort();
    }
    MBDynBitset ret_bitset(num_bits);
    /*if (num_64 == 1) {
      uint64_t* ret_data = ret_bitset.getData();
      ret_data[0] = (data[0] & bitset.data[0]);
      } else*/ {
      uint64_t* ret_data = ret_bitset.getData();
      for (size_t nn = 0; nn < num_64; nn++) {
	ret_data[nn] = (data[nn] & bitset.data[nn]);
      }
    }
    return ret_bitset;
  }

  bool operator<(const MBDynBitset& bitset) const {
    if (num_bits != bitset.num_bits) {
      std::cerr << "BAD 3\n";
      abort();
    }

    /*if (num_64 == 1) {
      return data[0] - bitset.data[0] < 0;
      } else*/ {
      for (size_t nn = 0; nn < num_64; nn++) {
	long long delta = data[nn] - bitset.data[nn];
	if (delta < 0) {
	  return true;
	}
	if (delta > 0) {
	  return false;
	}
      }
    }

    return false;
  }

  MBDynBitset operator~() {
    MBDynBitset ret_bitset(num_bits);
    for (size_t pos = 0; pos < num_bits; pos++) {
      ret_bitset.set(pos, !test(pos));
    }
    return ret_bitset;
  }

  bool none() const {
    /*if (num_64 == 1) {
      return !data[0];
      } else*/ {
      for (size_t nn = 0; nn < num_64; nn++) {
	if (data[nn]) {
	  return false;
	}
      }
    }
    return true;
  }

  uint64_t* getData() {return data;}

#ifdef MB_COUNT
  static size_t getAllocCount() {
    return alloc_cnt;
  }

  static size_t getDeleteCount() {
    return del_cnt;
  }

  static size_t getCopyCount() {
    return copy_cnt;
  }

  static size_t getTrueCopyCount() {
    return true_copy_cnt;
  }

  static size_t getAssignCount() {
    return assign_cnt;
  }

  static size_t getAssignAndCount() {
    return assign_and_cnt;
  }

  static size_t getNewCount() {
    return new_cnt;
  }

  static size_t getNewDefaultCount() {
    return new_default_cnt;
  }
#endif

  static void init_pthread();
  static void end_pthread();

  static void stats();

  uint64_t to_ulong() const {
    uint64_t ret = 0;
    size_t max = num_bytes;
    if (max > sizeof(uint64_t)) {
      max = sizeof(uint64_t);
    }
    ret = data[0];

    return ret;
  }

  ~MBDynBitset() {
    destroy(data, num_bytes);
  }
};

#ifdef HAS_UNORDERED_MAP
namespace std {
  template <> struct HASH_STRUCT<MBDynBitset > : public std::unary_function<MBDynBitset, size_t>
  {
    size_t operator()(const MBDynBitset& val) const {
      return val.to_ulong();
    }
  };

  template <> struct equal_to<MBDynBitset > : public binary_function<MBDynBitset, MBDynBitset, bool>
  {
    size_t operator()(const MBDynBitset& val1, const MBDynBitset& val2) const {
      return val1 == val2;
    }
  };

  // Added less operator, necessary for maps, sets. Code from https://stackoverflow.com/a/21245301/11713763
  template <> struct less<MBDynBitset> : public binary_function<MBDynBitset, MBDynBitset, bool>
  {
    size_t operator()(const MBDynBitset& val1, const MBDynBitset& val2) const {
      return val1 < val2;
    }
  };
}
#endif
