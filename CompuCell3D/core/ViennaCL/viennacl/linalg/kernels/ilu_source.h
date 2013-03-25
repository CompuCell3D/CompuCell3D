#ifndef VIENNACL_LINALG_KERNELS_ILU_SOURCE_HPP_
#define VIENNACL_LINALG_KERNELS_ILU_SOURCE_HPP_
//Automatically generated file from auxiliary-directory, do not edit manually!
/** @file ilu_source.h
 *  @brief OpenCL kernel source file, generated automatically. */
namespace viennacl
{
 namespace linalg
 {
  namespace kernels
  {
const char * const ilu_align1_level_scheduling_substitute = 
"__kernel void level_scheduling_substitute(\n"
"          __global const unsigned int * row_index_array,\n"
"          __global const unsigned int * row_indices,\n"
"          __global const unsigned int * column_indices, \n"
"          __global const float * elements,\n"
"          __global float * vec,  \n"
"          unsigned int size) \n"
"{ \n"
"  for (unsigned int row  = get_global_id(0);\n"
"                    row  < size;\n"
"                    row += get_global_size(0))\n"
"  {\n"
"    unsigned int eq_row = row_index_array[row];\n"
"    float vec_entry = vec[eq_row];\n"
"    unsigned int row_end = row_indices[row+1];\n"
"    \n"
"    for (unsigned int j = row_indices[row]; j < row_end; ++j)\n"
"      vec_entry -= vec[column_indices[j]] * elements[j];\n"
"    \n"
"    vec[eq_row] = vec_entry;\n"
"  }\n"
"}\n"
; //ilu_align1_level_scheduling_substitute

  }  //namespace kernels
 }  //namespace linalg
}  //namespace viennacl
#endif

