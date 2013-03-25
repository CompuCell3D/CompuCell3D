#ifndef VIENNACL_LINALG_KERNELS_ELL_MATRIX_SOURCE_HPP_
#define VIENNACL_LINALG_KERNELS_ELL_MATRIX_SOURCE_HPP_
//Automatically generated file from auxiliary-directory, do not edit manually!
/** @file ell_matrix_source.h
 *  @brief OpenCL kernel source file, generated automatically. */
namespace viennacl
{
 namespace linalg
 {
  namespace kernels
  {
const char * const ell_matrix_align1_vec_mul = 
"__kernel void vec_mul(\n"
"    __global const unsigned int * coords,\n"
"    __global const float * elements,\n"
"    __global const float * vector,\n"
"    __global float * result,\n"
"    unsigned int row_num,\n"
"    unsigned int col_num,\n"
"    unsigned int internal_row_num,\n"
"    unsigned int items_per_row,\n"
"    unsigned int aligned_items_per_row\n"
"    )\n"
"{\n"
"    uint glb_id = get_global_id(0);\n"
"    uint glb_sz = get_global_size(0);\n"
"    for(uint row_id = glb_id; row_id < row_num; row_id += glb_sz)\n"
"    {\n"
"        float sum = 0;\n"
"        \n"
"        uint offset = row_id;\n"
"        for(uint item_id = 0; item_id < items_per_row; item_id++, offset += internal_row_num)\n"
"        {\n"
"            float val = elements[offset];\n"
"            if(val != 0.0f)\n"
"            {\n"
"                int col = coords[offset];    \n"
"                sum += (vector[col] * val);\n"
"            }\n"
"            \n"
"        }\n"
"        result[row_id] = sum;\n"
"    }\n"
"}\n"
; //ell_matrix_align1_vec_mul

  }  //namespace kernels
 }  //namespace linalg
}  //namespace viennacl
#endif

