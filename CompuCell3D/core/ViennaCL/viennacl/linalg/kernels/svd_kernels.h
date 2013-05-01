#ifndef VIENNACL_SVD_KERNELS_HPP_
#define VIENNACL_SVD_KERNELS_HPP_
#include "viennacl/tools/tools.hpp"
#include "viennacl/ocl/kernel.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/ocl/utils.hpp"
#include "viennacl/linalg/kernels/svd_source.h"

//Automatically generated file from aux-directory, do not edit manually!
/** @file svd_kernels.h
 *  @brief OpenCL kernel file, generated automatically. */
namespace viennacl
{
 namespace linalg
 {
  namespace kernels
  {
   template<class TYPE, unsigned int alignment>
   struct svd;


    /////////////// single precision kernels //////////////// 
   template <>
   struct svd<float, 1>
   {
    static std::string program_name()
    {
      return "f_svd_1";
    }
    static void init()
    {
      viennacl::ocl::DOUBLE_PRECISION_CHECKER<float>::apply();
      static std::map<cl_context, bool> init_done;
      viennacl::ocl::context & context_ = viennacl::ocl::current_context();
      if (!init_done[context_.handle().get()])
      {
        std::string source;
        source.append(svd_align1_transpose_inplace);
        source.append(svd_align1_house_update_A_left);
        source.append(svd_align1_copy_col);
        source.append(svd_align1_final_iter_update);
        source.append(svd_align1_house_update_A_right);
        source.append(svd_align1_bidiag_pack);
        source.append(svd_align1_col_reduce_lcl_array);
        source.append(svd_align1_inverse_signs);
        source.append(svd_align1_update_qr_column);
        source.append(svd_align1_house_update_QL);
        source.append(svd_align1_copy_row);
        source.append(svd_align1_givens_prev);
        source.append(svd_align1_givens_next);
        source.append(svd_align1_house_update_QR);
        std::string prog_name = program_name();
        #ifdef VIENNACL_BUILD_INFO
        std::cout << "Creating program " << prog_name << std::endl;
        #endif
        context_.add_program(source, prog_name);
        viennacl::ocl::program & prog_ = context_.get_program(prog_name);
        prog_.add_kernel("transpose_inplace");
        prog_.add_kernel("house_update_A_left");
        prog_.add_kernel("copy_col");
        prog_.add_kernel("final_iter_update");
        prog_.add_kernel("house_update_A_right");
        prog_.add_kernel("bidiag_pack");
        prog_.add_kernel("col_reduce_lcl_array");
        prog_.add_kernel("inverse_signs");
        prog_.add_kernel("update_qr_column");
        prog_.add_kernel("house_update_QL");
        prog_.add_kernel("copy_row");
        prog_.add_kernel("givens_prev");
        prog_.add_kernel("givens_next");
        prog_.add_kernel("house_update_QR");
        init_done[context_.handle().get()] = true;
       } //if
     } //init
    }; // struct



    /////////////// double precision kernels //////////////// 
   template <>
   struct svd<double, 1>
   {
    static std::string program_name()
    {
      return "d_svd_1";
    }
    static void init()
    {
      viennacl::ocl::DOUBLE_PRECISION_CHECKER<double>::apply();
      static std::map<cl_context, bool> init_done;
      viennacl::ocl::context & context_ = viennacl::ocl::current_context();
      if (!init_done[context_.handle().get()])
      {
        std::string source;
        std::string fp64_ext = viennacl::ocl::current_device().double_support_extension();
        source.append(viennacl::tools::make_double_kernel(svd_align1_transpose_inplace, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(svd_align1_house_update_A_left, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(svd_align1_copy_col, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(svd_align1_final_iter_update, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(svd_align1_house_update_A_right, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(svd_align1_bidiag_pack, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(svd_align1_col_reduce_lcl_array, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(svd_align1_inverse_signs, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(svd_align1_update_qr_column, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(svd_align1_house_update_QL, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(svd_align1_copy_row, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(svd_align1_givens_prev, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(svd_align1_givens_next, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(svd_align1_house_update_QR, fp64_ext));
        std::string prog_name = program_name();
        #ifdef VIENNACL_BUILD_INFO
        std::cout << "Creating program " << prog_name << std::endl;
        #endif
        context_.add_program(source, prog_name);
        viennacl::ocl::program & prog_ = context_.get_program(prog_name);
        prog_.add_kernel("transpose_inplace");
        prog_.add_kernel("house_update_A_left");
        prog_.add_kernel("copy_col");
        prog_.add_kernel("final_iter_update");
        prog_.add_kernel("house_update_A_right");
        prog_.add_kernel("bidiag_pack");
        prog_.add_kernel("col_reduce_lcl_array");
        prog_.add_kernel("inverse_signs");
        prog_.add_kernel("update_qr_column");
        prog_.add_kernel("house_update_QL");
        prog_.add_kernel("copy_row");
        prog_.add_kernel("givens_prev");
        prog_.add_kernel("givens_next");
        prog_.add_kernel("house_update_QR");
        init_done[context_.handle().get()] = true;
       } //if
     } //init
    }; // struct


  }  //namespace kernels
 }  //namespace linalg
}  //namespace viennacl
#endif

