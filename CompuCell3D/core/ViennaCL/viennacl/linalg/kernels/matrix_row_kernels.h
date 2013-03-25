#ifndef VIENNACL_MATRIX_ROW_KERNELS_HPP_
#define VIENNACL_MATRIX_ROW_KERNELS_HPP_
#include "viennacl/tools/tools.hpp"
#include "viennacl/ocl/kernel.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/ocl/utils.hpp"
#include "viennacl/linalg/kernels/matrix_row_source.h"

//Automatically generated file from aux-directory, do not edit manually!
/** @file matrix_row_kernels.h
 *  @brief OpenCL kernel file, generated automatically. */
namespace viennacl
{
 namespace linalg
 {
  namespace kernels
  {
   template<class TYPE, unsigned int alignment>
   struct matrix_row;


    /////////////// single precision kernels //////////////// 
   template <>
   struct matrix_row<float, 16>
   {
    static std::string program_name()
    {
      return "f_matrix_row_16";
    }
    static void init()
    {
      viennacl::ocl::DOUBLE_PRECISION_CHECKER<float>::apply();
      static std::map<cl_context, bool> init_done;
      viennacl::ocl::context & context_ = viennacl::ocl::current_context();
      if (!init_done[context_.handle().get()])
      {
        std::string source;
        source.append(matrix_row_align1_ambm_m_cpu_gpu);
        source.append(matrix_row_align1_ambm_m_cpu_cpu);
        source.append(matrix_row_align1_ambm_m_gpu_cpu);
        source.append(matrix_row_align1_lu_factorize);
        source.append(matrix_row_align1_fft_reorder);
        source.append(matrix_row_align1_fft_radix2);
        source.append(matrix_row_align1_ambm_gpu_gpu);
        source.append(matrix_row_align1_am_gpu);
        source.append(matrix_row_align1_fft_direct);
        source.append(matrix_row_align1_trans_vec_mul);
        source.append(matrix_row_align1_am_cpu);
        source.append(matrix_row_align1_scaled_rank1_update_gpu);
        source.append(matrix_row_align1_assign_cpu);
        source.append(matrix_row_align1_fft_radix2_local);
        source.append(matrix_row_align1_triangular_substitute_inplace);
        source.append(matrix_row_align1_ambm_m_gpu_gpu);
        source.append(matrix_row_align1_vec_mul);
        source.append(matrix_row_align1_ambm_gpu_cpu);
        source.append(matrix_row_align1_ambm_cpu_cpu);
        source.append(matrix_row_align1_ambm_cpu_gpu);
        source.append(matrix_row_align1_scaled_rank1_update_cpu);
        source.append(matrix_row_align1_diagonal_assign_cpu);
        std::string prog_name = program_name();
        #ifdef VIENNACL_BUILD_INFO
        std::cout << "Creating program " << prog_name << std::endl;
        #endif
        context_.add_program(source, prog_name);
        viennacl::ocl::program & prog_ = context_.get_program(prog_name);
        prog_.add_kernel("ambm_m_cpu_gpu");
        prog_.add_kernel("ambm_m_cpu_cpu");
        prog_.add_kernel("ambm_m_gpu_cpu");
        prog_.add_kernel("lu_factorize");
        prog_.add_kernel("fft_reorder");
        prog_.add_kernel("fft_radix2");
        prog_.add_kernel("ambm_gpu_gpu");
        prog_.add_kernel("am_gpu");
        prog_.add_kernel("fft_direct");
        prog_.add_kernel("trans_vec_mul");
        prog_.add_kernel("am_cpu");
        prog_.add_kernel("scaled_rank1_update_gpu");
        prog_.add_kernel("assign_cpu");
        prog_.add_kernel("fft_radix2_local");
        prog_.add_kernel("triangular_substitute_inplace");
        prog_.add_kernel("ambm_m_gpu_gpu");
        prog_.add_kernel("vec_mul");
        prog_.add_kernel("ambm_gpu_cpu");
        prog_.add_kernel("ambm_cpu_cpu");
        prog_.add_kernel("ambm_cpu_gpu");
        prog_.add_kernel("scaled_rank1_update_cpu");
        prog_.add_kernel("diagonal_assign_cpu");
        init_done[context_.handle().get()] = true;
       } //if
     } //init
    }; // struct

   template <>
   struct matrix_row<float, 1>
   {
    static std::string program_name()
    {
      return "f_matrix_row_1";
    }
    static void init()
    {
      viennacl::ocl::DOUBLE_PRECISION_CHECKER<float>::apply();
      static std::map<cl_context, bool> init_done;
      viennacl::ocl::context & context_ = viennacl::ocl::current_context();
      if (!init_done[context_.handle().get()])
      {
        std::string source;
        source.append(matrix_row_align1_ambm_m_cpu_gpu);
        source.append(matrix_row_align1_ambm_m_cpu_cpu);
        source.append(matrix_row_align1_ambm_m_gpu_cpu);
        source.append(matrix_row_align1_lu_factorize);
        source.append(matrix_row_align1_fft_reorder);
        source.append(matrix_row_align1_fft_radix2);
        source.append(matrix_row_align1_ambm_gpu_gpu);
        source.append(matrix_row_align1_am_gpu);
        source.append(matrix_row_align1_fft_direct);
        source.append(matrix_row_align1_trans_vec_mul);
        source.append(matrix_row_align1_am_cpu);
        source.append(matrix_row_align1_scaled_rank1_update_gpu);
        source.append(matrix_row_align1_assign_cpu);
        source.append(matrix_row_align1_fft_radix2_local);
        source.append(matrix_row_align1_triangular_substitute_inplace);
        source.append(matrix_row_align1_ambm_m_gpu_gpu);
        source.append(matrix_row_align1_vec_mul);
        source.append(matrix_row_align1_ambm_gpu_cpu);
        source.append(matrix_row_align1_ambm_cpu_cpu);
        source.append(matrix_row_align1_ambm_cpu_gpu);
        source.append(matrix_row_align1_scaled_rank1_update_cpu);
        source.append(matrix_row_align1_diagonal_assign_cpu);
        std::string prog_name = program_name();
        #ifdef VIENNACL_BUILD_INFO
        std::cout << "Creating program " << prog_name << std::endl;
        #endif
        context_.add_program(source, prog_name);
        viennacl::ocl::program & prog_ = context_.get_program(prog_name);
        prog_.add_kernel("ambm_m_cpu_gpu");
        prog_.add_kernel("ambm_m_cpu_cpu");
        prog_.add_kernel("ambm_m_gpu_cpu");
        prog_.add_kernel("lu_factorize");
        prog_.add_kernel("fft_reorder");
        prog_.add_kernel("fft_radix2");
        prog_.add_kernel("ambm_gpu_gpu");
        prog_.add_kernel("am_gpu");
        prog_.add_kernel("fft_direct");
        prog_.add_kernel("trans_vec_mul");
        prog_.add_kernel("am_cpu");
        prog_.add_kernel("scaled_rank1_update_gpu");
        prog_.add_kernel("assign_cpu");
        prog_.add_kernel("fft_radix2_local");
        prog_.add_kernel("triangular_substitute_inplace");
        prog_.add_kernel("ambm_m_gpu_gpu");
        prog_.add_kernel("vec_mul");
        prog_.add_kernel("ambm_gpu_cpu");
        prog_.add_kernel("ambm_cpu_cpu");
        prog_.add_kernel("ambm_cpu_gpu");
        prog_.add_kernel("scaled_rank1_update_cpu");
        prog_.add_kernel("diagonal_assign_cpu");
        init_done[context_.handle().get()] = true;
       } //if
     } //init
    }; // struct



    /////////////// double precision kernels //////////////// 
   template <>
   struct matrix_row<double, 16>
   {
    static std::string program_name()
    {
      return "d_matrix_row_16";
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
        source.append(viennacl::tools::make_double_kernel(matrix_row_align1_ambm_m_cpu_gpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_row_align1_ambm_m_cpu_cpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_row_align1_ambm_m_gpu_cpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_row_align1_lu_factorize, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_row_align1_fft_reorder, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_row_align1_fft_radix2, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_row_align1_ambm_gpu_gpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_row_align1_am_gpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_row_align1_fft_direct, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_row_align1_trans_vec_mul, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_row_align1_am_cpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_row_align1_scaled_rank1_update_gpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_row_align1_assign_cpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_row_align1_fft_radix2_local, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_row_align1_triangular_substitute_inplace, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_row_align1_ambm_m_gpu_gpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_row_align1_vec_mul, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_row_align1_ambm_gpu_cpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_row_align1_ambm_cpu_cpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_row_align1_ambm_cpu_gpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_row_align1_scaled_rank1_update_cpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_row_align1_diagonal_assign_cpu, fp64_ext));
        std::string prog_name = program_name();
        #ifdef VIENNACL_BUILD_INFO
        std::cout << "Creating program " << prog_name << std::endl;
        #endif
        context_.add_program(source, prog_name);
        viennacl::ocl::program & prog_ = context_.get_program(prog_name);
        prog_.add_kernel("ambm_m_cpu_gpu");
        prog_.add_kernel("ambm_m_cpu_cpu");
        prog_.add_kernel("ambm_m_gpu_cpu");
        prog_.add_kernel("lu_factorize");
        prog_.add_kernel("fft_reorder");
        prog_.add_kernel("fft_radix2");
        prog_.add_kernel("ambm_gpu_gpu");
        prog_.add_kernel("am_gpu");
        prog_.add_kernel("fft_direct");
        prog_.add_kernel("trans_vec_mul");
        prog_.add_kernel("am_cpu");
        prog_.add_kernel("scaled_rank1_update_gpu");
        prog_.add_kernel("assign_cpu");
        prog_.add_kernel("fft_radix2_local");
        prog_.add_kernel("triangular_substitute_inplace");
        prog_.add_kernel("ambm_m_gpu_gpu");
        prog_.add_kernel("vec_mul");
        prog_.add_kernel("ambm_gpu_cpu");
        prog_.add_kernel("ambm_cpu_cpu");
        prog_.add_kernel("ambm_cpu_gpu");
        prog_.add_kernel("scaled_rank1_update_cpu");
        prog_.add_kernel("diagonal_assign_cpu");
        init_done[context_.handle().get()] = true;
       } //if
     } //init
    }; // struct

   template <>
   struct matrix_row<double, 1>
   {
    static std::string program_name()
    {
      return "d_matrix_row_1";
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
        source.append(viennacl::tools::make_double_kernel(matrix_row_align1_ambm_m_cpu_gpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_row_align1_ambm_m_cpu_cpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_row_align1_ambm_m_gpu_cpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_row_align1_lu_factorize, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_row_align1_fft_reorder, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_row_align1_fft_radix2, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_row_align1_ambm_gpu_gpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_row_align1_am_gpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_row_align1_fft_direct, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_row_align1_trans_vec_mul, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_row_align1_am_cpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_row_align1_scaled_rank1_update_gpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_row_align1_assign_cpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_row_align1_fft_radix2_local, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_row_align1_triangular_substitute_inplace, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_row_align1_ambm_m_gpu_gpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_row_align1_vec_mul, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_row_align1_ambm_gpu_cpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_row_align1_ambm_cpu_cpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_row_align1_ambm_cpu_gpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_row_align1_scaled_rank1_update_cpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_row_align1_diagonal_assign_cpu, fp64_ext));
        std::string prog_name = program_name();
        #ifdef VIENNACL_BUILD_INFO
        std::cout << "Creating program " << prog_name << std::endl;
        #endif
        context_.add_program(source, prog_name);
        viennacl::ocl::program & prog_ = context_.get_program(prog_name);
        prog_.add_kernel("ambm_m_cpu_gpu");
        prog_.add_kernel("ambm_m_cpu_cpu");
        prog_.add_kernel("ambm_m_gpu_cpu");
        prog_.add_kernel("lu_factorize");
        prog_.add_kernel("fft_reorder");
        prog_.add_kernel("fft_radix2");
        prog_.add_kernel("ambm_gpu_gpu");
        prog_.add_kernel("am_gpu");
        prog_.add_kernel("fft_direct");
        prog_.add_kernel("trans_vec_mul");
        prog_.add_kernel("am_cpu");
        prog_.add_kernel("scaled_rank1_update_gpu");
        prog_.add_kernel("assign_cpu");
        prog_.add_kernel("fft_radix2_local");
        prog_.add_kernel("triangular_substitute_inplace");
        prog_.add_kernel("ambm_m_gpu_gpu");
        prog_.add_kernel("vec_mul");
        prog_.add_kernel("ambm_gpu_cpu");
        prog_.add_kernel("ambm_cpu_cpu");
        prog_.add_kernel("ambm_cpu_gpu");
        prog_.add_kernel("scaled_rank1_update_cpu");
        prog_.add_kernel("diagonal_assign_cpu");
        init_done[context_.handle().get()] = true;
       } //if
     } //init
    }; // struct


  }  //namespace kernels
 }  //namespace linalg
}  //namespace viennacl
#endif

