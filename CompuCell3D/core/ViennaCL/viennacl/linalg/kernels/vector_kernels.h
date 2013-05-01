#ifndef VIENNACL_VECTOR_KERNELS_HPP_
#define VIENNACL_VECTOR_KERNELS_HPP_
#include "viennacl/tools/tools.hpp"
#include "viennacl/ocl/kernel.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/ocl/utils.hpp"
#include "viennacl/linalg/kernels/vector_source.h"

//Automatically generated file from aux-directory, do not edit manually!
/** @file vector_kernels.h
 *  @brief OpenCL kernel file, generated automatically. */
namespace viennacl
{
 namespace linalg
 {
  namespace kernels
  {
   template<class TYPE, unsigned int alignment>
   struct vector;


    /////////////// single precision kernels //////////////// 
   template <>
   struct vector<float, 16>
   {
    static std::string program_name()
    {
      return "f_vector_16";
    }
    static void init()
    {
      viennacl::ocl::DOUBLE_PRECISION_CHECKER<float>::apply();
      static std::map<cl_context, bool> init_done;
      viennacl::ocl::context & context_ = viennacl::ocl::current_context();
      if (!init_done[context_.handle().get()])
      {
        std::string source;
        source.append(vector_align1_index_norm_inf);
        source.append(vector_align1_plane_rotation);
        source.append(vector_align1_norm);
        source.append(vector_align1_avbv_cpu_cpu);
        source.append(vector_align1_avbv_gpu_cpu);
        source.append(vector_align1_avbv_v_cpu_gpu);
        source.append(vector_align1_diag_precond);
        source.append(vector_align1_assign_cpu);
        source.append(vector_align1_element_op);
        source.append(vector_align1_sum);
        source.append(vector_align1_avbv_v_gpu_cpu);
        source.append(vector_align1_av_gpu);
        source.append(vector_align1_av_cpu);
        source.append(vector_align1_avbv_gpu_gpu);
        source.append(vector_align1_avbv_v_gpu_gpu);
        source.append(vector_align1_inner_prod);
        source.append(vector_align1_swap);
        source.append(vector_align1_avbv_v_cpu_cpu);
        source.append(vector_align1_avbv_cpu_gpu);
        std::string prog_name = program_name();
        #ifdef VIENNACL_BUILD_INFO
        std::cout << "Creating program " << prog_name << std::endl;
        #endif
        context_.add_program(source, prog_name);
        viennacl::ocl::program & prog_ = context_.get_program(prog_name);
        prog_.add_kernel("index_norm_inf");
        prog_.add_kernel("plane_rotation");
        prog_.add_kernel("norm");
        prog_.add_kernel("avbv_cpu_cpu");
        prog_.add_kernel("avbv_gpu_cpu");
        prog_.add_kernel("avbv_v_cpu_gpu");
        prog_.add_kernel("diag_precond");
        prog_.add_kernel("assign_cpu");
        prog_.add_kernel("element_op");
        prog_.add_kernel("sum");
        prog_.add_kernel("avbv_v_gpu_cpu");
        prog_.add_kernel("av_gpu");
        prog_.add_kernel("av_cpu");
        prog_.add_kernel("avbv_gpu_gpu");
        prog_.add_kernel("avbv_v_gpu_gpu");
        prog_.add_kernel("inner_prod");
        prog_.add_kernel("swap");
        prog_.add_kernel("avbv_v_cpu_cpu");
        prog_.add_kernel("avbv_cpu_gpu");
        init_done[context_.handle().get()] = true;
       } //if
     } //init
    }; // struct

   template <>
   struct vector<float, 1>
   {
    static std::string program_name()
    {
      return "f_vector_1";
    }
    static void init()
    {
      viennacl::ocl::DOUBLE_PRECISION_CHECKER<float>::apply();
      static std::map<cl_context, bool> init_done;
      viennacl::ocl::context & context_ = viennacl::ocl::current_context();
      if (!init_done[context_.handle().get()])
      {
        std::string source;
        source.append(vector_align1_index_norm_inf);
        source.append(vector_align1_plane_rotation);
        source.append(vector_align1_norm);
        source.append(vector_align1_avbv_cpu_cpu);
        source.append(vector_align1_avbv_gpu_cpu);
        source.append(vector_align1_avbv_v_cpu_gpu);
        source.append(vector_align1_diag_precond);
        source.append(vector_align1_assign_cpu);
        source.append(vector_align1_element_op);
        source.append(vector_align1_sum);
        source.append(vector_align1_avbv_v_gpu_cpu);
        source.append(vector_align1_av_gpu);
        source.append(vector_align1_av_cpu);
        source.append(vector_align1_avbv_gpu_gpu);
        source.append(vector_align1_avbv_v_gpu_gpu);
        source.append(vector_align1_inner_prod);
        source.append(vector_align1_swap);
        source.append(vector_align1_avbv_v_cpu_cpu);
        source.append(vector_align1_avbv_cpu_gpu);
        std::string prog_name = program_name();
        #ifdef VIENNACL_BUILD_INFO
        std::cout << "Creating program " << prog_name << std::endl;
        #endif
        context_.add_program(source, prog_name);
        viennacl::ocl::program & prog_ = context_.get_program(prog_name);
        prog_.add_kernel("index_norm_inf");
        prog_.add_kernel("plane_rotation");
        prog_.add_kernel("norm");
        prog_.add_kernel("avbv_cpu_cpu");
        prog_.add_kernel("avbv_gpu_cpu");
        prog_.add_kernel("avbv_v_cpu_gpu");
        prog_.add_kernel("diag_precond");
        prog_.add_kernel("assign_cpu");
        prog_.add_kernel("element_op");
        prog_.add_kernel("sum");
        prog_.add_kernel("avbv_v_gpu_cpu");
        prog_.add_kernel("av_gpu");
        prog_.add_kernel("av_cpu");
        prog_.add_kernel("avbv_gpu_gpu");
        prog_.add_kernel("avbv_v_gpu_gpu");
        prog_.add_kernel("inner_prod");
        prog_.add_kernel("swap");
        prog_.add_kernel("avbv_v_cpu_cpu");
        prog_.add_kernel("avbv_cpu_gpu");
        init_done[context_.handle().get()] = true;
       } //if
     } //init
    }; // struct

   template <>
   struct vector<float, 4>
   {
    static std::string program_name()
    {
      return "f_vector_4";
    }
    static void init()
    {
      viennacl::ocl::DOUBLE_PRECISION_CHECKER<float>::apply();
      static std::map<cl_context, bool> init_done;
      viennacl::ocl::context & context_ = viennacl::ocl::current_context();
      if (!init_done[context_.handle().get()])
      {
        std::string source;
        source.append(vector_align1_index_norm_inf);
        source.append(vector_align1_plane_rotation);
        source.append(vector_align1_norm);
        source.append(vector_align1_avbv_cpu_cpu);
        source.append(vector_align1_avbv_gpu_cpu);
        source.append(vector_align1_avbv_v_cpu_gpu);
        source.append(vector_align1_diag_precond);
        source.append(vector_align1_assign_cpu);
        source.append(vector_align1_element_op);
        source.append(vector_align1_sum);
        source.append(vector_align1_avbv_v_gpu_cpu);
        source.append(vector_align1_av_gpu);
        source.append(vector_align1_av_cpu);
        source.append(vector_align1_avbv_gpu_gpu);
        source.append(vector_align1_avbv_v_gpu_gpu);
        source.append(vector_align1_inner_prod);
        source.append(vector_align1_swap);
        source.append(vector_align1_avbv_v_cpu_cpu);
        source.append(vector_align1_avbv_cpu_gpu);
        std::string prog_name = program_name();
        #ifdef VIENNACL_BUILD_INFO
        std::cout << "Creating program " << prog_name << std::endl;
        #endif
        context_.add_program(source, prog_name);
        viennacl::ocl::program & prog_ = context_.get_program(prog_name);
        prog_.add_kernel("index_norm_inf");
        prog_.add_kernel("plane_rotation");
        prog_.add_kernel("norm");
        prog_.add_kernel("avbv_cpu_cpu");
        prog_.add_kernel("avbv_gpu_cpu");
        prog_.add_kernel("avbv_v_cpu_gpu");
        prog_.add_kernel("diag_precond");
        prog_.add_kernel("assign_cpu");
        prog_.add_kernel("element_op");
        prog_.add_kernel("sum");
        prog_.add_kernel("avbv_v_gpu_cpu");
        prog_.add_kernel("av_gpu");
        prog_.add_kernel("av_cpu");
        prog_.add_kernel("avbv_gpu_gpu");
        prog_.add_kernel("avbv_v_gpu_gpu");
        prog_.add_kernel("inner_prod");
        prog_.add_kernel("swap");
        prog_.add_kernel("avbv_v_cpu_cpu");
        prog_.add_kernel("avbv_cpu_gpu");
        init_done[context_.handle().get()] = true;
       } //if
     } //init
    }; // struct



    /////////////// double precision kernels //////////////// 
   template <>
   struct vector<double, 16>
   {
    static std::string program_name()
    {
      return "d_vector_16";
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
        source.append(viennacl::tools::make_double_kernel(vector_align1_index_norm_inf, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(vector_align1_plane_rotation, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(vector_align1_norm, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(vector_align1_avbv_cpu_cpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(vector_align1_avbv_gpu_cpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(vector_align1_avbv_v_cpu_gpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(vector_align1_diag_precond, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(vector_align1_assign_cpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(vector_align1_element_op, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(vector_align1_sum, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(vector_align1_avbv_v_gpu_cpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(vector_align1_av_gpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(vector_align1_av_cpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(vector_align1_avbv_gpu_gpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(vector_align1_avbv_v_gpu_gpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(vector_align1_inner_prod, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(vector_align1_swap, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(vector_align1_avbv_v_cpu_cpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(vector_align1_avbv_cpu_gpu, fp64_ext));
        std::string prog_name = program_name();
        #ifdef VIENNACL_BUILD_INFO
        std::cout << "Creating program " << prog_name << std::endl;
        #endif
        context_.add_program(source, prog_name);
        viennacl::ocl::program & prog_ = context_.get_program(prog_name);
        prog_.add_kernel("index_norm_inf");
        prog_.add_kernel("plane_rotation");
        prog_.add_kernel("norm");
        prog_.add_kernel("avbv_cpu_cpu");
        prog_.add_kernel("avbv_gpu_cpu");
        prog_.add_kernel("avbv_v_cpu_gpu");
        prog_.add_kernel("diag_precond");
        prog_.add_kernel("assign_cpu");
        prog_.add_kernel("element_op");
        prog_.add_kernel("sum");
        prog_.add_kernel("avbv_v_gpu_cpu");
        prog_.add_kernel("av_gpu");
        prog_.add_kernel("av_cpu");
        prog_.add_kernel("avbv_gpu_gpu");
        prog_.add_kernel("avbv_v_gpu_gpu");
        prog_.add_kernel("inner_prod");
        prog_.add_kernel("swap");
        prog_.add_kernel("avbv_v_cpu_cpu");
        prog_.add_kernel("avbv_cpu_gpu");
        init_done[context_.handle().get()] = true;
       } //if
     } //init
    }; // struct

   template <>
   struct vector<double, 1>
   {
    static std::string program_name()
    {
      return "d_vector_1";
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
        source.append(viennacl::tools::make_double_kernel(vector_align1_index_norm_inf, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(vector_align1_plane_rotation, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(vector_align1_norm, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(vector_align1_avbv_cpu_cpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(vector_align1_avbv_gpu_cpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(vector_align1_avbv_v_cpu_gpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(vector_align1_diag_precond, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(vector_align1_assign_cpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(vector_align1_element_op, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(vector_align1_sum, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(vector_align1_avbv_v_gpu_cpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(vector_align1_av_gpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(vector_align1_av_cpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(vector_align1_avbv_gpu_gpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(vector_align1_avbv_v_gpu_gpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(vector_align1_inner_prod, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(vector_align1_swap, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(vector_align1_avbv_v_cpu_cpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(vector_align1_avbv_cpu_gpu, fp64_ext));
        std::string prog_name = program_name();
        #ifdef VIENNACL_BUILD_INFO
        std::cout << "Creating program " << prog_name << std::endl;
        #endif
        context_.add_program(source, prog_name);
        viennacl::ocl::program & prog_ = context_.get_program(prog_name);
        prog_.add_kernel("index_norm_inf");
        prog_.add_kernel("plane_rotation");
        prog_.add_kernel("norm");
        prog_.add_kernel("avbv_cpu_cpu");
        prog_.add_kernel("avbv_gpu_cpu");
        prog_.add_kernel("avbv_v_cpu_gpu");
        prog_.add_kernel("diag_precond");
        prog_.add_kernel("assign_cpu");
        prog_.add_kernel("element_op");
        prog_.add_kernel("sum");
        prog_.add_kernel("avbv_v_gpu_cpu");
        prog_.add_kernel("av_gpu");
        prog_.add_kernel("av_cpu");
        prog_.add_kernel("avbv_gpu_gpu");
        prog_.add_kernel("avbv_v_gpu_gpu");
        prog_.add_kernel("inner_prod");
        prog_.add_kernel("swap");
        prog_.add_kernel("avbv_v_cpu_cpu");
        prog_.add_kernel("avbv_cpu_gpu");
        init_done[context_.handle().get()] = true;
       } //if
     } //init
    }; // struct

   template <>
   struct vector<double, 4>
   {
    static std::string program_name()
    {
      return "d_vector_4";
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
        source.append(viennacl::tools::make_double_kernel(vector_align1_index_norm_inf, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(vector_align1_plane_rotation, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(vector_align1_norm, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(vector_align1_avbv_cpu_cpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(vector_align1_avbv_gpu_cpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(vector_align1_avbv_v_cpu_gpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(vector_align1_diag_precond, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(vector_align1_assign_cpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(vector_align1_element_op, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(vector_align1_sum, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(vector_align1_avbv_v_gpu_cpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(vector_align1_av_gpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(vector_align1_av_cpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(vector_align1_avbv_gpu_gpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(vector_align1_avbv_v_gpu_gpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(vector_align1_inner_prod, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(vector_align1_swap, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(vector_align1_avbv_v_cpu_cpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(vector_align1_avbv_cpu_gpu, fp64_ext));
        std::string prog_name = program_name();
        #ifdef VIENNACL_BUILD_INFO
        std::cout << "Creating program " << prog_name << std::endl;
        #endif
        context_.add_program(source, prog_name);
        viennacl::ocl::program & prog_ = context_.get_program(prog_name);
        prog_.add_kernel("index_norm_inf");
        prog_.add_kernel("plane_rotation");
        prog_.add_kernel("norm");
        prog_.add_kernel("avbv_cpu_cpu");
        prog_.add_kernel("avbv_gpu_cpu");
        prog_.add_kernel("avbv_v_cpu_gpu");
        prog_.add_kernel("diag_precond");
        prog_.add_kernel("assign_cpu");
        prog_.add_kernel("element_op");
        prog_.add_kernel("sum");
        prog_.add_kernel("avbv_v_gpu_cpu");
        prog_.add_kernel("av_gpu");
        prog_.add_kernel("av_cpu");
        prog_.add_kernel("avbv_gpu_gpu");
        prog_.add_kernel("avbv_v_gpu_gpu");
        prog_.add_kernel("inner_prod");
        prog_.add_kernel("swap");
        prog_.add_kernel("avbv_v_cpu_cpu");
        prog_.add_kernel("avbv_cpu_gpu");
        init_done[context_.handle().get()] = true;
       } //if
     } //init
    }; // struct


  }  //namespace kernels
 }  //namespace linalg
}  //namespace viennacl
#endif

