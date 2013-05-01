#ifndef VIENNACL_SCALAR_KERNELS_HPP_
#define VIENNACL_SCALAR_KERNELS_HPP_
#include "viennacl/tools/tools.hpp"
#include "viennacl/ocl/kernel.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/ocl/utils.hpp"
#include "viennacl/linalg/kernels/scalar_source.h"

//Automatically generated file from aux-directory, do not edit manually!
/** @file scalar_kernels.h
 *  @brief OpenCL kernel file, generated automatically. */
namespace viennacl
{
 namespace linalg
 {
  namespace kernels
  {
   template<class TYPE, unsigned int alignment>
   struct scalar;


    /////////////// single precision kernels //////////////// 
   template <>
   struct scalar<float, 1>
   {
    static std::string program_name()
    {
      return "f_scalar_1";
    }
    static void init()
    {
      viennacl::ocl::DOUBLE_PRECISION_CHECKER<float>::apply();
      static std::map<cl_context, bool> init_done;
      viennacl::ocl::context & context_ = viennacl::ocl::current_context();
      if (!init_done[context_.handle().get()])
      {
        std::string source;
        source.append(scalar_align1_asbs_gpu_cpu);
        source.append(scalar_align1_asbs_s_cpu_gpu);
        source.append(scalar_align1_asbs_gpu_gpu);
        source.append(scalar_align1_asbs_s_cpu_cpu);
        source.append(scalar_align1_asbs_cpu_cpu);
        source.append(scalar_align1_asbs_s_gpu_cpu);
        source.append(scalar_align1_as_gpu);
        source.append(scalar_align1_as_cpu);
        source.append(scalar_align1_asbs_s_gpu_gpu);
        source.append(scalar_align1_asbs_cpu_gpu);
        std::string prog_name = program_name();
        #ifdef VIENNACL_BUILD_INFO
        std::cout << "Creating program " << prog_name << std::endl;
        #endif
        context_.add_program(source, prog_name);
        viennacl::ocl::program & prog_ = context_.get_program(prog_name);
        prog_.add_kernel("asbs_gpu_cpu");
        prog_.add_kernel("asbs_s_cpu_gpu");
        prog_.add_kernel("asbs_gpu_gpu");
        prog_.add_kernel("asbs_s_cpu_cpu");
        prog_.add_kernel("asbs_cpu_cpu");
        prog_.add_kernel("asbs_s_gpu_cpu");
        prog_.add_kernel("as_gpu");
        prog_.add_kernel("as_cpu");
        prog_.add_kernel("asbs_s_gpu_gpu");
        prog_.add_kernel("asbs_cpu_gpu");
        init_done[context_.handle().get()] = true;
       } //if
     } //init
    }; // struct



    /////////////// double precision kernels //////////////// 
   template <>
   struct scalar<double, 1>
   {
    static std::string program_name()
    {
      return "d_scalar_1";
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
        source.append(viennacl::tools::make_double_kernel(scalar_align1_asbs_gpu_cpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(scalar_align1_asbs_s_cpu_gpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(scalar_align1_asbs_gpu_gpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(scalar_align1_asbs_s_cpu_cpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(scalar_align1_asbs_cpu_cpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(scalar_align1_asbs_s_gpu_cpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(scalar_align1_as_gpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(scalar_align1_as_cpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(scalar_align1_asbs_s_gpu_gpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(scalar_align1_asbs_cpu_gpu, fp64_ext));
        std::string prog_name = program_name();
        #ifdef VIENNACL_BUILD_INFO
        std::cout << "Creating program " << prog_name << std::endl;
        #endif
        context_.add_program(source, prog_name);
        viennacl::ocl::program & prog_ = context_.get_program(prog_name);
        prog_.add_kernel("asbs_gpu_cpu");
        prog_.add_kernel("asbs_s_cpu_gpu");
        prog_.add_kernel("asbs_gpu_gpu");
        prog_.add_kernel("asbs_s_cpu_cpu");
        prog_.add_kernel("asbs_cpu_cpu");
        prog_.add_kernel("asbs_s_gpu_cpu");
        prog_.add_kernel("as_gpu");
        prog_.add_kernel("as_cpu");
        prog_.add_kernel("asbs_s_gpu_gpu");
        prog_.add_kernel("asbs_cpu_gpu");
        init_done[context_.handle().get()] = true;
       } //if
     } //init
    }; // struct


  }  //namespace kernels
 }  //namespace linalg
}  //namespace viennacl
#endif

