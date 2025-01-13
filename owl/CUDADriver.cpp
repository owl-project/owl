#include "CUDADriver.h"
#ifdef _WIN32
# include <windows.h>
#else
# include <dlfcn.h>
#endif
#include "owl/common.h"

namespace owl {
#if OWL_CUDA_STATIC
# ifdef _WIN32
  void* getDriverFunction(const std::string& fctName)
  {
    static HMODULE libCUDA = LoadLibraryW(L"nvcuda.dll");
    
    if (!libCUDA) throw std::runtime_error("could not load nvcuda..dll");
    
    void* sym = (void*)GetProcAddress(libCUDA, fctName.c_str());
    if (!sym) throw std::runtime_error("could not find symbol '" + fctName + "' in libCUDA");
    return sym;
  }
# else
  void *getDriverFunction(const std::string &fctName)
  {
    static void *libCUDA = dlopen("libcuda.so.1",RTLD_LAZY|RTLD_LOCAL);
    if (!libCUDA) throw std::runtime_error("could not load libcuda.so.1");
    void *sym = dlsym(libCUDA,fctName.c_str());
    if (!sym) throw std::runtime_error("could not find symbol '"+fctName+"' in libCUDA");
    return sym;
  }
# endif

  CUresult _cuModuleGetFunction ( CUfunction* hfunc,
                                  CUmodule hmod,
                                  const char* name )
  {
    typedef CUresult (*Fct) ( CUfunction* hfunc,
                              CUmodule hmod,
                              const char* name );
    static Fct fct = (Fct)getDriverFunction("cuModuleGetFunction");
    
    return fct(hfunc,hmod,name);
  }
  CUresult _cuModuleGetGlobal(CUdeviceptr *dptr,
                              size_t *bytes,
                              CUmodule hmod,
                              const char *name)
  {
    typedef CUresult (*Fct)(CUdeviceptr *dptr,
                           size_t *bytes,
                           CUmodule hmod,
                           const char *name);
    static Fct fct = (Fct)getDriverFunction("cuModuleGetGlobal");
    return fct(dptr,bytes,hmod,name);
  }
  
  CUresult _cuCtxGetCurrent(CUcontext *pctx)
  {
    typedef CUresult (*Fct)(CUcontext *pctx);
    static Fct fct = (Fct)getDriverFunction("cuCtxGetCurrent");
    return fct(pctx);
  }

  CUresult _cuModuleLoadDataEx(CUmodule *module,
                               const void *image,
                               unsigned int numOptions,
                               CUjit_option *options,
                               void **optionValues)
  {
    typedef CUresult (*Fct)(CUmodule *module,
                            const void *image,
                            unsigned int numOptions,
                            CUjit_option *options,
                            void **optionValues);
    static Fct fct = (Fct)getDriverFunction("cuModuleLoadDataEx");
    return fct(module,image,numOptions,options,optionValues);
  }
    

  CUresult _cuModuleLoad(CUmodule *module, const char *fname)
  {
    typedef CUresult (*Fct)(CUmodule *module, const char *fname);
    static Fct fct = (Fct)getDriverFunction("cuModuleLoad");
    return fct(module,fname);
  }
  
  CUresult _cuModuleUnload(CUmodule module)
  {
    typedef CUresult (*Fct)(CUmodule module);
    static Fct fct = (Fct)getDriverFunction("cuModuleUnload");
    return fct(module);
  }
  
  CUresult _cuGetErrorName ( CUresult error,
                             const char** pStr )
  {
    typedef CUresult (*cuGetErrorNameFct)( CUresult error,
                                           const char** pStr );
    static cuGetErrorNameFct fct 
      = (cuGetErrorNameFct)
      getDriverFunction("cuGetErrorName");
    return fct(error,pStr);
  }
  CUresult _cuLaunchKernel ( CUfunction f,
                             unsigned int  gridDimX,
                             unsigned int  gridDimY,
                             unsigned int  gridDimZ,
                             unsigned int  blockDimX,
                             unsigned int  blockDimY,
                             unsigned int  blockDimZ,
                             unsigned int  sharedMemBytes,
                             CUstream hStream,
                             void** kernelParams,
                             void** extra )
  {
    static CUresult (*__cuLaunchKernel)( CUfunction f,
                                         unsigned int  gridDimX,
                                         unsigned int  gridDimY,
                                         unsigned int  gridDimZ,
                                         unsigned int  blockDimX,
                                         unsigned int  blockDimY,
                                         unsigned int  blockDimZ,
                                         unsigned int  sharedMemBytes,
                                         CUstream hStream,
                                         void** kernelParams,
                                         void** extra )
      = (CUresult (*)( CUfunction f,
                       unsigned int  gridDimX,
                       unsigned int  gridDimY,
                       unsigned int  gridDimZ,
                       unsigned int  blockDimX,
                       unsigned int  blockDimY,
                       unsigned int  blockDimZ,
                       unsigned int  sharedMemBytes,
                       CUstream hStream,
                       void** kernelParams,
                       void** extra ))
      getDriverFunction("cuLaunchKernel");
    return __cuLaunchKernel ( f,
                              gridDimX,
                              gridDimY,
                              gridDimZ,
                              blockDimX,
                              blockDimY,
                              blockDimZ,
                              sharedMemBytes,
                              hStream,
                              kernelParams,
                              extra );
  }
  
#else
  CUresult _cuModuleLoadDataEx(CUmodule *module,
                               const void *image,
                               unsigned int numOptions,
                               CUjit_option *options,
                               void **optionValues)
  {
    return  cuModuleLoadDataEx(module,
                               image,
                               numOptions,
                               options,
                               optionValues);
  }
  
  CUresult _cuCtxGetCurrent(CUcontext *pctx)
  {
    return cuCtxGetCurrent(pctx);
  }
  
  CUresult _cuModuleGetGlobal(CUdeviceptr *dptr,
                              size_t *bytes,
                              CUmodule hmod,
                              const char *name)
  {
    return  cuModuleGetGlobal(dptr,
                              bytes,
                              hmod,
                              name);
      };

    CUresult _cuModuleUnload(CUmodule module)
  {
    return cuModuleUnload(module);
  }
  

  CUresult _cuGetErrorName ( CUresult error,
                            const char** pStr )
  { return cuGetErrorName(error,pStr); }
  CUresult _cuLaunchKernel ( CUfunction f,
                            unsigned int  gridDimX,
                            unsigned int  gridDimY,
                            unsigned int  gridDimZ,
                            unsigned int  blockDimX,
                            unsigned int  blockDimY,
                            unsigned int  blockDimZ,
                            unsigned int  sharedMemBytes,
                            CUstream hStream,
                            void** kernelParams,
                            void** extra )
  { return cuLaunchKernel ( f,
                            gridDimX,
                            gridDimY,
                            gridDimZ,
                            blockDimX,
                            blockDimY,
                            blockDimZ,
                            sharedMemBytes,
                            hStream,
                            kernelParams,
                            extra ); }
  
   CUresult _cuModuleGetFunction ( CUfunction* hfunc,
                                  CUmodule hmod,
                                  const char* name )
   { return cuModuleGetFunction(hfunc,hmod,name); }
#endif

  
}

