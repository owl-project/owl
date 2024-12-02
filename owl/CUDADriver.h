#include <cuda_runtime.h>
#include <cuda.h>

namespace owl {
  CUresult _cuGetErrorName ( CUresult error,
                             const char** pStr );
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
                             void** extra );
  
  CUresult _cuModuleGetFunction ( CUfunction* hfunc,
                                  CUmodule hmod,
                                  const char* name );
  
  CUresult _cuModuleLoadDataEx(CUmodule *module,
                               const void *image,
                               unsigned int numOptions,
                               CUjit_option *options,
                               void **optionValues);

  CUresult _cuModuleGetGlobal(CUdeviceptr *dptr,
                              size_t *bytes,
                              CUmodule hmod,
                              const char *name);

  CUresult _cuCtxGetCurrent(CUcontext *pctx);

  CUresult _cuModuleUnload(CUmodule module);
  
}
