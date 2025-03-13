# ======================================================================== #
# Copyright 2019-2021 Ingo Wald                                            #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
# ======================================================================== #

find_package(OptiX REQUIRED)

include(cmake/configure_tbb.cmake)

set(OWL_SOURCES
  # -------------------------------------------------------
  # infrastructure
  # -------------------------------------------------------
  include/owl/owl.h
  include/owl/owl_host.h
  include/owl/owl_device.h
  include/owl/common/arrayND/array2D.h
  include/owl/common/arrayND/array3D.h
  include/owl/common/math/AffineSpace.h
  include/owl/common/math/box.h
  include/owl/common/math/constants.h
  include/owl/common/math/fixedpoint.h
  include/owl/common/math/LinearSpace.h
  include/owl/common/math/Quaternion.h
  include/owl/common/math/random.h
  include/owl/common/math/vec/compare.h
  include/owl/common/math/vec/functors.h
  include/owl/common/math/vec/rotate.h
  include/owl/common/math/vec.h
  include/owl/common/owl-common.h
  include/owl/common/parallel/parallel_for.h
  include/owl/owl.h
  include/owl/owl_device.h
  include/owl/owl_device_buffer.h
  include/owl/owl_host.h
  helper/cuda.h
  helper/optix.h
  common.h

  cmake/configure_tbb.cmake
  cmake/FindOptiX.cmake

  # -------------------------------------------------------
  # API layer
  # -------------------------------------------------------
  APIContext.h
  APIContext.cpp
  APIHandle.h
  APIHandle.cpp
  impl.cpp

  # -------------------------------------------------------
  # data handling
  # -------------------------------------------------------
  Buffer.h
  Buffer.cpp
  Texture.h
  Texture.cpp

  # -------------------------------------------------------
  # base classes
  # -------------------------------------------------------
  CUDADriver.h
  CUDADriver.cpp
  Object.h
  Object.cpp
  Module.h
  Module.cpp
  RegisteredObject.h
  RegisteredObject.cpp
  DeviceContext.h
  DeviceContext.cpp

  ObjectRegistry.h
  ObjectRegistry.cpp
  Context.h
  Context.cpp

  # -------------------------------------------------------
  # program types
  # -------------------------------------------------------
  SBTObject.h
  SBTObject.cpp
  RayGen.h
  RayGen.cpp
  LaunchParams.h
  LaunchParams.cpp
  MissProg.h
  MissProg.cpp
  Variable.h
  Variable.cpp

  # -------------------------------------------------------
  # geometries
  # -------------------------------------------------------
  Geometry.h
  Geometry.cpp
  Triangles.h
  Triangles.cu
  UserGeom.h
  UserGeom.cu
  CurvesGeom.h
  CurvesGeom.cu
  SphereGeom.h
  SphereGeom.cu

  # -------------------------------------------------------
  # accel structures
  # -------------------------------------------------------
  Group.h
  Group.cpp
  InstanceGroup.h
  InstanceGroup.cu
  TrianglesGeomGroup.h
  TrianglesGeomGroup.cpp
  CurvesGeomGroup.h
  CurvesGeomGroup.cpp
  UserGeomGroup.h
  UserGeomGroup.cpp
  SphereGeomGroup.h
  SphereGeomGroup.cpp
)

add_library(owl-config INTERFACE)
add_library(owl SHARED ${OWL_SOURCES})
add_library(owl_static STATIC ${OWL_SOURCES})
target_compile_definitions(owl-config INTERFACE -DOWL_CUDA_STATIC=1)
#target_compile_definitions(owl_static PUBLIC -DOWL_CUDA_STATIC=1)
add_library(owl_object OBJECT ${OWL_SOURCES})

set_target_properties(owl
  PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_USE_STATIC_CUDA_RUNTIME ON
  CXX_VISIBILITY_PRESET  hidden
  CUDA_VISIBILITY_PRESET hidden
)
set_target_properties(owl_static
  PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_USE_STATIC_CUDA_RUNTIME ON
  CXX_VISIBILITY_PRESET  hidden
  CUDA_VISIBILITY_PRESET hidden
)
set_target_properties(owl_object
  PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_USE_STATIC_CUDA_RUNTIME ON
  CXX_VISIBILITY_PRESET  hidden
  CUDA_VISIBILITY_PRESET hidden
)

target_link_libraries(owl-config INTERFACE
  OptiX::OptiX
  )
if (OWL_CUDA_STATIC)
  target_compile_definitions(owl-config INTERFACE -DOWL_CUDA_STATIC=1)
endif()
target_link_libraries(owl PUBLIC owl-config)
target_link_libraries(owl_static PUBLIC owl-config)
target_link_libraries(owl_object PUBLIC owl-config)

if (OWL_HAVE_TBB AND TBB_FOUND)
  if (TARGET TBB::tbb)
    target_link_libraries(owl PUBLIC TBB::tbb)
  else()
    target_link_libraries(owl PUBLIC ${TBB_LIBRARIES})
    target_include_directories(owl PUBLIC ${TBB_INCLUDE_DIR})
  endif()
endif()
# **NO** TBB for owl-static; there is not static tbb lib
target_compile_definitions(owl_static PUBLIC -DOWL_HAVE_TBB=0)
target_compile_definitions(owl_object PUBLIC -DOWL_HAVE_TBB=0)

# bind OWL to CUDA (includes and library)
find_package(CUDAToolkit REQUIRED)
if (OWL_CUDA_STATIC)
  target_link_libraries(owl-config INTERFACE
#    ${OWL_CUDA_STATIC}
    CUDA::cudart_static
    )
#  target_compile_definitions(owl_static PUBLIC -DOWL_CUDA_STATIC=1)
else()
  target_link_libraries(owl-config INTERFACE
    CUDA::cuda_driver
    CUDA::cudart_static
    )
endif()

target_include_directories(owl-config
  INTERFACE
    ${PROJECT_SOURCE_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/include
)
#set_property(TARGET owl-config
#  PROPERTY
#  CUDA_ARCHITECTURES "${OWL_CUDA_ARCHITECTURES}"
#  CUDA_PTX_COMPILATION ON
#  CUDA_COMPILE_PTX_COMPILATION OFF
#  )
#set_property(TARGET owl_static
#  PROPERTY CUDA_ARCHITECTURES all-major
#  )

if (WIN32)
  target_compile_definitions(owl-config INTERFACE -DNOMINMAX)
endif()

add_library(owl::owl ALIAS owl)
add_library(owl::owl_static ALIAS owl_static)
add_library(owl::owl_object ALIAS owl_object)
