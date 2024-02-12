// ======================================================================== //
// Copyright 2020-2021 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include "pyOWL/Context.h"

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

using namespace pyOWL;

PYBIND11_MODULE(py_owl, m) {
  // optional module docstring
  m.doc() = "OWL python wrappers";

  // create one context; almost all functions tare then per context
  m.def("context_create", &pyOWL::createContext,
        "Creates an OWL Context object");

  // define add function
  m.def("save_png_rgba8", &pyOWL::save_png_rgba8,
        "Saves a OWLBuffer of rgba8 values to a given file, in PNG format");
  
  // -------------------------------------------------------
  auto module
    = py::class_<pyOWL::Module,
                 std::shared_ptr<Module>>(m, "Module");

  // -------------------------------------------------------
  auto geomType
    = py::class_<pyOWL::GeomType,
                 std::shared_ptr<GeomType>>(m, "GeomType");
  geomType.def("set_closest_hit", &pyOWL::GeomType::setClosestHit);
  geomType.def("set_bounds_prog", &pyOWL::GeomType::setBoundsProg);
  geomType.def("set_intersect_prog", &pyOWL::GeomType::setIntersectProg);

  // -------------------------------------------------------
  auto geom
    = py::class_<pyOWL::Geom,
                 std::shared_ptr<Geom>>(m, "Geom");

  geom.def("set_prim_count", &pyOWL::Geom::setPrimCount);
  geom.def("set_vertices",&pyOWL::Geom::setVertices);
  geom.def("set_indices", &pyOWL::Geom::setIndices);
  geom.def("set_buffer", &pyOWL::Geom::setBuffer);
  geom.def("set_3f", &pyOWL::Geom::set3f);
  geom.def("set_1f", &pyOWL::Geom::set1f);
  
  // -------------------------------------------------------
  auto missProg
    = py::class_<pyOWL::MissProg,
                 std::shared_ptr<MissProg>>(m, "MissProg");

  missProg.def("set_3f", &pyOWL::MissProg::set3f);
  
  // -------------------------------------------------------
  auto rayGen
    = py::class_<pyOWL::RayGen,
                 std::shared_ptr<RayGen>>(m, "RayGen");

  rayGen.def("set_2i", &pyOWL::RayGen::set2i);
  rayGen.def("set_3f", &pyOWL::RayGen::set3f);
  rayGen.def("set_buffer", &pyOWL::RayGen::setBuffer);
  rayGen.def("set_group", &pyOWL::RayGen::setGroup);
  rayGen.def("launch",
              &pyOWL::RayGen::launch2D);
  
  // -------------------------------------------------------
  auto group
    = py::class_<pyOWL::Group,
                 std::shared_ptr<Group>>(m, "Group");
  group.def("build_accel", &pyOWL::Group::buildAccel);

  // -------------------------------------------------------
  auto buffer
    = py::class_<pyOWL::Buffer,
                 std::shared_ptr<Buffer>>(m, "Buffer");
  
  // -------------------------------------------------------
  auto context
    = py::class_<pyOWL::Context,
                 std::shared_ptr<Context>>(m, "Context");
  
  context.def(py::init<>());
  context.def("module_from_file",
              &pyOWL::Context::createModuleFromFile);
  context.def("module_create",
              &pyOWL::Context::createModuleFromString);
  context.def("geom_type_create", &pyOWL::Context::createGeomType);
  context.def("geom_create", &pyOWL::Context::createGeom);
  context.def("miss_prog_create", &pyOWL::Context::createMissProg);
  context.def("ray_gen_create", &pyOWL::Context::createRayGen);
  context.def("triangles_geom_group_create", &pyOWL::Context::createTrianglesGeomGroup);
  context.def("user_geom_group_create", &pyOWL::Context::createUserGeomGroup);
  context.def("instance_group_create", &pyOWL::Context::createInstanceGroup);
  context.def("device_buffer_create", &pyOWL::Context::createDeviceBuffer);
  context.def("host_pinned_buffer_create", &pyOWL::Context::createHostPinnedBuffer);

  context.def("build_programs",&pyOWL::Context::buildPrograms);
  context.def("build_pipeline",&pyOWL::Context::buildPipeline);
  context.def("build_SBT",&pyOWL::Context::buildSBT);
  context.def("context_destroy",&pyOWL::Context::destroy);
  
  // geom kinds 
  context.attr("GEOM_TRIANGLES") = py::int_((int)OWL_GEOM_TRIANGLES);
  context.attr("GEOM_USER")      = py::int_((int)OWL_GEOM_USER);
  // variable/data types
  context.attr("BUFPTR") = py::int_((int)OWL_BUFPTR);
  context.attr("INT")    = py::int_((int)OWL_INT);
  context.attr("INT2")   = py::int_((int)OWL_INT2);
  context.attr("INT3")   = py::int_((int)OWL_INT3);
  context.attr("INT4")   = py::int_((int)OWL_INT4);
  context.attr("FLOAT")  = py::int_((int)OWL_FLOAT);
  context.attr("FLOAT2") = py::int_((int)OWL_FLOAT2);
  context.attr("FLOAT3") = py::int_((int)OWL_FLOAT3);
  context.attr("FLOAT4") = py::int_((int)OWL_FLOAT4);
}
