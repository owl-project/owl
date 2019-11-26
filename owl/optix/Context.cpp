// ======================================================================== //
// Copyright 2019 Ingo Wald                                                 //
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


#include "Context.h"
#include "Module.h"
#include "Geometry.h"

namespace owl {

  template<size_t alignment>
  size_t smallestMultipleOf(size_t unalignedSize)
  {
    const size_t numBlocks = (unalignedSize+alignment-1)/alignment;
    return numBlocks*alignment;
  }

  std::string typeToString(const OWLDataType type)
  {
    throw std::runtime_error(std::string(__PRETTY_FUNCTION__)
                             +": not yet implemented for type #"
                             +std::to_string((int)type));
  }
  
  
  Buffer::SP Context::createBuffer()
  {
    PING;
    Buffer::SP buffer = std::make_shared<Buffer>(this);
    PING;
    assert(buffer);
    return buffer;
  }

  RayGen::SP
  Context::createRayGen(const std::shared_ptr<RayGenType> &type)
  {
    return std::make_shared<RayGen>(this,type);
  }

  GeometryGroup::SP Context::createGeometryGroup(size_t numChildren)
  {
    return std::make_shared<GeometryGroup>(this,numChildren);
  }

  InstanceGroup::SP Context::createInstanceGroup(size_t numChildren)
  {
    return std::make_shared<InstanceGroup>(this,numChildren);
  }


  RayGenType::SP
  Context::createRayGenType(Module::SP module,
                            const std::string &progName,
                            size_t varStructSize,
                            const std::vector<OWLVarDecl> &varDecls)
  {
    return std::make_shared<RayGenType>(this,
                                        module,progName,
                                        varStructSize,
                                        varDecls);
  }
  

  GeometryType::SP
  Context::createGeometryType(OWLGeometryKind kind,
                              size_t varStructSize,
                              const std::vector<OWLVarDecl> &varDecls)
  {
    switch(kind) {
    case OWL_GEOMETRY_TRIANGLES:
      return std::make_shared<TrianglesGeometryType>(this,varStructSize,varDecls);
    case OWL_GEOMETRY_USER:
      return std::make_shared<UserGeometryType>(this,varStructSize,varDecls);
    default:
      OWL_NOTIMPLEMENTED;
    }
  }

  Module::SP Context::createModule(const std::string &ptxCode)
  {
    return std::make_shared<Module>(ptxCode);
  }

  std::shared_ptr<Geometry> UserGeometryType::createGeometry()
  {
    GeometryType::SP self
      = std::dynamic_pointer_cast<GeometryType>(shared_from_this());
    assert(self);
    return std::make_shared<UserGeometry>(context,self);
  }

  std::shared_ptr<Geometry> TrianglesGeometryType::createGeometry()
  {
    GeometryType::SP self
      = std::dynamic_pointer_cast<GeometryType>(shared_from_this());
    assert(self);
    return std::make_shared<TrianglesGeometry>(context,self);
  }
  

  void Context::expBuildSBT()
  {
    std::cout << "=======================================================" << std::endl;
    PING;
    PRINT(groups.size());
    assert(!groups.empty());
    
    std::vector<int> sbtOffsetOfGroup(groups.size());
    size_t numHitGroupRecords = 0;
    size_t biggestVarStructSize = 0;
    for (int groupID=0;groupID<groups.size();groupID++) {
      sbtOffsetOfGroup[groupID] = numHitGroupRecords;
      Group::SP group = groups.getSP(groupID);
      if (!group)
        continue;
      GeometryGroup::SP gg
        = group->as<GeometryGroup>();
      if (!gg)
        continue;
      numHitGroupRecords += numRayTypes * gg->geometries.size();

      for (auto g : gg->geometries) {
        assert(g);
        biggestVarStructSize = std::max(biggestVarStructSize,g->type->varStructSize);
      }
    }
    PRINT(numHitGroupRecords);
    PRINT(biggestVarStructSize);

    size_t alignedSBTRecordSize
      = OPTIX_SBT_RECORD_HEADER_SIZE
      + smallestMultipleOf<OPTIX_SBT_RECORD_ALIGNMENT>(biggestVarStructSize);
    PRINT(alignedSBTRecordSize);
    size_t hitGroupRecordsSizeInBytes
      = alignedSBTRecordSize * numHitGroupRecords;
    
    std::cout << "=======================================================" << std::endl;
  }

} // ::owl
