// ======================================================================== //
// Copyright 2019-2025 Ingo Wald                                            //
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

#include "World.h"

namespace samples {

  using device::Triangle;
  
  int World::addMaterial(const Material &material)
  {
    materials.push_back(material);
    return materials.size()-1;
  }

  void World::addSphericalTriangle(const affine3f &xfm,
                                   int recDepth,
                                   const vec3f &a,
                                   const vec3f &b,
                                   const vec3f &c,
                                   int material)
  {
    if (recDepth <= 0) {
      if (dot(cross(b-a,c-a),a) < 0.f)
        
        triangles.push_back({xfmPoint(xfm,b),xfmPoint(xfm,a),xfmPoint(xfm,c),
#if HAVE_SHADING_NORMALS 
              b,a,c,
#endif
              material});
      else
        triangles.push_back({xfmPoint(xfm,a),xfmPoint(xfm,b),xfmPoint(xfm,c),
#if HAVE_SHADING_NORMALS 
              a,b,c,
#endif
              material});
    } else {
      
      const vec3f ab = normalize(a+b);
      const vec3f bc = normalize(b+c);
      const vec3f ca = normalize(c+a);
      
      addSphericalTriangle(xfm,recDepth-1,ab,bc,ca,material);
      addSphericalTriangle(xfm,recDepth-1,a,ab,ca,material);
      addSphericalTriangle(xfm,recDepth-1,b,bc,ab,material);
      addSphericalTriangle(xfm,recDepth-1,c,ca,bc,material);
    }
  }

  void World::addSphere(const vec3f center, const float radius,
                        int material,
                        int recDepth)
  {
    const affine3f xfm = affine3f::translate(center) * affine3f::scale(radius);
    addSphericalTriangle(xfm,recDepth,vec3f(+1,0,0),vec3f(0,+1,0),vec3f(0,0,+1),material);
    addSphericalTriangle(xfm,recDepth,vec3f(-1,0,0),vec3f(0,+1,0),vec3f(0,0,+1),material);

    addSphericalTriangle(xfm,recDepth,vec3f(+1,0,0),vec3f(0,-1,0),vec3f(0,0,+1),material);
    addSphericalTriangle(xfm,recDepth,vec3f(-1,0,0),vec3f(0,-1,0),vec3f(0,0,+1),material);

    addSphericalTriangle(xfm,recDepth,vec3f(+1,0,0),vec3f(0,+1,0),vec3f(0,0,-1),material);
    addSphericalTriangle(xfm,recDepth,vec3f(-1,0,0),vec3f(0,+1,0),vec3f(0,0,-1),material);

    addSphericalTriangle(xfm,recDepth,vec3f(+1,0,0),vec3f(0,-1,0),vec3f(0,0,-1),material);
    addSphericalTriangle(xfm,recDepth,vec3f(-1,0,0),vec3f(0,-1,0),vec3f(0,0,-1),material);
  }

  void World::addSphere(const vec3f center,
                        const float radius,
                        int materialID)
  {
    // tessellate w/ given number of subdivisions
    addSphere(center,radius,materialID,
              sphereRecDepth);
  }

  void World::finalize()
  {
    std::cout << "#prime-cuda: done w/ creating input scene, found "
              << prettyNumber(triangles.size()) << " triangles" 
              << std::endl;
    context = opContextCreate(OP_CONTEXT_GPU,0);
    assert(context);

    std::vector<vec3f> vertices;
    std::vector<vec3i> indices;
    for (int i=0;i<triangles.size();i++) {
      vertices.push_back(triangles[i].A);
      vertices.push_back(triangles[i].B);
      vertices.push_back(triangles[i].C);
      indices.push_back(3*i+vec3i(0,1,2));
    }
    
    OPGeom mesh = opMeshCreate(context,0ull,
                               (float *)vertices.data(),
                               vertices.size(),
                               sizeof(vec3f),
                               (int *)indices.data(),
                               indices.size(),
                               sizeof(vec3i));
    OPGroup group = opGroupCreate(context,&mesh,1);
    model = opModelCreate(context,&group,nullptr,1);
    assert(model);
  }

  int World::sphereRecDepth = 3;
  
  inline vec3f randomPointInUnitSphere(Random &rnd)
  {
    vec3f p;
    do {
      p = 2.f*vec3f(rnd(),rnd(),rnd()) - vec3f(1.f);
    } while (dot(p,p) >= 1.f);
    return p;
  }

  std::shared_ptr<World> createScene(Random &random)
  {
    std::shared_ptr<World> world = std::make_shared<World>();
    world->addSphere(vec3f(0.f, -1000.0f, -1.f), 1000.f,
                     world->addMaterial(Lambertian(vec3f(0.5f, 0.5f, 0.5f))));
    for (int a = -11; a < 11; a++) {
      for (int b = -11; b < 11; b++) {
        float choose_mat = random();
        int materialID;
        if (choose_mat < 0.8f) {
          materialID = world->addMaterial(Lambertian(vec3f(random()*random(),
                                                           random()*random(),
                                                           random()*random())));
        }
        else if (choose_mat < 0.95f) {
          materialID = world->addMaterial(Metal(vec3f(0.5f*(1.0f + random()),
                                                      0.5f*(1.0f + random()),
                                                      0.5f*(1.0f + random())),
                                                0.5f*random()));
        }
        else {
          materialID = world->addMaterial(Dielectric(1.5f));
        }
                           
        float choose_type = random();
        vec3f center(a + random(), 0.2f, b + random());
        if (choose_type < .5f) {
          const int NUM_VERTICES = 8;
          static const vec3f unitBoxVertices[NUM_VERTICES] =
            {
             {-1.f, -1.f, -1.f},
             {+1.f, -1.f, -1.f},
             {+1.f, +1.f, -1.f},
             {-1.f, +1.f, -1.f},
             {-1.f, +1.f, +1.f},
             {+1.f, +1.f, +1.f},
             {+1.f, -1.f, +1.f},
             {-1.f, -1.f, +1.f},
            };

          const int NUM_INDICES = 12;
          static const vec3i unitBoxIndices[NUM_INDICES] =
            {
             {0, 2, 1}, //face front
             {0, 3, 2},
             {2, 3, 4}, //face top
             {2, 4, 5},
             {1, 2, 5}, //face right
             {1, 5, 6},
             {0, 7, 4}, //face left
             {0, 4, 3},
             {5, 4, 7}, //face back
             {5, 7, 6},
             {0, 6, 7}, //face bottom
             {0, 1, 6}
            };

          const float size = .2f;
          const vec3f U = normalize(randomPointInUnitSphere(random));
          owl::affine3f xfm = owl::frame(U);
          xfm = owl::affine3f(owl::linear3f::rotate(U,random())) * xfm;
          xfm = owl::affine3f(owl::linear3f::scale(.7f*size)) * xfm;
          xfm = owl::affine3f(owl::affine3f::translate(center)) * xfm;

          for (int i=0;i<NUM_INDICES;i++) {
            vec3f A = unitBoxVertices[unitBoxIndices[i].x];
            vec3f B = unitBoxVertices[unitBoxIndices[i].y];
            vec3f C = unitBoxVertices[unitBoxIndices[i].z];
            A = owl::xfmPoint(xfm,A);
            B = owl::xfmPoint(xfm,B);
            C = owl::xfmPoint(xfm,C);
#if HAVE_SHADING_NORMALS
            const vec3f N = normalize(cross(B-A,C-A));
            world->triangles.push_back({A,B,C,N,N,N,materialID});
#else
            world->triangles.push_back({A,B,C,materialID});
#endif
          }
        } else {
          world->addSphere(center, 0.2f, materialID);
        }
      }
    }
    world->addSphere(vec3f( 0.f, 1.f, 0.f), 1.f, world->addMaterial(Dielectric(1.5f)));
    world->addSphere(vec3f(-4.f, 1.f, 0.f), 1.f, world->addMaterial(Lambertian(vec3f(0.4f, 0.2f, 0.1f))));
    world->addSphere(vec3f( 4.f, 1.f, 0.f), 1.f, world->addMaterial(Metal(vec3f(0.7f, 0.6f, 0.5f), 0.0f)));
    return world;
  }


  
}
