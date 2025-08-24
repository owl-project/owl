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

#include <owl/owl_prime.h>

#include "owl/common/parallel/parallel_for.h"
#include "owl/common/arrayND/array2D.h"
#include "owl/common/math/AffineSpace.h"
#include <random>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"
#include <cmath>
#include <random>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <memory>

#define HAVE_SHADING_NORMALS 1

namespace samples {
  using namespace owl::common;

  /*! careful - this has to have the same data layout os OPRay */
  struct Ray {
    vec3f origin;
    float tmin;
    vec3f direction;
    float tmax;
  };
  using Hit = OPHit;
    
  const int TILE_SIZE = 128;

  struct RNG {
    RNG(int seed) : gen(seed) {}
    
    inline float operator()()
    {
      std::uniform_real_distribution<float> dis(0.f, 1.f);
      return dis(gen);
    }
    
    std::mt19937 gen;//(0); //Standard mersenne_twister_engine seeded with rd()
  };
    
  float rnd()
  {
    static RNG g_rng(0);
    return g_rng();
  }

  inline float schlick(float cosine,
                       float ref_idx)
  {
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0f - r0)*powf((1.0f - cosine), 5.0f);
  }

  inline bool refract(const vec3f &v,
                      const vec3f &n,
                      float ni_over_nt,
                      vec3f &refracted)
  {
    vec3f uv = normalize(v);
    float dt = dot(uv, n);
    float discriminant = 1.0f - ni_over_nt * ni_over_nt*(1 - dt * dt);
    if (discriminant > 0) {
      refracted = ni_over_nt * (uv - n * dt) - n * sqrtf(discriminant);
      return true;
    }
    else
      return false;
  }

  inline  vec3f reflect(const vec3f &v, const vec3f &n)
  {
    return v - 2.0f*dot(v, n)*n;
  }

  inline vec3f random_in_unit_disk(RNG &local_rand_state) {
    vec3f p;
    do {
      p = 2.0f*vec3f(local_rand_state(), local_rand_state(), 0) - vec3f(1, 1, 0);
    } while (dot(p, p) >= 1.0f);
    return p;
  }


#define RANDVEC3F vec3f(rnd(),rnd(),rnd())

  inline vec3f random_in_unit_sphere(RNG &rnd) {
    vec3f p;
    do {
      p = 2.0f*RANDVEC3F - vec3f(1.f, 1.f, 1.f);
    } while (dot(p,p) >= 1.0f);
    return p;
  }

  struct ScatterEvent {
    // in:
    vec3f inDir;
    vec3f P;
    vec3f N;
    // out
    vec3f out_dir;
    vec3f out_org;
    vec3f attenuation;
  };

  /*! abstraction for a material that can create, and parameterize,
    a newly created GI's material and closest hit program */
  struct Material {
    virtual bool scatter(ScatterEvent &scatter,
                         RNG   &rndState) = 0;
  };

  /*! host side code for the "Lambertian" material; the actual
    sampling code is in the programs/lambertian.cu closest hit program */
  struct Lambertian : public Material {
    /*! constructor */
    Lambertian(const vec3f &albedo) : albedo(albedo) {}
    
    /*! the actual scatter function - in Pete's reference code, that's
      a virtual function, but since we have a different function per
      program we do not need this here */
    virtual bool scatter(ScatterEvent &scatter,
                         RNG   &rndState) override
    {
      if (dot(scatter.inDir,scatter.N) > 0.f)
        scatter.N = -scatter.N;
      vec3f target
        = scatter.P + (scatter.N + random_in_unit_sphere(rndState));
      
      // return scattering event
      scatter.out_org     = scatter.P;
      scatter.out_dir     = target-scatter.P;
      scatter.attenuation = albedo;
      return true;
    }

    
    const vec3f albedo;
  };

  /*! host side code for the "Metal" material; the actual
    sampling code is in the programs/metal.cu closest hit program */
  struct Metal : public Material {
    /*! constructor */
    Metal(const vec3f &albedo, const float fuzz) : albedo(albedo), fuzz(fuzz) {}
    /*! the actual scatter function - in Pete's reference code, that's a
      virtual function, but since we have a different function per program
      we do not need this here */
    virtual bool scatter(ScatterEvent &scatter,
                         RNG   &rndState) override
    {
      if (dot(scatter.inDir,scatter.N) > 0.f)
        scatter.N = -scatter.N;
      vec3f reflected = reflect(normalize(scatter.inDir),scatter.N);
      scatter.out_org     = scatter.P;
      scatter.out_dir     = (reflected+fuzz*random_in_unit_sphere(rndState));
      scatter.attenuation = vec3f(1.f);
      return (dot(scatter.out_dir, scatter.N) > 0.f);
    }
    
    const vec3f albedo;
    const float fuzz;
  };

  /*! host side code for the "Dielectric" material; the actual
    sampling code is in the programs/dielectric.cu closest hit program */
  struct Dielectric : public Material {
    /*! constructor */
    Dielectric(const float ref_idx) : ref_idx(ref_idx) {}
    /*! the actual scatter function - in Pete's reference code, that's a
      virtual function, but since we have a different function per program
      we do not need this here */
    virtual bool scatter(ScatterEvent &scatter,
                         RNG   &rndState) override
    {
      vec3f outward_normal;
      vec3f reflected = reflect(scatter.inDir, scatter.N);
      float ni_over_nt;
      scatter.attenuation = vec3f(1.f, 1.f, 1.f); 
      vec3f refracted;
      float reflect_prob;
      float cosine;

      float NdotD = dot(scatter.inDir, scatter.N);
      scatter.inDir = normalize(scatter.inDir);
      if (NdotD > 0.f) {
        outward_normal = -scatter.N;
        ni_over_nt = ref_idx;
        cosine = sqrtf(1.f - ref_idx*ref_idx*(1.f-NdotD*NdotD));
      }
      else {
        outward_normal = scatter.N;
        ni_over_nt = 1.0 / ref_idx;
        cosine = -NdotD;
      }
      
      if (refract(scatter.inDir, outward_normal, ni_over_nt, refracted)) 
        reflect_prob = schlick(cosine, ref_idx);
      else 
        reflect_prob = 1.f;

      scatter.out_org = scatter.P;
      if (rnd() < reflect_prob) 
        scatter.out_dir = reflected;
      else 
        scatter.out_dir = refracted;
  
      return true;
    }

    const float ref_idx;
  };

  struct World {
    struct Triangle {
      vec3f A, B, C;
    };
    struct TriangleData {
#if HAVE_SHADING_NORMALS
      vec3f Na,Nb,Nc;
#endif
      Material *material;
    };

    World(OPContextType contextType);
    
    void addSphericalTriangle(const affine3f &xfm,
                              int recDepth,
                              const vec3f &a,
                              const vec3f &b,
                              const vec3f &c,
                              Material *material);
    void addSphere(const vec3f center,
                   const float radius,
                   Material *material,
                   int recDepth);

    void addSphere(const vec3f center,
                   const float radius,
                   Material *material)
    {
      addSphere(center,radius,material,
                sphereRecDepth);
    }
    static int sphereRecDepth;
    
    void finalize()
    {
      std::vector<vec3i> indices;
      for (int i=0;i<(int)triangles.size();i++)
        indices.push_back(3*i+vec3i(0,1,2));
      OPGeom mesh = opMeshCreate(context,0ull,
                                 (float *)triangles.data(),
                                 3*triangles.size(),
                                 sizeof(vec3f),
                                 (int *)indices.data(),
                                 indices.size(),
                                 sizeof(vec3i));
      OPGroup group = opGroupCreate(context,&mesh,1);
      handle = opModelCreate(context,&group,nullptr,1);
    }

    std::vector<Triangle> triangles;
    std::vector<TriangleData> triangleData;
    OPModel   handle = nullptr;
    OPContext context = nullptr;
  };

  World::World(OPContextType contextType)
  {
    context = opContextCreate(contextType,0);
  }
  
  int World::sphereRecDepth = 3;
  
  void World::addSphericalTriangle(const affine3f &xfm,
                                   int recDepth,
                                   const vec3f &a,
                                   const vec3f &b,
                                   const vec3f &c,
                                   Material *material)
  {
    if (recDepth <= 0) {
      if (dot(cross(b-a,c-a),a) < 0.f) {
        
        triangles.push_back({xfmPoint(xfm,b),xfmPoint(xfm,a),xfmPoint(xfm,c)});
        triangleData.push_back({
#if HAVE_SHADING_NORMALS
            // we only use scale and translate, no need to transform
            b,a,c,
#endif
            material});
      } else {
        triangles.push_back({xfmPoint(xfm,a),xfmPoint(xfm,b),xfmPoint(xfm,c)});
        triangleData.push_back({
#if HAVE_SHADING_NORMALS 
            // we only use scale and translate, no need to transform
            a,b,c,
#endif
            material});
      }
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

  void World::addSphere(const vec3f center,
                        const float radius,
                        Material *material,
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



  struct Camera {
    Camera(const vec3f &lookfrom,
           const vec3f &lookat,
           const vec3f &vup, 
           float vfov,
           float aspect,
           float aperture,
           float focus_dist) 
    { // vfov is top to bottom in degrees
      lens_radius = aperture / 2.0f;
      float theta = vfov * ((float)M_PI) / 180.0f;
      float half_height = tan(theta / 2.0f);
      float half_width = aspect * half_height;
      origin = lookfrom;
      w = normalize(lookfrom - lookat);
      u = normalize(cross(vup, w));
      v = cross(w, u);
      lower_left_corner = origin - half_width * focus_dist*u - half_height * focus_dist*v - focus_dist * w;
      horizontal = 2.0f*half_width*focus_dist*u;
      vertical = 2.0f*half_height*focus_dist*v;
    }

    Ray generateRay(float s, float t, RNG &rnd) 
    {
      const vec3f rd = lens_radius * random_in_unit_disk(rnd);
      const vec3f lens_offset = u * rd.x + v * rd.y;
      const vec3f origin = this->origin + lens_offset;
      const vec3f direction
        = lower_left_corner
        + s * horizontal
        + t * vertical
        - origin;
      
      Ray ray;
      (vec3f&)ray.origin = origin;
      (vec3f&)ray.direction = direction;
      ray.tmin = 0.f;
      ray.tmax = std::numeric_limits<float>::infinity();
      return ray;
    }

    vec3f origin;
    vec3f lower_left_corner;
    vec3f horizontal;
    vec3f vertical;
    vec3f u, v, w;
    float lens_radius;
  };

  struct FrameBuffer {
    FrameBuffer(const vec2i &size)
      : size(size), pixels(size.x*size.y)
    {}
    
    const vec2i        size;
    std::vector<vec3f> pixels;
  };



  std::shared_ptr<World> createScene(OPContextType contextType)
  {
    std::shared_ptr<World> world = std::make_shared<World>(contextType);
    world->addSphere(vec3f(0.f, -1000.0f, -1.f), 1000.f,
                     new Lambertian(vec3f(0.5f, 0.5f, 0.5f)));
    
    for (int a = -11; a < 11; a++) {
      for (int b = -11; b < 11; b++) {
        float choose_mat = rnd();
        vec3f center(a + rnd(), 0.2f, b + rnd());
        if (choose_mat < 0.8f) {
          world->addSphere(center, 0.2f,
                           new Lambertian(vec3f(rnd()*rnd(), rnd()*rnd(), rnd()*rnd())));
        }
        else if (choose_mat < 0.95f) {
          world->addSphere(center, 0.2f,
                           new Metal(vec3f(0.5f*(1.0f + rnd()), 0.5f*(1.0f + rnd()), 0.5f*(1.0f + rnd())), 0.5f*rnd()));
        }
        else {
          world->addSphere(center, 0.2f,
                           new Dielectric(1.5f));
        }
      }
    }
    world->addSphere(vec3f( 0.f, 1.f, 0.f), 1.f, new Dielectric(1.5f));
    world->addSphere(vec3f(-4.f, 1.f, 0.f), 1.f, new Lambertian(vec3f(0.4f, 0.2f, 0.1f)));
    world->addSphere(vec3f( 4.f, 1.f, 0.f), 1.f, new Metal(vec3f(0.7f, 0.6f, 0.5f), 0.0f));
    return world;
  }


  
  unsigned char to255(float f)
  {
    if (f <= 0.f) return 0;
    if (f >= 1.f) return 255;
    return (unsigned char)(f*255.9f);
  }

  void savePNG(const std::string &fileName,
               FrameBuffer &image)
  {
    std::vector<unsigned char> rgba;
    for (int iy=image.size.y-1;iy>=0;--iy) 
      for (int ix=0;ix<image.size.x;ix++) {
        vec3f pixel = image.pixels[ix+image.size.x*iy];
        rgba.push_back(to255(pixel.x));
        rgba.push_back(to255(pixel.y));
        rgba.push_back(to255(pixel.z));
        rgba.push_back(255);
      }
    std::cout << "#owl-prime.rtWeekend: writing image " << fileName << std::endl;
    stbi_write_png(fileName.c_str(),image.size.x,image.size.y,4,
                   rgba.data(),image.size.x*sizeof(uint32_t));
  }

  vec3f missColor(const vec3f &rayDir)//vec2i pixelID, const vec2i &fbSize)
  {
    // const vec3f rayDir = normalize(ray.direction);
    const float t = 0.5f*(rayDir.y + 1.0f);
    const vec3f c = (1.0f - t)*vec3f(1.0f, 1.0f, 1.0f) + t * vec3f(0.5f, 0.7f, 1.0f);
    return c;
  }
  
  void renderFrame(std::shared_ptr<World> world,
                   std::shared_ptr<Camera> camera,
                   std::shared_ptr<FrameBuffer> fb,
                   int numSamplesPerPixel,
                   int maxPathLength)
  {
    vec2i numTiles = divRoundUp(fb->size,vec2i(TILE_SIZE));
    std::mutex mutex;
    array2D::serial_for
      (numTiles,
       [&](const vec2i &tileID){
        RNG random(array2D::linear(tileID,numTiles));
        std::vector<bool>  valid(TILE_SIZE*TILE_SIZE);
        std::vector<vec3f> color(TILE_SIZE*TILE_SIZE);
        array2D::for_each(vec2i(TILE_SIZE),[&](const vec2i &l_pixelID){
            const int lID = array2D::linear(l_pixelID,vec2i(TILE_SIZE));
            const vec2i pixelID = tileID*TILE_SIZE+l_pixelID;
            valid[lID] = (pixelID.x < fb->size.x && pixelID.y < fb->size.y);
            color[lID] = vec3f(0.f);
          });
        
        std::vector<Ray> rays(TILE_SIZE*TILE_SIZE);
        std::vector<Hit> isecs(TILE_SIZE*TILE_SIZE);
        std::vector<vec3f> weight(TILE_SIZE*TILE_SIZE);
        std::vector<bool>  active(TILE_SIZE*TILE_SIZE);
        for (int pixelSampleID=0;pixelSampleID<numSamplesPerPixel;pixelSampleID++) {
          
          // -------------------------------------------------------
          // ray generation
          // -------------------------------------------------------
          array2D::
            for_each
            // parallel_for
            (vec2i(TILE_SIZE),[&](const vec2i &l_pixelID){
              const int lID = array2D::linear(l_pixelID,vec2i(TILE_SIZE));
              
              active[lID] = valid[lID];
              weight[lID] = vec3f(1.f);
              if (!valid[lID]) {
                rays[lID].tmin = 0.f;
                rays[lID].tmax = -1.f;
                return;
              }
              
              const vec2i pixelID = tileID*TILE_SIZE+l_pixelID;
              float u = float(pixelID.x + random()) / float(fb->size.x);
              float v = float(pixelID.y + random()) / float(fb->size.y);
              rays[lID] = camera->generateRay(u,v,random);
              isecs[lID].primID = -1;
            });

          for (int depth=0;depth<maxPathLength;depth++) {
            // -------------------------------------------------------
            // trace
            // -------------------------------------------------------
            {
              std::lock_guard<std::mutex> lock(mutex);
              opTrace(world->handle,
                      TILE_SIZE*TILE_SIZE,
                      (OPRay*)&rays[0],
                      (OPHit*)&isecs[0]);
            }
            // -------------------------------------------------------
            // shade
            // -------------------------------------------------------
            int numActive = 0;
            array2D::
              for_each
              // parallel_for
              (vec2i(TILE_SIZE),[&](const vec2i &l_pixelID){
                const int lID = array2D::linear(l_pixelID,vec2i(TILE_SIZE));
                if (!active[lID])
                  return;
                
                if (isecs[lID].primID < 0) {
                  if (depth == 0) {
                    const vec2i pixelID = tileID*TILE_SIZE+l_pixelID;
                    weight[lID] = missColor(normalize(rays[lID].direction));
                  }
                  active[lID] = false;
                  rays[lID].tmin = 0.f;
                  rays[lID].tmax = 0.f;
                  return;
                }

                ScatterEvent scatter;
                vec3f &org = (vec3f&)rays[lID].origin;
                vec3f &dir = (vec3f&)rays[lID].direction;
                scatter.inDir = normalize(dir);
                scatter.P = org + isecs[lID].t * dir;
                const World::Triangle &tri = world->triangles[isecs[lID].primID];
                const World::TriangleData &triData = world->triangleData[isecs[lID].primID];
#if HAVE_SHADING_NORMALS
                const float u = isecs[lID].u;
                const float v = isecs[lID].v;
                scatter.N
                  = normalize((1.f-u-v) * triData.Na
                              + u       * triData.Nb
                              + v       * triData.Nc);
#else
                scatter.N = normalize(cross(tri.B-tri.A,tri.C-tri.A));
#endif
                if (triData.material->scatter(scatter,random)) {
                  numActive++;
                  org = scatter.out_org;
                  dir = scatter.out_dir;
                  rays[lID].tmin = 1e-3f;
                  rays[lID].tmax = std::numeric_limits<float>::infinity();
                  weight[lID] *= scatter.attenuation;
                } else {
                  active[lID] = false;
                  rays[lID].tmin = 0.f;
                  rays[lID].tmax = 0.f;
                }
              });
            if (numActive == 0) break;
          }

          // -------------------------------------------------------
          // path weight known, accumulate path
          // -------------------------------------------------------
          for (int i=0;i<TILE_SIZE*TILE_SIZE;i++)
            if (valid[i]) {
              color[i] += weight[i];
            }
        }
        // -------------------------------------------------------
        // all samples done, write frame buffer
        // -------------------------------------------------------
        array2D::
          for_each
          // parallel_for
          (vec2i(TILE_SIZE),[&](const vec2i &l_pixelID){
            const int lID = array2D::linear(l_pixelID,vec2i(TILE_SIZE));
            if (!valid[lID])
              return;
            const vec2i pixelID = tileID*TILE_SIZE+l_pixelID;
            fb->pixels[array2D::linear(pixelID,fb->size)]
              = color[lID] * (1.f / numSamplesPerPixel);
          });
      });
  }

}


using namespace samples;

  int main(int ac, const char **av)
  {
    // owl::prime::init(ac,av);

    int Nx = 800, Ny = 600;
    int sphereRecDepth = 5;
    int spp = 8;
    int maxPathLength = 3;
    OPContextType contextType = OP_CONTEXT_DEFAULT;
    for (int i=1;i<ac;i++) {
      const std::string arg = av[i];
      if (arg == "-fast" || arg == "-lq") {
        Nx = 400;
        Ny = 300;
        sphereRecDepth = 2;
        spp  = 1;
      } else if (arg == "--final" || arg == "-hq") {
        Nx = 2*800;
        Ny = 2*600;
        sphereRecDepth = 6;
        spp  = 1024;
      } else if (arg == "--size") {
        Nx = std::stoi(av[++i]);
        Ny = std::stoi(av[++i]);
      } else if (arg == "--tess") {
        sphereRecDepth = std::stoi(av[++i]);;
      } else if (arg == "--rec" || arg == "--bounces" || arg == "--max-path-length") {
        maxPathLength = std::stoi(av[++i]);;
      } else if (arg == "--spp" || arg == "-spp") {
        spp = std::stoi(av[++i]);
      } else if (arg == "--cpu" || arg == "-cpu") {
        contextType = OP_CONTEXT_HOST_FORCE_CPU;        
      } else throw std::runtime_error("unknown arg " +arg);
    }
    
    World::sphereRecDepth = sphereRecDepth;
    
    // create - and set - the camera
    const vec3f lookfrom(13, 2, 3);
    const vec3f lookat(0, 0, 0);
    std::shared_ptr<Camera>
      camera = std::make_shared<Camera>(lookfrom,
                                        lookat,
                                        /* up */ vec3f(0.f, 1.f, 0.f),
                                        /* fovy, in degrees */ 20.f,
                                        /* aspect */ float(Nx) / float(Ny),
                                        /* aperture */ 0.1f,
                                        /* dist to focus: */ 10.f);
    
    // create a frame buffer
    std::shared_ptr<FrameBuffer> fb = std::make_shared<FrameBuffer>(vec2i(Nx, Ny));
    std::shared_ptr<World> world = createScene(contextType);
    world->finalize();
    
    // render the frame (and time it)
    auto t0 = std::chrono::system_clock::now();
    renderFrame(world,camera,fb,
                spp,maxPathLength);
    auto t1 = std::chrono::system_clock::now();
    std::cout << "done rendering, which took "
              << std::setprecision(4) << std::chrono::duration<double>(t1-t0).count()
              << " seconds (for " << spp
              << " paths per pixel)" << std::endl;
       
    savePNG("finalChapter.png",*fb);

    // ... done.
    return 0;
  }

