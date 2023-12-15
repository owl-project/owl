# ======================================================================== #
# Copyright 2020-2021 Ingo Wald                                            #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http:#www.apache.org/licenses/LICENSE-2.0                            #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
# ======================================================================== #

# This sample builds on Pete Shirley's original 'ray tracing on a weekend'
# (RTOW) sample, as ported over to OWL in owl/samples/offline/s05-rtow. The
# correpsonding device code for this sample is in `deviceCode.cu`, and except
# for a few additional delcarations at beginning and end (to make the python
# host code 'recognize' the device-side data types) is identical to the regular
# OWL version of this sample.
#
# The overall process for how such a pyOWL program works is that all RTX device
# programs are in the CUDA `deviceCode.cu` file in exactlythe same way as for
# non-python host code. This code gets compiles with CUDA (nvcc) to produce a
# PTX file that compiles the pre-compiled device code for this file. On the
# python host code side, this PTX file gets loaded as a "Module" (see below),
# from which point the python user can then build geometries that use the
# programs and types contained in this module; or assign values to parameters
# of those types, etc.

#!/usr/bin/env python3

import py_owl
from py_owl import context_create
from py_owl import save_png_rgba8

import math;
import random;
import numpy as np
 
def rng():
    return random.random()

class Scene:
    lambertian_spheres = []
    dielectric_spheres = []
    metal_spheres = []

class Dielectric:
    def __init__(self,ref_idx):
        self.ref_idx = ref_idx
    def __str__(self):
        return "Dielectric{ref_idx="+str(self.ref_idx)+"}"
    ref_idx = 1.5

class Lambertian:
    def __init__(self,albedo):
        self.albedo = albedo
    def __str__(self):
        return "Lambertian{albedo="+str(self.albedo)+"}"
    albedo = (.5, .5, .5)

class Metal:
    def __init__(self,albedo, fuzz):
        self.albedo = albedo
        self.fuzz = fuzz
    def __str__(self):
        return "Metal{albedo="+str(self.albedo)+",fuzz="+str(self.fuzz)+"}"
    albedo = (.5, .5, .5)
    fuzz = .5
    
class Sphere:
    def __init__(self,center,radius,material):
        self.center = center
        self.radius = radius
        self.material = material
    def __str__(self):
        return "Sphere(center="+str(self.center)+",radius="+str(self.radius)+",material="+str(self.material)+")"
    center   = (0.,0.,0.)
    radius   = 1.
    material = None

def sub(a,b):
    return (a[0]-b[0],a[1]-b[1],a[2]-b[2])

def scale(f,v):
    return (f*v[0],f*v[1],f*v[2])

def dot(a,b):
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]

def length(a):
    return math.sqrt(dot(a,a))

def normalize(a):
    return scale(1/length(a),a)

def cross(a,b):
    return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])

def create_scene():
    print("#s05-rtow: creating scene")
    scene = Scene()
    mat = Lambertian((.5,.5,.5))
    center = (0,-1000,-1)
    scene.lambertian_spheres.append(Sphere(center,1000,mat))

    for a in range(-11,11):
        for b in range(-11,11):
            choose_mat = rng()
            center = (a+rng(), .2, b+rng() )
            if choose_mat < .8:
                mat = Lambertian((rng()*rng(),rng()*rng(),rng()*rng()))
                scene.lambertian_spheres.append(Sphere(center,.2,mat))
            elif choose_mat < .95:
                albedo = (.5*(1+rng()),.5*(1+rng()),.5*(1+rng()))
                mat = Metal(albedo,0.5*rng())
                scene.metal_spheres.append(Sphere(center,.2,mat))
            else:
                mat = Dielectric(1.5)
                scene.dielectric_spheres.append(Sphere(center,.2,mat))
    scene.dielectric_spheres.append(Sphere((0,1,0),1,Dielectric(1.5)))
    scene.lambertian_spheres.append(Sphere((-4,1,0),1,Lambertian((.4,.2,.1))))
    scene.metal_spheres.append(Sphere((4,1,0),1,Metal((.7,.6,.5),0)))
    return scene

def main():
    random.seed(290374)
    print("#s05-rtow: starting up")
    
    # -------------------------------------------------------
    scene = create_scene()
    print("#s05-rtow: created scene:")
    print("#s05-rtow: - num lambertian spheres : "+str(len(scene.lambertian_spheres)))
    print("#s05-rtow: - num metal spheres      : "+str(len(scene.metal_spheres)))
    print("#s05-rtow: - num dielectric spheres : "+str(len(scene.dielectric_spheres)))

    # -------------------------------------------------------
    print("#sample: creating context")
    owl = py_owl.context_create()
    print("#sample: creating module")
    module = owl.module_from_file("cmd05_rtow.ptx")
    print("#sample: creating geom type for 'lambertian' spheres")
    lambertian_sphere_gt = owl.geom_type_create(owl.GEOM_USER,module,"LambertianSphere")
    lambertian_sphere_gt.set_closest_hit(0,module,"LambertianSphere")
    lambertian_sphere_gt.set_bounds_prog(module,"LambertianSphere")
    lambertian_sphere_gt.set_intersect_prog(0,module,"LambertianSphere")

    print("#sample: creating geom type for 'metal' spheres")
    metal_sphere_gt = owl.geom_type_create(owl.GEOM_USER,module,"MetalSphere")
    metal_sphere_gt.set_closest_hit(0,module,"MetalSphere")
    metal_sphere_gt.set_bounds_prog(module,"MetalSphere")
    metal_sphere_gt.set_intersect_prog(0,module,"MetalSphere")
    
    print("#sample: creating geom type for 'dielectric' spheres")
    dielectric_sphere_gt = owl.geom_type_create(owl.GEOM_USER,module,"DielectricSphere")
    dielectric_sphere_gt.set_closest_hit(0,module,"DielectricSphere")
    dielectric_sphere_gt.set_bounds_prog(module,"DielectricSphere")
    dielectric_sphere_gt.set_intersect_prog(0,module,"DielectricSphere")
    
    # build programs so we have the bounds progs read for geometry creation
    print("#sample: building bounds programs")
    owl.build_programs()

    sphere_geoms = []
    print("#sample: creating all sphere geoms")
    for sphere in scene.lambertian_spheres:
        geom = owl.geom_create(lambertian_sphere_gt)
        geom.set_prim_count(1)
        geom.set_3f("sphere_center",sphere.center)
        geom.set_1f("sphere_radius",sphere.radius)
        geom.set_3f("material_albedo",sphere.material.albedo)
        sphere_geoms.append(geom)
    for sphere in scene.metal_spheres:
        geom = owl.geom_create(metal_sphere_gt)
        geom.set_prim_count(1)
        geom.set_3f("sphere_center",sphere.center)
        geom.set_1f("sphere_radius",sphere.radius)
        geom.set_3f("material_albedo",sphere.material.albedo)
        geom.set_1f("material_fuzz",sphere.material.fuzz)
        sphere_geoms.append(geom)
    for sphere in scene.dielectric_spheres:
        geom = owl.geom_create(dielectric_sphere_gt)
        geom.set_prim_count(1)
        geom.set_3f("sphere_center",sphere.center)
        geom.set_1f("sphere_radius",sphere.radius)
        geom.set_1f("material_ref_idx",sphere.material.ref_idx)
        sphere_geoms.append(geom)
    
    # ------------------------------------------------------------------
    # the group/accel for all those spheres
    # ------------------------------------------------------------------
    print("#sample: building spheres user group")
    spheres_group = owl.user_geom_group_create( sphere_geoms )
    print("#sample: building spheres accel")
    spheres_group.build_accel()
    
    world = owl.instance_group_create( [ spheres_group ] )
    world.build_accel()

    # -------------------------------------------------------
    # set up miss prog 
    # -------------------------------------------------------
    miss_prog = owl.miss_prog_create(module,"MissProgData","miss")

    # -------------------------------------------------------
    # set up ray gen program
    # -------------------------------------------------------
    ray_gen = owl.ray_gen_create(module,"RayGenData","rayGen")

    # hard-coded frame and camera settings for this sample:
    fb_size = (1600,800)
    look_from = (13, 2, 3)
    look_at = (0, 0, 0)
    look_up = (0,1,0)
    fovy = 20
    # compute camera model from those;
    vfov = fovy
    vup = look_up
    aspect = fb_size[0] / fb_size[1]
    theta = vfov * math.pi / 180
    half_height = math.tan(theta / 2)
    half_width = aspect * half_height
    focus_dist = 10
    origin = look_from
    w = normalize(sub(look_from,look_at))
    u = normalize(cross(vup, w))
    v = cross(w, u)
    lower_left_corner = origin
    lower_left_corner = sub(lower_left_corner,scale(half_width * focus_dist,u))
    lower_left_corner = sub(lower_left_corner,scale(half_height * focus_dist,v))
    lower_left_corner = sub(lower_left_corner,scale(focus_dist,w))
    horizontal = scale(2*half_width*focus_dist, u)
    vertical   = scale(2*half_height*focus_dist,v)
    frame_buffer = owl.host_pinned_buffer_create(owl.INT,fb_size[0]*fb_size[1])

    #// ----------- set variables  ----------------------------
    ray_gen.set_buffer("fb_ptr",        frame_buffer);
    ray_gen.set_2i    ("fb_size",       fb_size);
    ray_gen.set_group ("world",        world);
    ray_gen.set_3f    ("camera_horizontal",   horizontal);
    ray_gen.set_3f    ("camera_vertical",vertical);
    ray_gen.set_3f    ("camera_origin",origin);
    ray_gen.set_3f    ("camera_lower_left_corner",lower_left_corner);
  
    #// ##################################################################
    #// build *SBT* required to trace the groups
    #// ##################################################################
    owl.build_programs()
    owl.build_pipeline()
    owl.build_SBT()
    
    #// ##################################################################
    #// now that everything is ready: launch it ....
    #// ##################################################################
  
    print("launching ...\n");
    ray_gen.launch(fb_size);
  
    print("done with launch, writing picture ...")
    
    outFileName = "pyOWL-RTOW.png"
    fb = save_png_rgba8(frame_buffer,fb_size,outFileName);
    print("written rendered frame buffer to file "+outFileName)
    
    # ##################################################################
    # and finally, clean up
    # ##################################################################
    
    print("destroying devicegroup ...")
    owl.context_destroy()
  
    print("seems all went OK; app is done, this should be the last output ...")

    
main()

