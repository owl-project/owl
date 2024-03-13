#!/usr/bin/env python3

import py_owl
from py_owl import context_create
from py_owl import save_png_rgba8

import math;

import numpy as np

NUM_VERTICES = 8
vertices = np.array([
    -1.,-1.,-1.,
    +1.,-1.,-1.,
    -1.,+1.,-1.,
    +1.,+1.,-1.,
    -1.,-1.,+1.,
    +1.,-1.,+1.,
    -1.,+1.,+1.,
    +1.,+1.,+1.
    ], dtype='float32')

NUM_INDICES = 12
indices = np.array([
     0,1,3 ,  2,3,0 ,
     5,7,6 ,  5,6,4 ,
     0,4,5 ,  0,5,1 ,
     2,3,7 ,  2,7,6 ,
     1,5,7 ,  1,7,3 ,
     4,0,2 ,  4,2,6
     ], dtype='int32')


outFileName = "pyowl-s01-simpleTriangles.png"
fb_size = ( 800, 600 )
look_from = ( -4., -3., -2. )
look_at = ( 0., 0., 0. )
look_up = ( 0., 1., 0. )
cos_fovy = 0.66

def sub( a, b ):
    return ( a[0]-b[0], a[1]-b[1], a[2]-b[2] )

def scale_f( f, vec ):
    result = ( f*vec[0], f*vec[1], f*vec[2] )
    return result

def dot( a, b ):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def cross( a, b ):
    return (
        a[1]*b[2]-a[2]*b[1],
        a[2]*b[0]-a[0]*b[2],
        a[0]*b[1]-a[1]*b[0]
        )

def normalize( vec ):
    length = math.sqrt(dot(vec,vec))
    scale = 1 / length
    result = scale_f( scale, vec)
    return result
        
def main():
    print("#sample: creating context")
    owl = py_owl.context_create()
    module = owl.module_from_file("cmd01_simpleTriangles.ptx")
    print("#sample: creating geom type")
    gt = owl.geom_type_create(owl.GEOM_TRIANGLES,module,"TrianglesGeomData")

    gt.set_closest_hit(0,module,"TriangleMesh")
    
    print("#sample: creating vertex, index, and frame buffer(s) ...")
    vertex_buffer = owl.device_buffer_create(owl.FLOAT3,vertices)
    index_buffer = owl.device_buffer_create(owl.INT3,indices)
    frame_buffer = owl.host_pinned_buffer_create(owl.INT,fb_size[0]*fb_size[1])

    print("#sample: building geometries ...")
    triangles_geom = owl.geom_create(gt)
    
    #owlTrianglesSetVertices(trianglesGeom,vertexBuffer,NUM_VERTICES,sizeof(vec3f),0);
    triangles_geom.set_vertices(vertex_buffer)
    
    #owlTrianglesSetIndices(trianglesGeom,indexBuffer,NUM_INDICES,sizeof(vec3i),0);
    triangles_geom.set_indices(index_buffer)
    
    triangles_geom.set_buffer("vertex",vertex_buffer);
    triangles_geom.set_buffer("index",index_buffer);
    triangles_geom.set_3f("color",(0,1,0))
    
    # ------------------------------------------------------------------
    # the group/accel for that mesh
    # ------------------------------------------------------------------
    triangles_group = owl.triangles_geom_group_create( [ triangles_geom ] )
    triangles_group.build_accel()
    
    world = owl.instance_group_create( [ triangles_group ] )
    world.build_accel()

    # ##################################################################
    # set miss and raygen program required for SBT
    # ##################################################################
    
    # -------------------------------------------------------
    # set up miss prog 
    # -------------------------------------------------------
    miss_prog = owl.miss_prog_create(module,"MissProgData","miss")
    miss_prog.set_3f("color0",(.8,0.,0.))
    miss_prog.set_3f("color1",(.8,.8,.8))

    # -------------------------------------------------------
    # set up ray gen program
    # -------------------------------------------------------
    ray_gen = owl.ray_gen_create(module,"RayGenData","simpleRayGen")
    
    # ----------- compute variable values  ------------------
    camera_pos = look_from
    camera_d00 = normalize(sub(look_at, look_from))
    aspect = fb_size[0] / float(fb_size[1])
    camera_ddu = scale_f(cos_fovy * aspect, normalize(cross(camera_d00,look_up)))
    camera_ddv = scale_f(cos_fovy, normalize(cross(camera_ddu,camera_d00)))
    camera_d00 = sub(camera_d00, scale_f(0.5, camera_ddu))
    camera_d00 = sub(camera_d00, scale_f(0.5, camera_ddv))
    
    #// ----------- set variables  ----------------------------
    ray_gen.set_buffer("fbPtr",        frame_buffer);
    ray_gen.set_2i    ("fbSize",       fb_size);
    ray_gen.set_group ("world",        world);
    ray_gen.set_3f    ("camera_pos",   camera_pos);
    ray_gen.set_3f    ("camera_dir_00",camera_d00);
    ray_gen.set_3f    ("camera_dir_du",camera_ddu);
    ray_gen.set_3f    ("camera_dir_dv",camera_ddv);
  
    #// ##################################################################
    #// build *SBT* required to trace the groups
    #// ##################################################################
    owl.build_programs()
    owl.build_pipeline()
    owl.build_SBT()
    
    #// ##################################################################
    #// now that everything is ready: launch it ....
    #// ##################################################################
  
    ray_gen.launch(fb_size);
  
    fb = save_png_rgba8(frame_buffer,fb_size,outFileName);
    print("#sample: written rendered frame buffer to file "+outFileName)
    
    # ##################################################################
    # and finally, clean up
    # ##################################################################
    
    owl.context_destroy()

main()

