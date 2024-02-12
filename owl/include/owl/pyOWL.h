#define PYOWL_EXPORT_TYPE(Type)                                 \
  __constant__ int __owl_typeDecl__##Type = sizeof(Type);

/* 'export' a device variable in device code, such that host-side
   python bindings can 'find' it and write to it. First parameter is
   the (device-)type of that variable (e.g., OWL_FLOAT3, OWL_INT,
   etc); second one is the device type/struct that this varaible is a
   member of (eg, 'RayGen' if it's a variable of a ray-gen program
   whose data is stored in a struct of that name; third parameter is
   how we want to refer to this variable on the python side (eg,
   'camera_origin'), and last is he name of the member of the
   device-side struct that stores the actual variable (eg,
   'camera.origin') 
*/
#define PYOWL_EXPORT_VARIABLE(type,typeName,varName,var)                \
  __constant__ int __owl_varDeclOffset__##typeName##____##varName = (size_t)&((typeName*)0)->var; \
  __constant__ int __owl_varDeclType__##typeName##____##varName = type;


/*
Example: Suppose we have a device-side ray-gen program that uses the
following struct to store its data:

# pyOWL CUDA device code:
struct RayGen {
   struct {
     uint32_t *memory;
     uint2     size;
   } fb;
   ...
}

Then the device-side struct 'RayGen' would be declared as

# pyOWL CUDA device code:
    PYOWL_EXPORT_TYPE(RayGen)

and its variable RayGen::fb.memory and RayGen::fb.size could be exported as

# pyOWL CUDA device code:
    PYOWL_EXPORT_VARIABLE(OWL_BUFPTR,RayGen,fb_memory,fb.size)
    PYOWL_EXPORT_VARIABLE(OWL_BUFPTR,RayGen,fb_size,fb.size)

On the host side --- assuming the user has already created some python
variables `frame_buffer` and `fb_size` --- these can then be set as

# main user python code:
   ray_gen.set_2i("fb_size",fb_size)
   ray_gen.set_buffer("fb_memory",frame_buffer)
*/




