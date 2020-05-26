owl - "Optix Wrappers Library" on top of Optix 7
================================================

<!--- ------------------------------------------------------- -->
What is OWL?
============

OWL is a OptiX 7 based library that aims at providing some of the
convenience of OptiX 6's Node Graph API on top of OptiX 7. This aims
at making it easier to port existing OptiX 6-style applications over
to OptiX 7, and, in particular, to make it easier to get started with
OptiX 7 even without having a full grasp of things like Shader Binding
Tables, Multi-GPU rendering, etc.

OWL is still in early stages, which means that it as yet lacks many of
the features that OptiX 6 offered, and in particular, that many things
are still changing rather rapidly; however, it already contains
several working mini-samples, so at the very least should allow to
serve as a "show-and-tell" example of how to set up a OptiX 7
pipeline, how to build acceleration structures, how to do things like
compaction, setting up an SBT, etc.

Key links:

- For latest code on github: https://github.com/owl-project/owl

- For a brief (visual) overview over latest samples:
  http://owl-project.github.io/Samples.html
  
- The OWL API is explained in part here: https://owl-project.github.io/ng-api-overview.html


<!--- ------------------------------------------------------- -->
API Abstraction Level and Directory Structure
=============================================

The OWL API aims at an abstraction that is similar to OptiX 6's node
graph. To do so it introduces logical types such as `OWLBuffer`s (to
handle buffers of data), `OWLGeomType`(s) (abstractions for geometry
*types* that define what closest hit, any hit, etc programs to run for
a given type), `OWLGeom`s (instances of geometry types, with specific
inputs for those programs to operate on), etc.

Internally OWL actually uses two different abstration layers that live
on top of each other: a "low-level" API (`llowl`) that is still rather
basic; and the actual "node graph" owl API (`owl`). Externally only
the public owl API is exposed.

Directory Structure of the Project:

- `owl/`: the root OWL *library* defined in this project
  - `owl/ll/`: *implementation* the low-level API layer
  - `owl/ng/`: *implementation* the node graph API
  - `owl/common/`: common helper classes
  - `include/owl/owl_host.h`: header for the public, C-linkage host-side API 
  - `include/owl/owl_device.h`: header to be used on device side, for isec/CH/AH programs.
  - `include/owl/common`: C++ math/vector classes required for the device-side API

- `samples/`: Samples/Tutorials/TestCases for OWL
  - `samples/`: tutorial-style samples that show how to create geometries, instances, etc
  - `samples/advanced`: more advanced sample containing a full OBJ model viewer
     (based on the siggraph 2019 OptiX course)

- `tests/`: Some internal test cases; only for CI/testing/debugging,
  should usually not be of much interest to users of the API

<!--- ------------------------------------------------------- -->
Supported Platforms
===================

General Requirements:
- OptiX 7 SDK
- CUDA 10 (preferably 10.2, but 10.1 is tested, too)
- a C++-11 capable compiler (regular gcc on CentOS and Linux should do, VS on Windows)

Per-OS Instructions:

- Ubuntu 18 & 19 (automatically tested on 18)
    - Requires: `sudo apt install cmake-curses-gui`
	- Build:
	```
	mkdir build
	cd build
	cmake ..
	make
	```
- CentOS 7:
    - Requires: `sudo yum install cmake3`
	- Build:
	```
	mkdir build
	cd build
	cmake3 ..
	make
	```
	(mind to use `cmake3`, not `cmake`, using the wrong one will mess up the build directory)
- Windows
    - Requires: Visual Studio (both 2017 and 2019 work), OptiX 7.0, cmake
	- Build: Use CMake-GUI to build Visual Studio project, then use VS to build
		- Specifics: source code path is ```...Gitlab/owl```, binaries ```...Gitlab/owl/build```, and after pushing the Configure button choose ```x64``` for the optional platform.
		- You may need to Configure twice.
		- If you get "OptiX headers (optix.h and friends) not found." then define OptiX_INCLUDE manually in CMake-gui by setting it to ```C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.0.0/include```

<!--- ------------------------------------------------------- -->
(Main) TODOs:
=============

- more samples/test cases

- add "Launch Params" functionality

- add c-style API on top of ll layer 
  - wrap `DeviceGroup*` into `LLOContext` type
  - wrap every `DeviceGroup::xyz(...)` function into a `lloXyz(context,...)` c-linkage API function
  - build into dll/so


<!--- ------------------------------------------------------- -->
Latest Progress/Revision History
================================

v0.8.x - Revamped build system, owl viewer, interative samples, and textures
----------------------------------------------------------------------

*v0.8.1*: first light of textures

- added basic infrastructure for textures: OWLTexture type,
  OWLTextureFormat and OWLTextureFilterMode enums, OWL_TEXTURE
  variable types, owlVariableSetTexture, owl<Type>SetTexture(),
  etc. Textures currently only working for OWL_TEXEL_FORMAT_RGBA8 and
  OWL_TEXTURE_FILETER_LINEAR.

- added int10-texturedTriangles that opens a window with a
  checkerboard-textured box.

*v0.8.0*: build system, glfw, and owl viewer

- cmake build system now cleaner, and can use entire owl dir as
  add_subdirectory in other projects; main owl variables
  (OWL_INCLUDES, OWL_VIEWER_LIBRARIES, etc) now get exported to
  whoever includes, thus allowing includee to use same build flags,
  proper set of libraries an dincludes, etc.
  
- existing (glut-based) viewerWidget got replaced with glfw based
  OWLViewer class. New class has cleaner setup code, and no longer
  requires installing glut binaries for windows build
  
- build system picks up glfw where available, and otherwise builds
  glfw from source (full glfw source in samples/3rdParty)

- owlViewer now handles frame buffer resize and display internally (no
  longer app's job to do that), and does so with cuda/gl resource
  sharing using managed mem frame buffer. App still writes render() method,
  but simply writes final pixels to viewer-handles frame buffer.
  
- added first two interactive samples, using owlViewer

- changed samples/advaned/optixCourse to use owlViewer - glut now
  completely eliminated from owl, and all samples use same viewer.


v0.7.x - Unifiction of ng and ll APIs into one single owl API
----------------------------------------------------------------------

*v0.7.4*: major cleanups of "low-level" and "api" layer abstractions

- 'll' and 'ng' layers mostly merged, at least from the API layer; led
  to significant reduction in duplicate code.
  
- merged in PRs to enable TBB on windows, and to add cuda grphics resource buffer

*v0.7.3*: performance "guiding"

- OWL no longer allows for tracing directly into BLASes ... this is
  highly discouraged in the driver, so better to just disallow it.

*v0.7.3*: bug hotfix

- hotfix for bug introduced when auto-freeing of device memory, which
  accidentally freed instance BVH whiel still in use.
  
- various windows fixes; in particular removing tbb by default (windows only)

*v0.7.2*: various feature extensions and bug fixes

- lots of additional use in owl prime, m-owl-ana, distributed renderer, etc.

- fixed memory leak in instance builder

- added multi-device sample (`s07-rtow-multiGPU`), including
  `samples/s07-rtow-multiGPU/README.md` with brief notes on how to do
  multi-GPU in owl

- added a lot more documentation to api functions (though much is still missing)

- added ManagedMemory buffer type

- added several sanity checks for group sizes, traversable graph
  depth, etc (checking w/ optix limits before trying to build)

- added AnyHit shader support

*v0.7.1*: bugfix release.

- added variable plumbing for missing unsigned and 64-bit types 

- fixed race condition in creating/using api handles that caused
  random crashes when setting launch params from different threads

*v0.7.0*: merged ng and ll APIs into one single API

- now have a single owl library, a single header file, etc

- eliminated all old ll/ samples (they only confused users)

- sierpinski, rtow, and rtow-mixedGeom samples now in owl API

- eliminated compaction in user geom and instance groups (doesn't
  help, anyway, and now have lower peak memory)
  
- camke now defines (and all samples etc use) cmake variables for
  `OWL_INCLUDES` and `OWL_LIBRARIES`

- fixes for TBB; TBB now gets detected more automatically, and used
  if found, with fallback to serial implementation if not

- added array3D, various cleanups and exntensions to owl/common

- various bugfixes and sanity/range checks throughout


v0.6.x - Buffer updates, launch params, first interactive example, ...
----------------------------------------------------------------------

*v0.6.1*: cleanup/flesh-out of instance transform API

- added creation of instance groups to ng/owl API

- added ability to use different matrix memory layouts (row major, column major)

- on ll layer: axed 'transform' parameter in `InstanceGroupSetChild`;
  matrices should now get passed to dedicated
  `InstanceGroupSetTransform` function.

- fixed various bugs related to instancing

*v0.6.0*: Buffer updates, launch params, first interactive example, ...

- Added OWL-port of github.com/ingowald/optix7course siggraph course
  notes sample as a first 'advanced' sample that allows for
  interactive fly-throughs trhough a "real" model. See original course
  notes for instructions on how to use. 

Process of adding this sample also required, among others, the following
feature changes/additions

- added support for resizing (`owlBufferResize`), destroying
  (`owlBufferDestroy`), and uploading data to (`owlBufferUpload`)
  buffers.

- added concept of LaunchParams (with variables, similar to geoms),
  both for creating (`owlLaunchParamsCreate`) and launching with
  launch params (`owlParamLaunch2D`)

- added support for *asynchronous* launches, where multiple launches
  can be in flight in parallel, using different streams. Wrote
  experimental "owl prime" project to debug, debug, and test this;
  project allow, for example, highly threaded cpu-side shading with
  owl-based, async GPU offload of the ray tracing (this project is not
  yet included).
  
- worked on better interoperability between owl and CUDA-based host
  app; app can query buffer device pointers, add can add raw data (eg,
  cuda textures) to objects, app can query streams used for async
  launches (eg to add async cudamemcpys to that same stream, sync
  itself w/ owl, etc) and fixed includes to allow mixing cuda and
  optix code

- added ability to create user-type variables (to pass, e.g.,
  CUDA texture objects as parameters to meshes)

- rewritten interface for getting and setting variables using macros,
  all object and variable types now supported
  
- added support for more than one ray type (`owlContextSetRayTypeCount`)

- clamped down on verbosity of the logging - most logs now visible
  only in debug mode

- various minor bugfixes throughout the code


v0.5.x - First Public Release Cleanups
--------------------------------------

*v0.5.4*: First external windows-app

- various changes to cmake scripts, library names, and in partciualr
  owl/common/viewerWidget to remove roadblocks for windows apps using
  that infrastructure
  
- first external windows sandbox app (particle viewer) using owl/ng
  and owl/viewerWidget

*v0.5.3*: First *serious* node graph sample

- ported `ll05-rtow` sample to node graph api

- added bound program, user geom, user geom group, setprimcount and
  other missing functionality to node graph api

- `ng05-rtow` ported, working, and passing tests

*v0.5.2*: First (partial) node graph sample

- first working version of subset of node graph library (all that is
  required for 'firstTriangleMesh' example)

- `ng01-firstTriangleMesh` working

- significant renames and cleanups of owl/common (in particular, all
  'gdt::' and 'gdt/' merged into owl::common and owl/common)
  
- cleaned up owl/common/viewerWidget. Not used in owl itself (to avoid
  dependencies to glut etc), but now working successfully in first
  external test project

*v0.5.1*: First "c-api" version

- added public c-linkage api (in `include/owl/ll.h`)

- changed to build both static and dynamic/shared lib (tested working
  both linux and windows)

- ported all samples to this new api


*v0.5.0*: First public release

- first publicly accessible project on
  http://github.com/owl-project/owl
  
- major cleanups: "inlined" al the gdt submodule sources into
  owl/common to make owl external-dependency-fee. Feplaced gdt::
  namespace with owl::common:: to match.

v0.4.x - Instances
------------------

*v0.4.5*: `ll08-sierpinski` now uses path tracing

*v0.4.4*: multi-level instancing

- added new `DeviceGroup::setMaxInstancingDepth` that allows to set max
  instance depth and stack depth on pipeline.

- added `ll08-sierpinski` example that allows for testing user-supplied number
  of instance levels with a sierpinski pyramid (Thx Nate!)
  
*v0.4.3*: new api fcts to set transforms and children for instance groups

- added `instanceGroupSetChild` and `instanceGroupSetTransform`
- extended `ll07-groupOfGroups` by two test cases that set transforms

*v0.4.2*: bugfix - all samples working in multi-device again

*v0.4.1*: example `ll06-rtow-mixedGeometries.png` 
 working w/ manual sucessive traced into two different accels

*v0.4.0*: new way of building SBT now based on groups

- api change: allocated geom groups now have their program size
  set in geomTypeCreate(), miss and raygen programs have it set in 
  type rather than in sbt{raygen/miss}build (ie, program size now
  for all types set exactly once in type, then max size computed during
  sbt built)
  
- can handle more than one group; for non-0 group has to query
  geomGroupGetSbtOffset() and pass that value to trace
  
- new sbt structure no longer uses 'one entry per geom' (that unfortunately
  doesnt' work), but now builds sbt by iterating over all groups, and
  putting each groups' geom children in one block before putting
  next group. groups store the allcoated SBT offset for later use
  by instances

v0.3.x - User Geometries
------------------------

*v0.3.4*: bugfix: adding bounds prog broke bounds buffer variant. fixed.

*v0.3.4*: first 'serious' example: RTOW-finalChapter on OWL

- added `s05-rtow` example that runs Pete's "final chapter" example 
  (iterative version) on top of OWL, with multi-device, different material, etc.

*v0.3.3*: major bugfix in bounds program for geoms w/ more than 128 prims.

*v0.3.2*: added two explicit examples for uesr geom - one with
  host-generation of bounds passed thrugh buffer, and one with bounds
  program

*v0.3.1*: First draft of *device-side* user prim bounds generation

- added `groupBuildPrimitiveBounds` function that builds, for a
  user geom group, all the the primbounds required for the respective
  user geoms and prims in that group. The input for the user geoms' 
  bounding bxo functions is generated using same callback mechanism
  as sbt writing.

*v0.3.0*: First example of user geometry working

- can create user geometries through `createUserGeom`, and set
  type's isec program through `setGeomTypeIntersect`
- supports passing of new `userGeomSetBoundsBuffer` fct to pass user
  geoms through a buffer
- first example (8 sphere geometries, each with one sphere per geom)
  available as `s03-userGeometry`

v0.2.x
------

*v0.2.1*: multiple triangle meshes working
- multiple triangle meshes in same group debugged and working
- added `ll02-multipleTriangleGroups` sample that generates 8 boxes

*v0.2.0*: first triangle mesh with trace and SBT data working
- finalized `llTest` sample that ray traced image of one (tessellated) box

v0.1.x
------

- first version that does "some" sort of launch with mostly functional SBT

Contributors
============

- Ingo Wald
- Nate Morrical
- Eric Haines
