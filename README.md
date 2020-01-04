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


<!--- ------------------------------------------------------- -->
API Abstraction Level and Directory Structure
=============================================

One of the key insights of early exprimentation with OWL was that a
single node graph API on top of OptiX 7 is a major
undertaking. Consequently, owl actually aims for two independent but
stacked API layers: one as-minimalistic-as-possible low-level API
layer (`ll-owl`) that does intentionally not deal with nodes,
variables, lifetime-handling, etc; and the actual node graph
(`owl-ng`) that then builds on this.

As of the time of this writing the ll layer is significantly more
fleshed out than the node graph layer. Though there are clearly
missing pieces even in the ll layer I do already have several of my
originally Optix 6 base research sandboxes ported over to owl-ll; the
node graph layer can - since 0.5.3 - also support all that is requires
for the "Ray Tracing in One Weekend" example (see
`samples/ng/s05-rtow`), but the other ll examples are not yet ported,
and will surely be missing a few bits and pieces.

To clearly separate the two API layers the project's directory
structure is organized into separate "ll/" and "ng/" directory layers:

- `owl/`: The Optix Wrappers *library*
  - `owl/ll/`: the owl *low-level* API layer
    - `owl/ll/include`: public API headers for the `llowl` shared library
    - `owl/ll/<other>`: implementation of that api layer
  - `owl/ng/`: the owl *node graph* API layer (build on top of owl/ll)
    - `owl/ng/include`: public API headers for the `owl-ng` shared library
    - `owl/ng/<other>`: implementation of that api layer

- `samples/`: Samples/Tutorials/TestCases for OWL
  - `samples/ll/`: samples for the ll layer
  - `samples/ng/`: samples for the ng layer (some ll samples not yet ported over)

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
    - Requires: Visual Studio (2019 works), cmake
	- Build: Use CMake-GUI to build visual studio project, then use VS to build

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
