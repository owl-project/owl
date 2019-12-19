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

- For a brief (visual) overview over latest samples: http://owl-project.github.io/Samples.html


<!--- ------------------------------------------------------- -->
API Abstraction Level and Directory Structure
=============================================

One of the key insights of early exprimentation with OWL was that a
single node graph API on top of OptiX 7 is a major
undertaking. Consequently, owl actually aims for two independent but
stacked API layers: one as-minimalistic-as-possible low-level API
layer (`ll-owl`) that does not yet deal with nodes, variables,
lifetime-handling, etc; and the actual node graph (`owl-ng`) that then
builds on this.

As of the time of this writing the node graph layer is not yet
functional, and thus not yet included; the ll-layer is still rather
basic, but at least functional enough to reproduce previous OptiX
samples such as the "Ray Tracing in One Weekend in OptiX" example.

To eventually accomodate two separate API layers the project's
directory structure is already organized into separate "ll/" and "ng/"
directory layers (though as the latter isn't functional yet it is
still missing in master an devel branches):

- `owl/`: The Optix Wrappers *library*
  - `owl/ll/`: the owl *low-level* API layer
  - `owl/ng/`: the owl *node graph* API layer (currently disabled because not yet functional)

- `samples/`: Samples/Tutorials/TestCases for OWL
  - `samples/ll/`: samples for the ll layer
  - `samples/ng/`: samples for the ng layer (currently disabled because not yet functional)

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
