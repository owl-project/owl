owl - "Optix Wrappers Library" on top of Optix 7
================================================


Explanation of Directory Structure
==================================

When looking at the directory structure you'll likely stumble over the
fact that instead of owl/xyz there's always owl/ll/xyz. This is
because *eventually* there is supposed to be another "node graph"
layer on top of the low-level layer, so the *final* directory
structure is supposed to look like this:

- `owl/`: The Optix Wrappers *library*
  - `owl/ll/`: the owl *low-level* API layer
  - `owl/ng/`: the owl *node graph* API layer (currently disabled because not yet working)

- `samples/`: Samples/Tutorials/TestCases for OWL
  - `samples/ll/`: samples for the ll layer
  - `samples/ng/`: samples for the ng layer (currently disabled because not yet working)

TODO
====

- CI for windows

- api function naming cleanup. Currently have 'createUserGeomGroup'
  but 'instanceGroupCreate'. Make all use the latter format, so all
  functions start with the name of the type affected (similar to
  InstnaceGroup::create)

- more examples

  - optix 6 samples
  - optix 6 advanced samples
  - pbrtParser(?)
  - optix prime like + cuda interop
  - vishal spatial queries

Revision History
================

v0.4.x - Instances
------------------

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
