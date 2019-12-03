owl - "Optix Wrappers Library" on top of Optix 7
================================================


Directory Structure
===================

- `owl/`: The Optix Wrappers *library*
  - `owl/ll/`: the owl *low-level* API layer
  - `owl/ng/`: the owl *node graph* API layer (WIP)

- `samples/`: Samples/Tutorials/TestCases for OWL
  - `samples/ll/`: samples for the ll layer
  - `samples/ng/`: samples for the ng layer

Revision History
================

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
