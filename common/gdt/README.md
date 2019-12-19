# GDT - GPU Developer Tools

This entire directory tree was imported from an originally standalone
submodule (gitlab.com/ingowald/gdt) that provided a set of shared
infrastructure (glut based viewer, stb image reader/writer,
vector/math library, etc) across a range of different GPU/CPU
projects.

While initially used as a submodule this was eventually included as a
direct copy, in order to no longer have a submodule requirement that
will otherwise only confuse non-expert git users. 

To avoid naming conflicts with some applications potentially using the
original gdt project all namespaces have been renamed from ::gdt to
::owl::common; however, because of this project's origianlly wider
scope these directories may still contains "leftovers" that are not
currently needed, and may require cleanup.


