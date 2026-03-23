MUSICA SCRIP file and MESH file are from Louisa Emmons.

_cesm_se_mm_Copy1.py is modified reader file for unstructured model data, like MUSICA. It will use the MESH file information to regrid MUSICA to 0.1 degree grid.

XRegrid_MUSICA_to_MPAS.ipynb is the file trying to convert SCRIP information to MESH. Currently, it is functional but has some minor issue and leads to double-count for vertices.

XRegrid_MUSICA_to_tenth_degree.ipynb provides an example for how to use the XRegrid package and mesh grid information to regird a MUSICA file to 0.1 degree.
