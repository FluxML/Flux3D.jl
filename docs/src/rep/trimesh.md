```@meta
CurrentModule = Flux3D
```

# TriMesh Structure

## Heterogenous Batching

Since triangle mesh are heterogenous in nature, TriMesh follows heterogenous batching which include three different represenatation. Assuming, `m = TriMesh([v1, v2], [f1, f2])` where size of `v1` is `3 x V1` and `v2` is `3 x V2`.

* List      - list of arrays in the batch. Like, `get_verts_list(m)` returns list of arrays of verts in the batch, `[v1, v2]`. 
* Packed    - Packed reperesentation concatenates the list of arrays in the batch into single packed array. Like, `get_verts_packed(m)` returns an array of size `3 x (V1+V2)`.
* Padded    - Padded representation stack up the list of arrays after paddding extra values. Like, `get_verts_padded(m)` returns an array of size `3 x max(V1,V2) x 2` and extra values are filled with `0`.

---

## List of the TriMesh constructors and all TriMesh functions.

```@index
Pages   = ["trimesh.md"]
```

---

## TriMesh Constructors

```@docs
TriMesh
```

---

## Loading and Saving from file

```@docs
load_trimesh
save_trimesh
```

---

## Support for GeometryBasics.jl

TriMesh structure can be converted to and from GeometryBasics.jl Structure.

```@docs
GBMesh
gbmeshes
```

---

## Vertices and Faces

```@docs
get_verts_list
get_verts_packed
get_verts_padded
get_faces_list
get_faces_packed
get_faces_padded
```

---

## Edges, Laplacian Matrix and other adjacency info

```@docs
get_edges_packed
get_edges_to_key
get_faces_to_edges_packed
get_laplacian_packed
get_laplacian_sparse
```

---

## Normals for verts/faces and faces areas

```@docs
compute_verts_normals_list
compute_verts_normals_packed
compute_verts_normals_padded
compute_faces_normals_list
compute_faces_normals_packed
compute_faces_normals_padded
compute_faces_areas_list
compute_faces_areas_packed
compute_faces_areas_padded
```
