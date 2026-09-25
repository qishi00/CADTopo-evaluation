"""CAD-TopoBench topology reference implementation (v1.0, frozen).

THE single canonical chain for every topology label in the benchmark
(GT, generated, top-k candidates — no exceptions):

    DeepCAD quantized vec (52x17 int)
      -> cadquery solid           (cad_topobench.geometry.vec_to_solid)
      -> tessellate (surface mesh)
      -> trimesh weld (merge duplicate vertices across faces)
      -> GUDHI surface homology   (persistence_dim_max=True)
      -> solid Betti numbers [b0, b1, b2]

Mapping from welded-surface homology to solid homology
(valid for orientable closed B-Rep boundaries):

    b0 = #Solids
    b1 = (surface beta_1 total) / 2        # through-holes / handles
    b2 = (surface beta_2) - #Solids        # enclosed voids / cavities

Why not naive V-E+F: OCCT's canonical representation of cylindrical /
circular features uses loop edges and seam edges, so the raw B-Rep
counts do not form a regular CW complex and the naive Euler formula
returns wrong values (e.g. a washer reports chi=2). The welded surface
mesh is the robust object; the mapping above recovers solid homology.

Any other topology counter (e.g. counting face inner wires) is a
non-reference implementation and MUST NOT be used for benchmark labels.
"""
import numpy as np
import trimesh
import gudhi

REFERENCE_VERSION = "1.0"
TESSELLATION_TOL = 1e-3


RETRY_TOL = 1e-4


def _surface_complex(solid, tol):
    """tessellate + weld -> (trimesh, surf_betti) or (None, error_str)."""
    try:
        verts, tris = solid.tessellate(tol)
    except Exception as e:
        return None, f"tessellate_failed:{type(e).__name__}"
    vs = np.array([[v.x, v.y, v.z] for v in verts], dtype=np.float64)
    ts = np.asarray(tris, dtype=np.int64)
    if len(vs) == 0 or len(ts) == 0:
        return None, "empty_mesh"
    mesh = trimesh.Trimesh(vs, ts, process=True)  # weld duplicate vertices
    mesh.remove_unreferenced_vertices()
    st = gudhi.SimplexTree()
    for t in mesh.faces:
        st.insert([int(t[0]), int(t[1]), int(t[2])])
    for i in range(len(mesh.vertices)):
        st.insert([i])
    st.compute_persistence(persistence_dim_max=True)
    sb = st.betti_numbers()
    while len(sb) < 3:
        sb.append(0)
    return mesh, [int(x) for x in sb[:3]]


def solid_betti(solid, tol=TESSELLATION_TOL):
    """cadquery solid -> solid Betti numbers via the reference chain.

    Policy: a label is emitted ONLY for watertight welded surface meshes
    (every edge shared by exactly 2 faces). If the default tessellation
    is not watertight, retry once with finer tolerance; otherwise the
    sample is reported invalid (ok=False, error='non_watertight') and
    excluded from benchmark metrics (exclusion rate is reported).

    Returns dict:
        ok        bool
        betti     [b0, b1, b2] or None
        surf_betti welded-surface Betti numbers (diagnostic)
        watertight bool
        n_tris    int
        volume    float
        tol_used  tessellation tolerance that produced the label
        error     str or None
    """
    out = dict(ok=False, betti=None, surf_betti=None, watertight=None,
               n_tris=0, volume=None, tol_used=None, error=None)
    try:
        out["volume"] = float(solid.Volume())
    except Exception:
        pass

    # Zero-volume gate (hardening, 2026-07-28): a degenerate solid with no
    # enclosed volume can never yield a meaningful topology label, so reject
    # it as F0-A before tessellation instead of relying on the downstream
    # empty_mesh path alone. Provably inactive on all frozen v1.0 labels
    # (min labeled volume = 1.5e-6 > 0 across GT 7,731 + Gen 7,331 labels),
    # i.e. this gate changes no released label; it only hardens future runs.
    if out["volume"] is not None and out["volume"] <= 0.0:
        out["error"] = "zero_volume"
        return out

    mesh = None
    sb = None
    for t in (tol, RETRY_TOL):
        mesh, sb = _surface_complex(solid, t)
        if mesh is None:
            out["error"] = sb
            return out
        out["tol_used"] = t
        if mesh.is_watertight:
            break
    out["surf_betti"] = sb
    out["n_tris"] = int(len(mesh.faces))
    out["watertight"] = bool(mesh.is_watertight)
    if not out["watertight"]:
        out["error"] = "non_watertight"
        return out

    try:
        n_solids = max(1, len(solid.Solids()))
    except Exception:
        n_solids = 1
    b0 = int(n_solids)
    b1 = int(sb[1] // 2)
    b2 = max(0, int(sb[2]) - n_solids)
    out["betti"] = [b0, b1, b2]
    out["ok"] = True
    return out


def vec_betti(vec, tol=TESSELLATION_TOL):
    """DeepCAD quantized vec -> solid Betti numbers via the reference chain.

    Requires the DeepCAD repo at external/DeepCAD (provides cadlib); the
    import is deferred so that STEP labeling (solid_betti) does not need it.
    """
    from .geometry import vec_to_solid
    solid = vec_to_solid(np.asarray(vec))
    if solid is None:
        return dict(ok=False, betti=None, surf_betti=None, watertight=None,
                    n_tris=0, volume=None, tol_used=None, error="build_failed")
    return solid_betti(solid, tol=tol)
