"""Geometry backend for CAD-TopoBench.

DeepCAD command vector (60x17, quantized int) -> B-Rep solid -> point cloud -> Chamfer Distance.

Replicates the semantics of DeepCAD's cadlib.visualize.create_CAD (pythonOCC)
using cadquery 2.8 (OCP backend), including:
  - profile.denormalize(sketch_size)
  - sketch local->global frame transform (p_global = px*x_axis + py*y_axis + origin)
  - OneSide / Symmetric / TwoSides extent types
  - Fuse / Cut / Intersect booleans
"""
import sys

import numpy as np

from .paths import DEEPCAD_ROOT  # noqa: E402  (path via CAD_TOPOBENCH_ROOT)
if str(DEEPCAD_ROOT) not in sys.path:
    sys.path.insert(0, str(DEEPCAD_ROOT))

from cadlib.extrude import CADSequence  # noqa: E402
from cadlib.macro import EXTENT_TYPE  # noqa: E402
from cadquery import Wire, Edge, Vector, Solid, Face  # noqa: E402

import trimesh  # noqa: E402
from trimesh.sample import sample_surface  # noqa: E402

import torch  # noqa: E402


# ---------------------------------------------------------------- vec -> solid

def _to_global(point2d, sketch_plane):
    """DeepCAD point_local2global: p.x*x_axis + p.y*y_axis + origin."""
    p = point2d[0] * sketch_plane.x_axis + point2d[1] * sketch_plane.y_axis + sketch_plane.origin
    return Vector(float(p[0]), float(p[1]), float(p[2]))


def _curve_to_edge(curve, sketch_plane):
    cls = curve.__class__.__name__
    if cls == 'Line':
        if np.allclose(curve.start_point, curve.end_point):
            return None
        return Edge.makeLine(_to_global(curve.start_point, sketch_plane),
                             _to_global(curve.end_point, sketch_plane))
    if cls == 'Arc':
        return Edge.makeThreePointArc(_to_global(curve.start_point, sketch_plane),
                                      _to_global(curve.mid_point, sketch_plane),
                                      _to_global(curve.end_point, sketch_plane))
    if cls == 'Circle':
        center = _to_global(curve.center, sketch_plane)
        normal = Vector(*[float(v) for v in sketch_plane.normal])
        return Edge.makeCircle(abs(float(curve.radius)), center, normal)
    raise ValueError(f'unknown curve type: {cls}')


def _loop_to_wire(loop, sketch_plane):
    edges = []
    for curve in loop.children:
        e = _curve_to_edge(curve, sketch_plane)
        if e is not None:
            edges.append(e)
    if not edges:
        return None
    return Wire.assembleEdges(edges)


def _extrude_body(extrude_op):
    """Build one extruded body from a DeepCAD Extrude op (correct sketch frame)."""
    from copy import copy
    profile = copy(extrude_op.profile)
    profile.denormalize(extrude_op.sketch_size)

    plane = extrude_op.sketch_plane
    loops = [l for l in profile.children if l.children]
    if not loops:
        return None
    outer = _loop_to_wire(loops[0], plane)
    if outer is None:
        return None
    inners = [w for w in (_loop_to_wire(l, plane) for l in loops[1:]) if w is not None]

    normal = Vector(*[float(v) for v in plane.normal])
    ext_one = float(extrude_op.extent_one)
    if abs(ext_one) < 1e-9:
        return None
    body = Solid.extrudeLinear(outer, inners, normal.multiply(ext_one))

    etype = int(extrude_op.extent_type)
    if etype == EXTENT_TYPE.index('SymmetricFeatureExtentType'):
        body2 = Solid.extrudeLinear(outer, inners, normal.multiply(-ext_one))
        body = body.fuse(body2)
    elif etype == EXTENT_TYPE.index('TwoSidesFeatureExtentType'):
        ext_two = float(extrude_op.extent_two)
        if abs(ext_two) > 1e-9:
            body2 = Solid.extrudeLinear(outer, inners, normal.multiply(-ext_two))
            body = body.fuse(body2)
    return body


def vec_to_solid(vec):
    """DeepCAD quantized command vector -> solid, or None if any step fails."""
    try:
        cad_seq = CADSequence.from_vector(vec.astype(np.int64), is_numerical=True, n=256)
    except Exception:
        return None
    result = None
    for ext in cad_seq.seq:
        try:
            body = _extrude_body(ext)
        except Exception:
            return None
        if body is None:
            return None
        try:
            if result is None:
                result = body
            elif ext.operation in (0, 1):  # NewBody / Join
                result = result.fuse(body)
            elif ext.operation == 2:  # Cut
                result = result.cut(body)
            elif ext.operation == 3:  # Intersect
                result = result.intersect(body)
        except Exception:
            return None
    return result


# ------------------------------------------------------- solid -> point cloud

def solid_to_pc(solid, n_points=2000, tol=0.02, seed=0):
    """Tessellate solid and uniformly sample n_points on the surface."""
    verts, tris = solid.tessellate(tol)
    vs = np.array([[v.x, v.y, v.z] for v in verts], dtype=np.float64)
    if len(vs) == 0 or len(tris) == 0:
        return None
    mesh = trimesh.Trimesh(vs, np.asarray(tris), process=False)
    if mesh.area <= 0:
        return None
    pc, _ = sample_surface(mesh, n_points, seed=seed)
    return np.asarray(pc, dtype=np.float32)


def vec_to_pc(vec, n_points=2000, tol=0.02, seed=0):
    solid = vec_to_solid(vec)
    if solid is None:
        return None
    return solid_to_pc(solid, n_points=n_points, tol=tol, seed=seed)


# ------------------------------------------------------------------- metrics

def scale_consistency_ratio(pc_a, pc_b, eps=1e-12):
    """BBox max-extent ratio between two point clouds: (ratio, (ext_a, ext_b)).

    ratio >= 1 by construction. Gate for any cross-source geometric comparison
    (e.g. gen from one builder vs GT from another): ratio > 1.5 signals a
    cross-builder / cross-normalization scale mismatch — the 2026-08-01
    Text2CAD T2 incident (DeepCAD-h5 GT at extents ~1.5 vs Text2CAD-STEP gen
    at ~0.75 gave ratio ~2.0 and scrambled CD rankings). For a pipeline, the
    *median* ratio over >=20 random pairs should be < 1.5; per-pair variation
    from generation imperfection is normal, a systematic offset is not.
    """
    a = np.asarray(pc_a, dtype=np.float64)
    b = np.asarray(pc_b, dtype=np.float64)
    ext_a = float((a.max(axis=0) - a.min(axis=0)).max())
    ext_b = float((b.max(axis=0) - b.min(axis=0)).max())
    ratio = max(ext_a, ext_b) / max(min(ext_a, ext_b), eps)
    return ratio, (ext_a, ext_b)


def chamfer_distance(pc_a, pc_b):
    """Symmetric Chamfer Distance (DeepCAD evaluation convention):
    mean of (min dist a->b).mean() and (min dist b->a).mean(), averaged."""
    a = torch.from_numpy(np.asarray(pc_a, dtype=np.float32)).unsqueeze(0)
    b = torch.from_numpy(np.asarray(pc_b, dtype=np.float32)).unsqueeze(0)
    d = torch.cdist(a, b)  # (1, Na, Nb)
    cd = (d.min(dim=2).values.mean() + d.min(dim=1).values.mean()) / 2.0
    return float(cd.item())


def f_score(pc_a, pc_b, threshold=0.01):
    """F-score at distance threshold (precision/recall of nearest-neighbor hits)."""
    a = torch.from_numpy(np.asarray(pc_a, dtype=np.float32)).unsqueeze(0)
    b = torch.from_numpy(np.asarray(pc_b, dtype=np.float32)).unsqueeze(0)
    d = torch.cdist(a, b)
    precision = (d.min(dim=2).values < threshold).float().mean().item()
    recall = (d.min(dim=1).values < threshold).float().mean().item()
    if precision + recall == 0:
        return 0.0, precision, recall
    f = 2 * precision * recall / (precision + recall)
    return float(f), float(precision), float(recall)
