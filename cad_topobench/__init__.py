"""CAD-TopoBench: Failure-Oriented Diagnostic Benchmark for Topological Reliability in Neural CAD Generation.

Subpackage layout:
  reference  - solid_betti(): watertight Betti signature extraction (reference v1.0)
  geometry   - vec -> solid -> point cloud -> Chamfer Distance (cadquery 2.8 / OCP backend)
  paths      - all locations resolve via CAD_TOPOBENCH_ROOT (default: repo root)
"""

__version__ = "0.1.0"
