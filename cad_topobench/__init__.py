"""CAD-TopoBench: Failure-Oriented Diagnostic Benchmark for Topological Reliability in Neural CAD Generation.

Subpackage layout:
  geometry  - vec -> solid -> point cloud -> Chamfer Distance (cadquery 2.8 / OCP backend)
  topology  - Betti signature utilities and failure classification
  dataset   - benchmark record building / loading
  metrics   - F1-F4 diagnostic metric implementations
"""

__version__ = "0.1.0"
