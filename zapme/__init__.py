"""zapme top-level package marker.

Real code lives in `zapme.src` and its sub-packages. This file exists so
the outer `zapme` directory is recognized as a regular package, which
keeps imports like `zapme.src.model.vision` resolvable both for the
installed package (`pip install -e .`) and for tooling that prefers
explicit package boundaries over implicit namespace packages.
"""
