#!/usr/bin/env python3
"""Pre-generate all Cython ufuncs for weathercop copulas.

This script imports all copulas, triggering the generation of their
associated Cython ufunc modules. This ensures that all required .pyx
and .c files exist before the package is built, allowing them to be
compiled into the wheel distribution.

Without this step, the ufuncs would need to be compiled on-demand at
runtime, which causes race conditions with parallel test execution.
"""

import sys
from pathlib import Path

# Ensure the src directory is in the path for development builds
src_path = Path(__file__).parent.parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))


def main():
    print("=" * 70)
    print("Pre-generating Cython ufuncs for all weathercop copulas")
    print("=" * 70)
    print()

    # Import copulae module - this triggers metaclass initialization
    # which generates all ufunc modules as a side effect
    print("Importing weathercop.copulae...")
    from weathercop import copulae

    # Count generated files
    ufunc_dir = Path(copulae.get_ufunc_dir())
    pyx_files = list(ufunc_dir.glob("*.pyx"))
    c_files = list(ufunc_dir.glob("*code_0.c"))

    print(f"\nGenerated {len(pyx_files)} .pyx wrapper files")
    print(f"Generated {len(c_files)} .c implementation files")
    print(f"\nAll files are in: {ufunc_dir}")
    print()

    # Display copula count
    print(f"Total copulas defined: {len(copulae.all_cops)}")
    print(f"  - Base copulas: {sum(1 for name in copulae.all_cops if not any(r in name for r in ['_90', '_180', '_270']))}")
    print(f"  - Rotated variants: {sum(1 for name in copulae.all_cops if any(r in name for r in ['_90', '_180', '_270']))}")
    print()

    print("=" * 70)
    print("Pre-generation complete!")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())