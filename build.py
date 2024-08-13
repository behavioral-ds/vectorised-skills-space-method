import pkg_resources
from typing import Literal

installed_packages = pkg_resources.working_set

numpy_variant: Literal["cupy", "mlx.core", "numpy"] = "numpy"

for package in installed_packages:
    package_name = package.key
    
    if "cupy" in package_name:
        numpy_variant = "cupy"
        break

    if package_name == "mlx":
        numpy_variant = "mlx.core"
        break
    
with open("./utils/numpy_variant.py", "w") as f:
    f.write(f"import {numpy_variant} as xp\n")
