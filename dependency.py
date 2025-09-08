import pkgutil
import subprocess
import sys

REQUIRED_PACKAGES = [
    "numpy",
    "pandas",
    "lightgbm",
    "pyyaml",  # for reading YAML configuration files
    "scikit-learn",  # optional utilities for model evaluation
]

def install_missing_packages(packages):
    for pkg in packages:
        if pkgutil.find_loader(pkg) is None:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

if __name__ == "__main__":
    install_missing_packages(REQUIRED_PACKAGES)
