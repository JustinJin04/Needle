from setuptools import setup, find_packages

setup(
    # --- Basic Info ---
    name="needle",
    version="0.1.0",  # You can update this as you release new versions
    description="Needle Deep Learning Framework (CMU 10-714)",
    
    # --- Author Info (Optional but Good Practice) ---
    # author="Your Name",
    # author_email="your.email@example.com",

    # --- Package Finding ---
    # This tells setuptools that your package's source code is
    # in the 'python' directory.
    package_dir={'': 'python'},
    
    # This automatically finds all packages (directories with __init__.py)
    # inside the 'python' directory.
    # It will find 'needle', 'needle.nn', 'needle.data', 'needle.data.datasets', etc.
    packages=find_packages(where='python'),

    # --- Include Non-Python Files ---
    # This is CRITICAL for including your compiled .so files.
    # It tells setuptools to find any '.so' files inside the
    # 'needle/backend_ndarray' package directory.
    package_data={
        'needle': ['backend_ndarray/*.so'],
    },
    include_package_data=True,

)