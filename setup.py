from setuptools import setup, find_packages

setup(
    name="poetryeeg_anlys",
    version="0.1.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "mne",
        "numpy",
        "pandas",
        "matplotlib",
        "scipy",
        "scikit-learn",
        "pyarrow",
        "fastparquet",
        # "PyWavelets",
    ],
    extras_require={
        "dl": ["torch", "tensorflow"],
        "dev": ["pytest", "black", "ruff"],
    },
)

# run from the root directoy
# pip install -e .
