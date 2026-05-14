from setuptools import setup, find_packages

setup(
    name="poetryeeg_anlys",
    version="0.1.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "mne",
        "numpy",
        "pandas",
        "matplotlib",
        "pathlib",
        "scipy",
        "scikit-learn",
        "tensorflow",
        "torch",
        "pyarrow",
        "fastparquet"
        # "PyWavelets",
        # add more dependencies here if needed
    ],
)

# run from the root directoy
# pip install -e .
