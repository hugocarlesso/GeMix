from setuptools import setup, find_packages

setup(
    name="gemix",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "tqdm",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "pandas",
        "opencv-python",
        "pillow",
    ],
)
