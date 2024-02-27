from setuptools import find_packages, setup

setup(
    name="brisc",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "pathlib",
        "matplotlib",
        "czifile",
        "bg-atlasapi",
        "opencv-python",
        "Pillow",
        "iss_preprocess @ git+ssh://git@github.com/znamlab/iss-preprocess.git",
        "cricksaw_analysis @ git+ssh://git@github.com/znamlab/cricksaw-analysis.git",
        "flexiznam @ git+ssh://git@github.com/znamlab/flexiznam.git",
        "scanpy",
    ],
)
