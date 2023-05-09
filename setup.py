from setuptools import find_packages, setup

setup(
    name="brisco",
    packages=find_packages(),
    install_requires=[
        "cricksaw_analysis @ git+ssh://git@github.com/znamlab/cricksaw-analysis.git",
        "flexiznam @ git+ssh://git@github.com/znamlab/flexiznam.git",
    ],
)
