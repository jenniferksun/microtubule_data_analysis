import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name='microtubule_data_analysis',
    version='0.0.1',
    author='Jennifer Sun',
    author_email='jksun@caltech.edu',
    description='Microtubule catastrophe times data analysis',
    long_description=long_description,
    long_description_content_type='ext/markdown',
    packages=setuptools.find_packages(),
    install_requires=["numpy","pandas", "bokeh>=1.4.0"],
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ),
)