from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='netallocation',
    version='0.0.5',
    author='Fabian Hofmann (FIAS)',
    author_email='hofmann@fias.uni-frankfurt.de',
    description='Package for allocating flows and costs in a PyPSA network',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/FRESNA/netallocation',
    license='GPLv3',
    packages=find_packages(exclude=['doc', 'test']),
    include_package_data=True,
    install_requires=['pypsa','pandas', 'pyyaml', 'xarray', 'progressbar2',
                      'sparse', 'dask', 'h5py', 'scipy', 'geopandas', 'pyyaml',
                      'netcdf4'],
    extras_require={
        'docs': [
            'sphinx >= 1.4',
            'sphinx_rtd_theme']},
    classifiers=[
#        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Natural Language :: English',
        'Operating System :: OS Independent',
    ])

