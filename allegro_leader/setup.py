import os
from setuptools import setup, find_packages


def package_files(root_dir: str):
    """
    Recursively collect files under root_dir for package_data inclusion.
    Returns paths relative to the package directory.
    """
    paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(full_path, 'allegro_leader')
            paths.append(rel_path)
    return paths

urdf_files = package_files('allegro_leader/vis/urdf')

setup(
    name='allegro_leader',
    version='0.1.0',
    author='Takahisa Ueno',
    author_email='takahisaueno@fuji.waseda.jp',
    description='Package for using leader structure for Allegro hand teleoperation',
    url='https://github.com/yourusername/allegro_leader',
    packages=find_packages(exclude=['tests', 'docs']),
    include_package_data=True,
    package_data={
        'allegro_leader': urdf_files,
    },
    install_requires=[
        'numpy>=1.18.0',
        'pin-pink>=0.1.0',
        'meshcat>=0.3.2',
        'pyserial>=3.4',
    ],
    python_requires='>=3.7',
)
