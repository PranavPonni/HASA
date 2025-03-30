from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup
d = generate_distutils_setup(
    packages=['allegro_package'],
    package_dir={'': 'src'},
    requires=['rospy']
)

setup(**d)
