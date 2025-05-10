# Import the 'setup' function from the Python 'distutils' library.
# This function is used to handle the installation process for Python packages.
from distutils.core import setup

# Import 'generate_distutils_setup' from 'catkin_pkg.python_setup'.
# This function creates a dictionary of setup parameters based on the package's structure,
# which is tailored for Catkin (the ROS build system).
from catkin_pkg.python_setup import generate_distutils_setup

# Generate the distutils setup configuration for the ROS package.
# Here we specify that the package name is "allegro_hand", and its source code
# is located in the "src" directory. The empty string ("") in the package_dir dictionary
# indicates that the package's Python modules are directly under the "src" folder.
allegro_hand_setup = generate_distutils_setup(
    packages=["allegro_hand"], package_dir={"": "scripts"}
)

# Call the setup function with the generated setup parameters.
# This initiates the packaging process so that the "allegro_hand" package
# can be properly installed and used within the ROS ecosystem.
setup(**allegro_hand_setup)
