from setuptools import setup, find_packages

setup(
    name="allegro_leader",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy<2",
    ],
    author="Takahisa Ueno",
    author_email="takahisaueno@fuji.waseda.jp",
    description="Package for using leader structure for allegrohand",
    python_requires=">=3.7",
    classifiers=[
    ],
)
