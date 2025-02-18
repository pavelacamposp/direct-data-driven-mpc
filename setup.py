import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="direct_data_driven_mpc",
    version='1.0',
    description=("Python implementation of robust and nonlinear Direct "
                 "Data-Driven MPC controllers for LTI and nonlinear systems, "
                 "based on the work of J. Berberich et al."),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Pável A. Campos-Peña',
    author_email='pcamposp@uni.pe',
    url='https://github.com/pavelacamposp/direct_data_driven_mpc',
    packages=setuptools.find_packages(include=["direct_data_driven_mpc*"]),
    install_requires=['numpy',
                      'matplotlib>=3.9.0',
                      'cvxpy',
                      'tqdm',
                      'PyYAML',
                      'PyQt5'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    license='MIT'
)