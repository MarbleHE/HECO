# ABC Installation Notes

The following notes are mainly for an installation on Fedora or Debian.

## Prerequisites

### Packages

Install `cmake`, `doxygen`, and `clang`/`gcc`.
`libboost` and `python3` dev tools are needed for pybind11.

Fedora:
```
dnf install cmake doxygen clang boost-devel python3-devel
```

Debian:
```
apt install cmake doxygen clang libboost-all-dev python3-dev
```

### Install SEAL

#### Clone repo

Currently, the most recent version of SEAL with which this project was tested is `3.6.6`.

```
git clone -b v3.6.6 https://github.com/microsoft/SEAL.git
```

#### Building

Use the following commands to build SEAL. For more build options, see the official [SEAL repo](https://github.com/Microsoft/SEAL#getting-started).

<!-- 
TODO [Miro]
@Alex: the `-DSEAL_THROW_ON_TRANSPARENT_CIPHERTEXT=OFF` is here because EVA requires it. Do we need it too?
-->

```
cmake -DSEAL_THROW_ON_TRANSPARENT_CIPHERTEXT=OFF -S . -B build
cmake --build build
```

#### Install

Install the library.

```
sudo cmake --install build   
```

## Install ABC

### Clone repo

```
git clone git@github.com:MarbleHE/ABC.git
```

### Use CMakefiles

The project is developed with CLion. Open it there and let the CMakefiles install all dependencies.

## Check implementation

1. Check that the CMake project runs through without any fatal error 

    - Troubleshooting: first, try to use "Reload Cmake Project" and/or delete the `cmake-build-debug` folder to make a fresh new build.
2. Run the "testing-all" target in CLion to execute all tests and make sure they pass on your local system. Some tests are disabled.
    - Troubleshooting: if this entry is missing, do the following to add it:
      - Open the dropdown menu with "Run/Debug Configurations"
      - Select "Edit Configurations"
      - Go to Google Tests
      - Click "add new run configuration"
      - Name it "testing-all"
      - Select "resting-all" as target
      - Save it and run (play symbol) the target
