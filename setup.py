import setuptools

def get_version():
    with(open("src/idiff/__init__.py", "r")) as fh:
        for line in fh:
            if line.startswith("__version__"):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
        else:
            raise RuntimeError("Unable to find version string.")
                
def get_long_description():
    with open("README.md", "r") as fh: description = fh.read()
    return(description)
    
setuptools.setup(
    name="idiff",
    version=get_version(),
    author="Mirek Slouf",
    author_email="mirek.slouf@gmail.com",
    description="Improve 2D powder diffraction patterns",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/mirekslouf/idiff/",
    project_urls={
        "Documentation": "https://mirekslouf.github.io/idiff/"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"],
    license='MIT',
    package_dir={"":"src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "scipy",         # bkg2d.py, deconv.py, peaks.py
        "matplotlib",    # deconv.py, psf.py
        "scikit-image",  # bkg2d.py, peaks.py
        "opencv-python", # bkg2d.py: cv2.GaussianBlur
        "onnxruntime",   # bkg2d.py: neural-network background removal
        "tqdm",          # deconv.py
        "stemdiff",      # psf.py: stemdiff.io for reading 4D-STEM datafiles
    ],
    extras_require={
        # GPU acceleration for deconv.py. CuPy is optional: the wheel must
        # match the installed CUDA version, and deconv.py already guards the
        # import and falls back to the CPU path (cuda=False is the default).
        "gpu": ["cupy-cuda12x"],
    },
    include_package_data=True)
