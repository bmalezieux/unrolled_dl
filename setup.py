import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="plipy",
    version="0.0.1",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
        'cython',
        'scipy',
        'pandas',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'torch',
        'torchvision',
        'Pillow',
        'pathlib',
        'tqdm',
        'joblib',
        'pygam',
        'alphacsc',
        'spams'
    ],
    python_requires='>=3.6',
)
