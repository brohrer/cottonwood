import setuptools as st

with open("README.md", "r") as fh:
    long_description = fh.read()

st.setup(
    name='cottonwood',
    version='7.1',
    description='A flexible machine learning framework',
    url='http://github.com/brohrer/cottonwood',
    download_url='https://github.com/brohrer/cottonwood/tags/',
    author='Brandon Rohrer',
    author_email='brohrer@gmail.com',
    license='MIT',
    install_requires=[
        'matplotlib',
        'numpy',
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=st.find_packages(exclude=("")),
    package_data={
        "": [
            "README.md",
            "LICENSE",
        ],
        "cottonwood": [],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
