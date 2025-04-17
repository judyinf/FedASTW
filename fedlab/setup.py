from setuptools import setup, find_packages

setup(
    name="fedlab",
    version="0.1.0",
    description="A library for federated learning.",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        # Add your dependencies here, e.g., 'numpy', 'torch'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)