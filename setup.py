from setuptools import setup, find_packages

setup(
    name="interpretable_anomaly_detection",
    version="0.0.1",
    author="Felix",
    author_email="felix.gerschner@hs-aalen.de",
    description="A brief description of your package",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="http://github.com/felixg97/interpretable-anomaly-detection",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)
