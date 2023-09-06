import setuptools
import os

with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.rst'), encoding='utf-8') as f:
    readme = f.read()

with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'requirements.txt'), encoding='utf-8') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="deepod",
    version="0.4.1",
    author="Hongzuo Xu",
    author_email="hongzuoxu@126.com",
    description="",
    long_description=readme,
    long_description_content_type="text/x-rst",
    license='MIT License',
    url="https://github.com/xuhongzuo/DeepOD",
    keywords=['outlier detection', 'anomaly detection', 'deep anomaly detection',
              'deep learning', 'data mining'],
    packages=setuptools.find_packages(exclude=['test']),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: BSD License",
    ],
)
