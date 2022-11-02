import setuptools
import os

with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.rst'), encoding='utf-8') as f:
    readme = f.read()

setuptools.setup(
    name="deepod",
    version="0.1.1",
    author="Hongzuo Xu",
    author_email="hongzuoxu@126.com",
    description="",
    long_description=readme,
    long_description_content_type="text/x-rst",
    license='MIT License',
    url="https://github.com/xuhongzuo/DeepOD",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: BSD License",
    ],
)
