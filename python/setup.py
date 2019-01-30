#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='Anomaly Detection Examples',
    url='https://github.com/shubhomoydas/ad_examples',
    author='Shubhomoy Das',
    author_email='smd.shubhomoydas@gmail.com',
    license='MIT',
    packages=find_packages(),
    scripts=[
        'iso_gan.sh',
        'glad.sh',
        'gan.sh',
    ],
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
)
