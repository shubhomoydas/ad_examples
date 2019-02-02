#!/usr/bin/env python

import os
from distutils.core import setup


version = '0.0.1'
package_name = 'ad-examples'


# collect all the test dataset paths
def package_files(directory, relative_parent=""):
    paths = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        relative_path = os.path.join(relative_parent, filename)
        if os.path.isfile(filepath):
            if not str(filename).startswith("."):  # ignore .DS_Store
                paths.append(relative_path)
        else:
            paths.extend(package_files(filepath, relative_path))
    return paths


package_data = {'ad_examples.datasets': package_files("ad_examples/datasets")}


setup(
    name=package_name,
    version=version,
    description='Active Anomaly Detection and Simple Anomaly Detection Examples',
    long_description="""
        A collection of anomaly detection methods (iid/point-based, graph and time series) 
        including active learning for anomaly detection/discovery, bayesian rule-mining, 
        description for diversity/explanation/interpretability.
    """,
    author='Shubhomoy Das',
    author_email='da.shubhomoy@gmail.com',
    url='https://github.com/shubhomoydas/ad_examples',
    package_dir = {'ad_examples': 'ad_examples'},
    packages=['ad_examples',
              'ad_examples.aad',
              'ad_examples.ad',
              'ad_examples.bayesian_ruleset',
              'ad_examples.classifier',
              'ad_examples.common',
              'ad_examples.datasets',
              'ad_examples.dnn',
              'ad_examples.glad',
              'ad_examples.graph',
              'ad_examples.loda',
              'ad_examples.percept',
              'ad_examples.timeseries'],
    package_data=package_data,
    include_package_data=True,
    install_requires=[
        'cvxopt>=1.1.9',
        'matplotlib>=2.1.0',
        'numpy>=1.14.0',
        'pandas>=0.22.0',
        'ranking',
        'scipy>=1.0.0',
        'statsmodels>=0.9.0',
        'scikit-learn>=0.19.1',
        # 'tensorflow>=1.6.0',  # *NOT* required for AAD. Required for GLAD and a few other stuff such as timeseries
    ],
    classifiers=[
        'Programming Language :: Python',
    ],
)
