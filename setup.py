# -*- coding: utf-8 -*-
"""
Created on Fri May 14 11:40:22 2021

@author: pedro
"""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()


setuptools.setup(name='MGEDT',
                 version='0.2.46',
                 description='MGEDT - Multi-objective Grammatical Evolution Decision Trees for classification',
                 author='Pedro José Pereira, Paulo Cortez, Rui Mendes',
                 license='MIT',
                 long_description=long_description,
                 long_description_content_type="text/markdown",
                 author_email='id6927@alunos.uminho.pt',
                 packages=setuptools.find_packages(),
                 install_requires=requirements,
                 url="https://github.com/p-pereira/MGEDT",
                 keywords=['Decision Trees', 'Grammatical Evolution', 'Lamarckian Evolution'],
                 python_requires='>=3.5',
                 include_package_data=True)