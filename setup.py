from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='mcl_toolbox',
    version='',
    packages=['mcl_toolbox', 'mcl_toolbox.env', 'mcl_toolbox.utils', 'mcl_toolbox.models', 'mcl_toolbox.mcrl_modelling',
              'mcl_toolbox.computational_microscope'],
    url='',
    license='',
    author='Ruiqi He, Yash Raj Jain',
    author_email='',
    description='',
    install_requires=requirements
)
