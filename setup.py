from setuptools import setup

setup(
   name='stochastic-thermo',
   version='1.1',
   description='Stochastic thermodynamics in Python',
   author='Artemy Kolchinsky',
   author_email='artemyk@gmail.com',
   packages=['stochastic-thermo'],
   install_requires=['numpy', 'scipy',],
)