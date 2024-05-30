from setuptools import setup, find_packages


setup(name='taskplan',
      version='1.0.0',
      description='Core code for task planning in restaurant environment',
      license="MIT",
      author='Raihan Islam Arnob, Ridwan Hossain',
      author_email='rarnob@gmu.edu',
      packages=find_packages(),
      install_requires=['numpy', 'matplotlib'])