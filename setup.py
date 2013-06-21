import os.path
from setuptools import setup, find_packages

HERE = os.path.abspath(os.path.dirname(__file__))

README_PATH = os.path.join(HERE, 'README.md')
try:
    README = open(README_PATH).read()
except IOError:
    README = ''


setup(
  name='codd',
  py_modules = ['codd'],
  version='0.1.7',
  description='relational alegrbra for functional programs',
  long_description=README,
  author='Scott Robertson',
  author_email='scott@triv.io',
  url='http://github.com/trivio/codd',
  classifiers=[
      "Programming Language :: Python",
      "License :: OSI Approved :: MIT License",
      "Operating System :: OS Independent",
      "Development Status :: 3 - Alpha",
      "Intended Audience :: Developers",
      "Topic :: Software Development",
  ],
  install_requires=[
    'xlocal'
  ]
)
