from setuptools import find_packages, setup
from typing import List


HYPHEN_E_DOT = '-e .'

def get_packages(file_path:str) -> List[str]:
    '''
    This function will return the list of requirement packages
    '''
    requirements = []
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [req.replace('\n', '') for req in requirements]

    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)

    return requirements


setup(
    name = 'real-time-style-transfer',
    version = '0.0.1',
    author='Tushaar',
    author_email='tushaar.sharma05@gmail.com',
    packages = find_packages(),
    install_requires = get_packages('requirements.txt')
)