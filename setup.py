from src.utils import *
from setuptools import find_packages, setup

root_dir = get_root_directory()
hyphen_e_dot = "-e ."

def get_requirements(file_path:str) -> list[str]:
    """
    this function will return the list of requirements
    Args:
        file_path (str): directory of the requirement file

    Returns:
        list[str]: _description_
    """
    
    requirements = []
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [req.replace("\n","") for req in requirements]
        
        if hyphen_e_dot in requirements:
            requirements.remove(hyphen_e_dot)
            
    return requirements

setup(
    
name='cropyieldproject',
version='0.01',
author='melvin',
author_email='nnjuaka@gmail.com',
packages=find_packages,
install_requires=get_requirements('requirements.txt')

)