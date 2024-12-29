from setuptools import setup,find_packages
from typing import List

HYPEN_E='-e .'
def get_requirements(file_path:str)->List[str]:
    requiremets=[]
    with open(file_path)  as data:
            requiremets=data.readlines()
            requiremets=(req.replace("\n","") for req in requiremets)

            if 'HYPEN_E' in requiremets:
                requiremets.remove("HYPEN_E")
                    
    return requiremets


setup(
name='mlproject',
version='0.0.1',
author='S158984',
author_email='sdharshan.nalla@gmail.com',
packages=find_packages(),
#install_requrires=['pandas','seaborn','numpy','scikit-learn','matplotlib'],

nstall_requrires=get_requirements('requirements.txt')

)