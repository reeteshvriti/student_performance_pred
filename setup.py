from setuptools import find_packages, setup

# this file is a kind of package installation

hype_e  = "-e ."

def get_requirements(file_name):
    ''' this function will return the list
    of requirements'''

    requirements =[]
    with open(file_name) as file:
        requirements= file.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if hype_e in requirements:
            requirements.remove(hype_e)

    return requirements        


setup(
name = "student_performance", 
version = '0.0.1',
author='Reetesh V',
author_email= "reeteshvenki@gmail.com",
packages= find_packages(),
install_requires = get_requirements('requirements.txt')

)