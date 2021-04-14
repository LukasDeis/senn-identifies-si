
"""
In a normal environment the following will install all necessary packages:
!pip install sklearn
!pip install numpy
!pip install pandas
!pip install tensorflow #if posible use -gpu
!pip install pydot
!pip install pydotplus
!pip install graphviz
!pip install datetime
!pip install packaging
!pip install keras
"""


"""
In the RadboudUMC DRE environment everything was installed like this:
from local_package_installer.local_package_installer import install_local

install_local('sklearn')
install_local('pandas')
#install_local('tensorflow')
#install_local('pydot')
#install_local('pydotplus')
#install_local('graphviz')
install_local('datetime')
install_local('packaging')
install_local('keras')
install_local('numpy==18.4')
install_local('tensorflow-gpu') #tensorflow-gpu
install_local("pyreadstat==1.0.5") # this one required me to manually copy a dll to a different location

"""

"""
#using this function:
#Run following commands in your Python session (only once per virtual machine per Python
# version/environment):
import sys
import subprocess
import re

def installer_local(package):
    try:
        print('Installing local package installer.')
        call = [sys.executable, '-m', 'pip', 'install', '--user', '--upgrade',\
                '--trusted-host=drefilesrv01.researchenvironment.org',\
                '--index-url=http://drefilesrv01.researchenvironment.org/PythonInstaller/',\
                package]
        process = subprocess.Popen(call, stdin=subprocess.PIPE, stdout=subprocess.PIPE,\
                                   stderr=subprocess.STDOUT, universal_newlines=True)
        while True:
            output = process.stdout.readline()
            if process.poll() is not None:
                break
            if any(re.findall(r'error', output.strip(), re.IGNORECASE)):
                raise Exception
            if output:
                print(output.strip())
        print('Even if the package is installed, you possibly have to restart Python before you '\
              'can import the module.')
    except:
        print('Package could not be installed.')

installer_local('pip')
installer_local('local_package_installer')

#Run following commands in your Python session (per Python session):
from local_package_installer.local_package_installer import install_local

#Examples, remove the hashtag and run the command. Replace the package
#name (and version if applicable) for the package you want to install:
#install_local('numpy')
"""