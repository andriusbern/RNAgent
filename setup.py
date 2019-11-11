from setuptools import setup

packages = [
    'numpy==1.17.1',        
    'tensorflow==1.13.1',   
    'stable-baselines==2.7.0',
    'gym',
    'forgi',
    'matplotlib',
    'PyQt5',
    'pyqtgraph',
    'tqdm',
    'pyyaml==5.1']

setup(
    name='rlif',
    description='Inverse RNA folding with reinforcement learning.',
    long_description='',
    version='0.2',
    packages=['rlif'],
    scripts=[],
    author='Andrius Bernatavicius',
    author_email='andrius.bernatavicius@gmail.com',
    url='none',
    download_url='none',
    install_requires=packages)

print("Installation complete.\n")
