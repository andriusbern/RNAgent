from setuptools import setup

packages = [
    'tensorflow==2.6.4',
    'numpy',  
    'gym',
    'stable_baselines==2.7.0',
    'forgi',
    'matplotlib',
    'Pyside2',
    'pyqtgraph==0.11.0rc0',
    'tqdm',
    'pyyaml==3.13']

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
