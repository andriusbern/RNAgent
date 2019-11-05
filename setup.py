from setuptools import setup
import time, os

packages = ['numpy',        
            'tensorflow==1.4.0',   # ML,
            'gym',
            'sklearn',
            'opencv-python',# Image processing
            'matplotlib',   # Visualization
            # 'python-qt',
            # 'PyQt5',         
            # 'pyqtgraph',
            'stable_baselines',
            'pyyaml',
            'forgi',
            'selenium',
            'tqdm'
            ]
setup(
    name='rlfold',
    description='Inverse RNA folding with reinforcement learning.',
    long_description='',
    version='0.2',
    packages=['rlfold'],
    scripts=[],
    author='Andrius Bernatavicius',
    author_email='andrius.bernatavicius@gmail.com',
    url='none',
    download_url='none',
    install_requires=packages
)

print ("Installation complete.\n")
