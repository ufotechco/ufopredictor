from setuptools import setup

setup(
    name='ufopredictor',
    version='0.1.0',    
    description='Model predictor',
    url='https://github.com/ufotechco/ufopredictor',
    author='UFOTECH',
    author_email='gerencia@ufotech.co',
    license='GNU',
    packages=['UFOPredictor'],
    install_requires=[
        'opencv-python>4.0',
        'numpy==1.17.3',
        'Keras==2.3.1',
        'Keras-Applications==1.0.8',
        'Keras-Preprocessing==1.1.0',
        'h5py==2.10.0',
        'matplotlib>3',
        'tensorflow'
    ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)