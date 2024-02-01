from setuptools import setup, find_packages

setup(
    name='DS_Exercise9', # fill this
    version='0.0.0',    
    description='A example Python package',
    url='https://github.com/VitusPa/DS_Exercise9', # fill this
    author='Vitus Past', # fill this if you want
    author_email='none', # fill this if you want
    license='BSD 2-clause',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=['pandas>2.0',
                      'numpy',
                      'matplotlib',
                      'pyarrow'                     
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