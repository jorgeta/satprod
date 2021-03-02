from setuptools import setup, find_packages

setup(
    name='satprod',
    package_dir={'': 'src'},
    packages=find_packages('src'),
    entry_points={
        'console_scripts': [
            'satprod=scripts.app:main'
        ]
    }
)