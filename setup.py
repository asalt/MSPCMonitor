from setuptools import setup, find_packages

version = "0.0.1"

setup(
    name='mspcmonitor',
    version=version,
    #packages=find_packages(include=['bin', 'bin.*']),
    install_requires=[
        #'requests',
        #'importlib; python_version >= "3.6"',
    ],
    #packages=find_packages(),
    #package_dir = {"MSPCRunner": "bin"},
    author = 'Alex Saltzman',
    author_email = 'a.saltzman920@gmail.com',
    description = 'xxCLI runner interface for processing mass spec proteomics data',
    entry_points= """
    [console_scripts]
    mspcmonitor=mspcmonitor.main:app
    """,
    #py_modules = ['main'],
    #package_data = {"mspcrunner": ['../../ext/*', '../../params/*']},
    include_package_data = True

)
