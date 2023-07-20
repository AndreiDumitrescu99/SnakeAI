from setuptools import setup, find_packages

setup(
    name="snake_ai",
    version="1.0.0",
    author="Andrei Dumitrescu",
    author_email="andreidumitrescu99@gmail.com",
    packages=find_packages(),
    install_requires=[
        "gymnasium",
        "pygame==2.1.0"
    ]
)