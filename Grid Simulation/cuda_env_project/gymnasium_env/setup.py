from setuptools import setup, find_packages

setup(
    name="gymnasium_env",          # Replace with your project name.
    version="0.1",
    packages=find_packages(),         # Automatically find packages in your project.
    install_requires=[                # List your project dependencies here.
        "gymnasium",
        "torch",
        "numpy",
        "matplotlib",
        # Add any other dependencies your project requires.
    ],
)
