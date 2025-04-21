from setuptools import setup, find_packages

setup(
    name="model_validation",
    version="0.1.0",
    packages=find_packages(),
    description="A module provide standard methods of classfication model evaluation",
    author="Swapnil amale",
    author_email="swapnil.amale@99Acres.com",
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "shap",
        "tqdm",
        "typing_extensions",
        "scikit-learn"
    ],
    python_requires='>=3.8'
)
