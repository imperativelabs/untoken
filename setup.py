from setuptools import setup, find_packages

setup(
    name="untoken",
    version="0.1.0",
    description="Token compression for LLM prompts",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Adib Mohsin",
    url="https://github.com/pacifio/untoken",
    license="MIT",
    python_requires=">=3.10",
    packages=find_packages(include=["untoken*"]),
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "tokenizers>=0.15.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "train": [
            "datasets>=2.16.0",
            "sentence-transformers>=2.2.0",
            "accelerate>=0.25.0",
            "wandb>=0.16.0",
            "bert-score>=0.3.13",
            "rouge-score>=0.1.2",
            "scipy>=1.11.0",
            "scikit-learn>=1.3.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0",
            "ruff>=0.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "untoken=untoken.inference:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    keywords="llm prompt compression token reduction nlp transformers",
)
