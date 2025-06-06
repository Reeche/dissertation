from setuptools import setup

setup(
    name="mcl_toolbox",
    version="",
    packages=[
        "mcl_toolbox",
        "mcl_toolbox.env",
        "mcl_toolbox.utils",
        "mcl_toolbox.models",
        "mcl_toolbox.mcrl_modelling",
        "mcl_toolbox.computational_microscope",
    ],
    url="",
    license="",
    author="Ruiqi He, Yash Raj Jain",
    author_email="",
    description="",
    setup_requires=["wheel"],
    include_package_data=True,
    install_requires=[
        "mouselab @ git+https://github.com/RationalityEnhancementGroup/mouselab-mdp-tools.git@main#egg=mouselab",  # noqa
        "graphviz",
        "statsmodels",
        "toolz",
        "mpmath",
        "pandas",
        "hyperopt",
        "torch",
        "matplotlib",
        "scipy",
        "pyabc",
        "seaborn",
        "joblib",
        "numpy",
        "imageio",
        "ipython",
        "pymannkendall",
        "scikit-learn",
        "rpy2",
        "pyro-ppl"
    ],
)
