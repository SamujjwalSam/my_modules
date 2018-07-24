from setuptools import setup

setup(name='my_modules',
    version='0.1',
    description='My python codes',
    url='https://github.com/samiith/my_modules',
    author='Samujjwal Ghosh',
    author_email='cs16resch01001@iith.ac.in',
    license='Python',
    packages=['my_modules'],
    zip_safe=False,
    install_requires=[
        'unidecode',
        'numpy',
        'scipy==1.1.0',
        'scikit-learn==0.19.1',
        'matplotlib==2.2.2',
        'textblob',
        'numpy',
        'nltk',
        'scikit-learn',
        'gensim',
        'spacy',
        'pandas'
        ]
    )
