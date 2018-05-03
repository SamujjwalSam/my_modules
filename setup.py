from setuptools import setup

setup(name='my_modules',
      version='0.1',
      description='My python codes',
      # url='http://github.com/storborg/funniest',
      author='Samujjwal Ghosh',
      # author_email='flyingcircus@example.com',
      license='MIT',
      packages=['my_modules'],
      zip_safe=False,
      install_requires=['textblob', 'numpy', 'nltk', 'scikit-learn', 'gensim',
                        'spacy', 'pandas'])
