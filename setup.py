from setuptools import setup

setup(name='MSATwtdenoiser',
      version='0.0.2',
      description='Denoising a 2D signal based on wavelet',
      long_description='',
      url='https://github.com/ahsouri/MSATwtdenoiser',
      author='Amir Souri',
      author_email='ahsouri@gmail.com',
      license='MIT',
      packages=['MSATwtdenoiser'],
      install_requires=[
          'numpy','matplotlib','scipy','PyWavelets'
      ],
      zip_safe=False)
