from setuptools import setup

setup(name='MSATwtdenoiser',
      version='0.0.1',
      description='Denoising a 2D signal based on wavelet',
      url='',
      author='Amir Souri',
      author_email='ahsouri@gmail.com',
      license='MIT',
      packages = ["MSATwtdenoiser"],
      install_requires=[
          'numpy','matplotlib','scipy','pywt'
      ],
      zip_safe=False)
