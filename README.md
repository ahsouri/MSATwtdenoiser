# MSATwtdenoiser

A wavelet-based de-noising method designed for MethaneSAT xCH4 retrieval (or any other noisy image)

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install MSATwtdenoiser.

```bash
pip install MSATwtdenoiser==0.0.2
```

## Usage

see example.py

or

```bash
from MSATwtdenoiser import MSATdenoise,example
example()
```
or
```bash
from MSATwtdenoiser import MSATdenoise
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import inspect


img = np.array(mpimg.imread('steve.jpg'))
img.astype(float)
img = np.mean(img,axis=2)
img = img+np.random.normal(0,100,[np.shape(img)[0],np.shape(img)[1]])

# denoiser

denoiser = MSATdenoise(img,'db4',5)
denoised_img = denoiser.denoised

#plotting
fig = plt.figure(figsize=(12, 3))
ax = fig.add_subplot(1, 2, 1)
ax.imshow(img, interpolation="nearest", cmap=plt.cm.gray)
ax = fig.add_subplot(1, 2, 2)
ax.imshow(denoised_img, interpolation="nearest", cmap=plt.cm.gray)
fig.tight_layout()
plt.show()
```

## License
[MIT](https://choosealicense.com/licenses/mit/)
