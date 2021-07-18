# UTF-8
# Perform a wavelet denoising method
# Amir Souri (ahsouri@cfa.harvard.edu;ahsouri@gmail.com)
# heavily inspired by MATLAB's wdenoise2()

def signal2wt(image,wtname,level):
    '''
    Applying wavelet transform to the image (signal)
    ARGS: 
        image[m,n] (float), np.array
        wtname (char) -> e.g, 'db2'
        level (int)
    OUTs:
        coeff [level] (list): wavelet approximation(level=0) and details (>0)
    '''
    import pywt
    import numpy as np

    coeffs = pywt.wavedec2(image, wtname,level=level)

    return coeffs

def ThreshSURE(x):
    """
    Threshold based on SURE
    ARGS: 
        x[3,n,m] (float): H,V,D details 
    OUTs:
        Thr[1x1] (float): Noise/signal threshold
    """
    import numpy as np

    x=x.flatten()
    n = np.size(x)
    sx = np.sort(np.abs(x))
    sx2 = sx**2
    n1 = n-2*np.arange(0,n,1)
    n2 = np.arange(n-1,-1,-1)
    cs1 = np.cumsum(sx2,axis=0)
    risk = (n1+cs1+n2*sx2)/n
    ibest = np.argmin(risk)
    thr = sx[ibest]

    return thr

def wtdenoiser(coeffs2,wtname,level):
    '''
    Removing details based on the SURE threshold
    ARGS: 
        image[m,n] (float)
        wtname (char) -> e.g, 'db2'
        level (int)
    OUTs:
        denoised[m,n] (float): denoised image
    '''
    import numpy as np
    import pywt
    from scipy.special import erfcinv
    # varwt
    normfac = -np.sqrt(2)*erfcinv(2*0.75) 
    # finding thresholds for each level
    thr = np.zeros((level,1))
    cfs_denoised = []
    cfs_denoised.append(coeffs2[0]) #approx
    for lev in range(level):
        cfs = coeffs2[lev+1]
        sigmaest = np.median(np.abs(cfs))*(1/normfac)
        thr = ThreshSURE(cfs/sigmaest)
        thr = sigmaest*thr
        cfs_denoised.append(list(pywt.threshold(cfs,thr,'soft'))) #details
    #reconstruct the signal   
    denoised = pywt.waverec2(cfs_denoised, wtname)

    return denoised

def MSATwtdenoiser(data,wtname='db4',level=5):
    '''
    Removing details based on the SURE threshold
    ARGS: 
        data[m,n] (float)
        wtname (char) -> e.g, 'db2'
        level (int)
    OUTs:
        denoised[m,n] (float): denoised image
    '''
    import numpy as np

    data.astype(float)
    # apply wt on the data
    coeff1 = signal2wt(data,wtname,level)
    # denoise
    denoised = wtdenoiser(coeff1,wtname,level)
    #reshape to the 2D shape
    denoised = np.reshape(denoised,(np.shape(data)[0],np.shape(data)[1]))

    return denoised

def example():
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    import numpy as np

    img = np.array(mpimg.imread('steve.jpg'))
    img.astype(float)
    img = np.mean(img,axis=2)
    img = img+np.random.normal(0,100,[np.shape(img)[0],np.shape(img)[1]])
    # denoiser
    denoised_img = MSATwtdenoiser(img)

    #plotting
    fig = plt.figure(figsize=(12, 3))
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(img, interpolation="nearest", cmap=plt.cm.gray)
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(denoised_img, interpolation="nearest", cmap=plt.cm.gray)
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    print('running the example')
    example()
