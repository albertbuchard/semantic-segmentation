import numpy as np
import scipy.signal as cp
import matplotlib.pyplot as plt
import scipy.ndimage.filters as sc
import skimage.transform as ski
import scipy.ndimage.filters as med


def mr_filter(alpha, niter, k, levels, sigma, soft = False):

    lvl, n1,n2 = np.shape(alpha)
    
    M = np.zeros((lvl,n1,n2))+0.
    M[-1,:,:] = 1

    th = np.ones((lvl,n1,n2))*k
    
    th[0,:,:] = th[0,:,:]+1.
    th[-1,:,:] = 0

    th = th*levels*sigma
    alphanew = 0
    i =0

    R= 0

    M[np.where(np.abs(alpha)-np.abs(th) > 0)] = 1

    alphanew = M*alpha

    return iuwt(alphanew)


def symmetrise(img, size):

    n3, n4 = np.shape(img)
    n1,n2 = size
    img[:(n3-n1)/2, :] = np.flipud(img[(n3-n1)/2:(n3-n1),:])
    img[:,:(n4-n2)/2] = np.fliplr(img[:,(n4-n2)/2:(n4-n2)])
    img[(n3+n1)/2:,:] = np.flipud(img[n1:(n3+n1)/2,:])
    img[:,(n4+n2)/2:] = np.fliplr(img[:,n2:(n4+n2)/2])

    return img


def fft_convolve(X,Y, inv = 0):
    
    XF = np.fft.rfft2(X)
    YF = np.fft.rfft2(Y)
#    YF0 = np.copy(YF)
#    YF.imag = 0
#    XF.imag = 0
    if inv == 1:
 #       plt.imshow(np.real(YF)); plt.colorbar(); plt.show()
        YF = np.conj(YF)

    SF = XF*YF
    
    S = np.fft.irfft2(SF)
    n1,n2 = np.shape(S)

    S = np.roll(S,-n1/2+1,axis = 0)
    S = np.roll(S,-n2/2+1,axis = 1)

    return np.real(S)

    
def wave_transform(img, lvl, Filter = 'Bspline', newwave = 1, convol2d = 0):

    mode = 'nearest'
    
    lvl = lvl-1
    sh = np.shape(img)
    if np.size(sh) ==3:
        mn = np.min(sh)
        wave = np.zeros([lvl+1,sh[1], sh[1],mn])
        for h in np.linspace(0,mn-1, mn):
            if mn == sh[0]:
                wave[:,:,:,h] = wave_transform(img[h,:,:],lvl+1, Filter = Filter)
            else:
                wave[:,:,:,h] = wave_transform(img[:,:,h],lvl+1, Filter = Filter)
        return wave
    n1 = sh[1]
    n2 = sh[1]
    
    if Filter == 'Bspline':
        h = [1./16, 1./4, 3./8, 1./4, 1./16]
    else:
        h = [1./4,1./2,1./4]
    n = np.size(h)
    h = np.array(h)
    
    if n+2**(lvl-1)*(n-1) >= np.min([n1,n2])/2.:
        lvl = np.int_(np.log2((n1-1)/(n-1.))+1)

    c = img
    ## wavelet set of coefficients.
    wave = np.zeros([lvl+1,n1,n2])
  
    for i in np.linspace(0,lvl-1,lvl):
        newh = np.zeros((1,n+(n-1)*(2**i-1)))
        newh[0,np.int_(np.linspace(0,np.size(newh)-1,len(h)))] = h
        H = np.dot(newh.T,newh)

        ######Calculates c(j+1)
        ###### Line convolution
        if convol2d == 1:
            cnew = cp.convolve2d(c, H, mode='same', boundary='symm')
        else:
            cnew = sc.convolve1d(c,newh[0,:],axis = 0, mode =mode)

            ###### Column convolution
            cnew = sc.convolve1d(cnew,newh[0,:],axis = 1, mode =mode)

 
      
        if newwave ==1:
            ###### hoh for g; Column convolution
            if convol2d == 1:
                hc = cp.convolve2d(cnew, H, mode='same', boundary='symm')
            else:
                hc = sc.convolve1d(cnew,newh[0,:],axis = 0, mode = mode)
 
                ###### hoh for g; Line convolution
                hc = sc.convolve1d(hc,newh[0,:],axis = 1, mode = mode)
            
            ###### wj+1 = cj-hcj+1
            wave[i,:,:] = c-hc
            
        else:
            ###### wj+1 = cj-cj+1
            wave[i,:,:] = c-cnew
 

        c = cnew
     
    wave[i+1,:,:] = c

    return wave

def iuwt(wave, convol2d =0):
    mode = 'nearest'
    
    lvl,n1,n2 = np.shape(wave)
    h = np.array([1./16, 1./4, 3./8, 1./4, 1./16])
    n = np.size(h)

    cJ = np.copy(wave[lvl-1,:,:])
    
    
    for i in np.linspace(1,lvl-1,lvl-1):
        
        newh = np.zeros((1,n+(n-1)*(2**(lvl-1-i)-1)))
        newh[0,np.int_(np.linspace(0,np.size(newh)-1,len(h)))] = h
        H = np.dot(newh.T,newh)

        ###### Line convolution
        if convol2d == 1:
            cnew = cp.convolve2d(cJ, H, mode='same', boundary='symm')
        else:
          cnew = sc.convolve1d(cJ,newh[0,:],axis = 0, mode = mode)
            ###### Column convolution
          cnew = sc.convolve1d(cnew,newh[0,:],axis = 1, mode = mode)

        cJ = cnew+wave[lvl-1-i,:,:]

    return np.reshape(cJ,(n1,n2))

def mk_line(n1,theta):
    x = np.linspace(0,n1-1,n1)
    y = np.copy(x)*0
    count = 0
    for t in x:
        y[count] = t*np.tan(theta*np.pi/180.)
        count +=1

    return x,y-y[-1]/2


def ridgelet(img):
    n1,n2 = np.shape(img)
    lvl = np.int(np.log2(n1))
    rad = ski.radon(img, theta = np.linspace(0,180,n1))
    nr,nt = np.shape(rad)
    rad_ridgelet = np.zeros((lvl,nr,nt))
    for i in np.linspace(0,nt-1,nt):
        rad_ridgelet[:,:,i] = np.reshape(wavelet_1D(rad[:,i],lvl),(lvl,nr))
#    ridgelet = np.zeros((lvl,n1,n2))
    ##Not right
   # for l in np.linspace(0,lvl-1,lvl):
 #       ridgelet[l,:,:] = ski.iradon(rad_ridgelet[l,:,:])
    ridgelet = rad_ridgelet
    return ridgelet

def iridgelet(ridge):
    lvl, nr,nt = np.shape(ridge)
 #   for l in np.linspace(0,lvl-1,lvl):
#        x = ski.radon(ridge[l,:,:],theta = np.linspace(0,180-1,nt))
#        if l ==0:
#            nr,nt = np.shape(x)
#            rad_ridgelet = np.zeros((lvl,nr,nt))
#        rad_ridgelet[l,:,:] = x
    rad_ridgelet = ridge
    rad = np.zeros((nr,nt))
        
    for i in np.linspace(0,nt-1,nt):
        rad[:,i] = iuwt_1D(rad_ridgelet[:,:,i])
    img = ski.iradon(rad)
    return img

def mk_thresh_star(n1,n2):
    dirac = np.zeros((n1,n2))
    dirac[n1/2.,n2/2.] = 1
    ridge_rac = ridgelet(dirac)
    lvl = np.sqrt(np.sum(np.sum(ridge_rac**2,1),1))
    return lvl

def mk_thresh(n1,n2):
    dirac = np.zeros((n1,n2))
    dirac[n1/2.,:] = 1
    ridge_rac = ridgelet(dirac)
    lvl = np.sqrt(np.sum(np.sum(ridge_rac**2,1),1))
    return lvl

def ridgelet_filter(img,k, niter,sigma):
    mx = np.max(img)
    if np.size(img.shape)>2:
        n1,n2,loc = np.shape(img)
        imfil = np.zeros((n1,n2,loc))
        for j in range(loc):
            imfil[:,:,j] = ridgelet_filter(img[:,:,j],k,niter)
        return imfil
            
    n1,n2 = np.shape(img)

    l = np.int(np.log2(n1))
    lvl = mk_thresh(n1,n2)
    ridge = ridgelet(img)
    l,nr1,nr2 = np.shape(ridge)
    th = np.ones((l,nr1,nr2))
    th = np.multiply(th.T,lvl).T*k*sigma
    i =0
    M = th*0
    M[np.where(np.abs(ridge)-np.abs(th) > 0)] = 1


    ridgenew = M*ridge

    
    imfil = iridgelet(ridgenew)
    if np.max(imfil)!=0:
        imfil = imfil*mx/np.float(np.max(imfil))
    imfil[np.where(imfil<0)] = 0
    return imfil

def MAD(x,n=3,fil=1):
        if fil == 1:
            meda = med.median_filter(x,size = (n,n))
        else:
            meda = np.median(x)
        medfil = np.abs(x-meda)
        sh = np.shape(x)
        sigma = 1.48*np.median((medfil))
        return sigma

def iwavelet(ridge):
    lvl,n21,n21 = np.shape(ridge)
    n1 = n21/2.
    radon = np.zeros((n21,n21))
    for i in range(n21):
        radon[i,:] = iuwt_1D(np.reshape(ridge[:,i],(lvl,n21))) 

    return radon

def wavelet_1D(curve, lvl):
    """
    Performs starlet decomposition of an image
    INPUTS:
        img: image with size n1xn2 to be decomposed.
        lvl: number of wavelet levels used in the decomposition.
    OUTPUTS:
        wave: starlet decomposition returned as lvlxn1xn2 cube.
    OPTIONS:
        Filter: if set to 'Bspline', a bicubic spline filter is used (default is True).
        newave: if set to True, the new generation starlet decomposition is used (default is True).
        convol2d: if set, a 2D version of the filter is used (slower, default is 0).
        
    """
    mode = 'nearest'
    
    lvl = lvl-1
    sh = np.shape(curve)
    if np.size(sh) ==2:
        mn = np.min(sh)
        mx = np.max(sh)
        wave = np.zeros([lvl+1, mx,mn])
        for h in np.linspace(0,mn-1, mn):
            if mn == sh[0]:
                wave[:,:,h] = wavelet(curve[h,:],lvl+1)
            else:
                wave[:,:,h] = wavelet(curve[:,h],lvl+1)
        return wave
    n1 = curve.size

    h = [1./16, 1./4, 3./8, 1./4, 1./16]
    h = np.array(h)
    n = np.size(h)
    
    
    if n+2**(lvl-1)*(n-1) >= n1/2.:
        lvl = np.int_(np.log2((n1-1)/(n-1.))+1)

    c = curve
    ## wavelet set of coefficients.
    wave = np.zeros([lvl+1,n1])
  
    for i in np.linspace(0,lvl-1,lvl):
        newh = np.zeros((1,n+(n-1)*(2**i-1)))
        newh[0,np.int_(np.linspace(0,np.size(newh)-1,len(h)))] = h
        H = np.dot(newh.T,newh)

        ######Calculates c(j+1)
        ###### Line convolution

        cnew = sc.convolve1d(c,newh[0,:],axis = 0, mode =mode)

#  hc = sc.convolve1d(cnew,newh[0,:],axis = 0, mode = mode)
 
            
            ###### wj+1 = cj-hcj+1
        wave[i,:] = c-cnew#hc
 

        c = cnew
     
    wave[i+1,:] = c

    return wave

def iuwt_1D(wave):
    """
    Inverse Starlet transform.
    INPUTS:
        wave: wavelet decomposition of an image.
    OUTPUTS:
        out: image reconstructed from wavelet coefficients
    OPTIONS:
        convol2d:  if set, a 2D version of the filter is used (slower, default is 0)
        
    """
    mode = 'nearest'
    
    lvl,n1= np.shape(wave)
    h = np.array([1./16, 1./4, 3./8, 1./4, 1./16])
    n = np.size(h)

    cJ = np.copy(wave[lvl-1,:])
    
    
    for i in np.linspace(1,lvl-1,lvl-1):
        
        newh = np.zeros((1,n+(n-1)*(2**(lvl-1-i)-1)))
        newh[0,np.int_(np.linspace(0,np.size(newh)-1,len(h)))] = h
        H = np.dot(newh.T,newh)

        ###### Line convolution

        cnew = sc.convolve1d(cJ,newh[0,:],axis = 0, mode = mode)
        cJ = cnew+wave[lvl-1-i,:]

    out = cJ
    return out


def epsilon_enhance(img, g, rho, mu, tau):
    alpha0 = ridgelet(img)
    alpha = np.abs(alpha0)
    lvl,na1,na2 = np.shape(alpha)
    n1,n2 = np.shape(img)
    sigma = MAD(img)
    level = mk_thresh(n1,n2)*sigma
    alpha_enhanced = np.copy(alpha)*0
    for i in range(lvl-1):
        alpha_enhanced[i, np.where(alpha[i,:,:]<mu)] = (mu/(alpha[i,np.where(alpha[i,:,:].astype(float)<mu)]))**g
        plt.imshow(np.log10(alpha_enhanced[i,:,:])); plt.title('mu');plt.colorbar(); plt.show()
        alpha_enhanced[i, np.where(alpha[i,:,:]<2*tau*level[i])] = ((alpha[i, np.where(alpha[i,:,:]<2*tau*level[i])].astype(float)-tau*level[i])/(tau*level[i]))*(mu/(tau*level[i]))**g +(2*tau*level[i]-alpha[i, np.where(alpha[i,:,:]<2*tau*level[i])].astype(float))/(tau*level[i]) 
        plt.imshow(np.log10(alpha_enhanced[i,:,:])); plt.title('2sigma');plt.colorbar(); plt.show()
        alpha_enhanced[i, np.where(alpha[i,:,:]<tau*level[i])] = 1.
        plt.imshow(np.log10(alpha_enhanced[i,:,:])); plt.title('1'); plt.colorbar(); plt.show()
        alpha_enhanced[i, np.where(alpha[i,:,:]>=mu)] = (mu/alpha[i,np.where(alpha[i,:,:]>=mu)].astype(float))**rho       
        plt.imshow(np.log10(alpha_enhanced[i,:,:]));plt.title('final'); plt.colorbar(); plt.show()
    alpha_enhanced[-1,:,:] = 1
    return alpha0*alpha_enhanced

def ridgelet_enhancement(img, g, rho, mu, tau):
    alpha = epsilon_enhance(img, g, rho, mu, tau)
    return iridgelet(alpha)



def mMCA(img, niter, k):
    if np.size(img.shape)>2:
        n1,n2,loc = np.shape(img)
        ridgefil = np.zeros((n1,n2,loc))
        starfil = np.zeros((n1,n2,loc))
        for j in range(loc):
            ridgefil[:,:,j], starfil[:,:,j] = mMCA(img[:,:,j],niter,k)
        return ridgefil, starfil
    n1,n2 = np.shape(img)
    Ridge = np.random.randn(n1,n2)*0.00001
    Star = np.random.randn(n1,n2)*0.00001
    sigma = 0.000001#MAD(img)*0.05
    lvl = np.int(np.log2(n1))
    levels = np.multiply(np.ones((lvl,n1,n2)).T,mk_thresh_star(n1,n2)).T

    Star_w = wave_transform(img, lvl)
    Ridge_w = ridgelet(img)
    nr,nr1,nr2 = np.shape(Ridge_w)
    levelr = np.multiply(np.ones((nr,nr1,nr2)).T,mk_thresh(n1,n2)).T
    star_k = np.max(Star_w[:-1]/(levels[:-1]*sigma))
    ridge_k = np.max(Ridge_w[:-1]/(levelr[:-1]*sigma))
    k0 = np.min([star_k,ridge_k])+0.1*np.abs(star_k-ridge_k)
    step = (k0-k)/(niter-5)
    kc = k0
    muS = 1
    muR = 1
    for i in range(niter):
        print(i)
        RS = (img-Ridge-Star)

        Ridge = ridgelet_filter(Ridge+muR*RS, kc+3, 20, sigma)

        arg = Star+muS*RS
 #       sum_arg = np.max(arg)
        alpha_star = wave_transform(arg, lvl)
        Star = mr_filter(alpha_star, 20,kc-4, levels, sigma,soft = 0)

        Ridge[Ridge<0] = 0
        Star[Star<0] = 0

        kc = kc-step
        kc = np.max([kc,k])
    return Ridge, Star








