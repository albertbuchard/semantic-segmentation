import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import wave_transform as mw
import tools
import tools_reg
import warnings
import pyfits as pf
from PIL import Image
import scipy
import mask_to_submission as sub
import scipy.signal as scp
import skimage.transform as ski
warnings.simplefilter("ignore")

## Normalises images for saving.
def format_image(img):
'''
INPUTS:
    img: a 2D or 3D array image
OUTPUTS:
    img_reg: The same, just normalised for saving
'''
    img = img*255/np.max(img)
    return img.astype(np.uint8)

## Performs kernel convolution on images to try and remove false detections
def isolate(img, size, size_ker):
'''
INPUTS:
    img: predictions as a 2D array
    size: int, size of the rebining (should correspond to the resolution of prediction)
    size_ker: int, size of the convolution kernel
OUTPUTS:
    res: a cleaned prediction as a 2D array with the same size as its input
'''

    ## Definition of the kernel
    h = np.zeros((size_ker,size_ker))
    h[size_ker/2,:] = 1
    h[:,size_ker/2] = 1
    h[size_ker/2,size_ker/2] = 1
    h = h/np.float(np.sum(h))
    ## Normalisation of the image
    img = img/np.float(np.max(img))

    ## Initialisations
    newimg = np.copy(img)*0
    n1,n2 = np.shape(img)
    N1,N2 = n1/size,n2/size
    grid = np.zeros((N1,N2))
    
    ## Regriding to match the prediction resolution
    for  i in range(N1):
        for j in range(N2):
            grid[i,j] = np.mean(img[i*(size):(i+1)*(size),j*(size):(j+1)*(size)])
    grid = (grid-0.5)*2
    newgrid = np.copy(grid)*0
    ## Kernel convolution
    newgrid = scp.convolve2d(grid,h,mode = 'same', boundary = 'symm')
    ## Degriding to come back to the original shape of the image
    for  i in range(N1):
        for j in range(N2):
            newimg[i*(size):(i+1)*(size),j*(size):(j+1)*(size)] = newgrid[i,j]

    ## Thresholding
    t = 0
    newimg2 = np.copy(newimg)*0
    newimg2[newimg<=t]= -1
    newimg2[newimg>t] = 1
    
    return (newimg2+1)/2.

## Recombines predictions from rotated images
def unrotate(rot0, rot1, rot2, rot3, rot4, rot5, rot6, rot7):
    '''
INPUTS:
    rot0 ... rot7: All 2D arrays that should all be 90n degrees rotations and translation of rot0
OUTPUTS:
    unrot: The logical combination of the inputs. It is a 2D array.
'''
    unrot = np.copy(rot0)
    unrot*=np.rot90((rot1),k=3)
    unrot*=np.rot90((rot2),k=2)
    unrot*=np.rot90((rot3),k=1)
    rot4t = np.zeros(rot0.shape)
    rot5t = np.zeros(rot0.shape)
    rot6t = np.zeros(rot0.shape)
    rot7t = np.zeros(rot0.shape)

    unrot*=(rot4.T)
    unrot*=np.rot90((rot5),k=3).T
    unrot*=np.rot90((rot6),k=2).T
    t = 4
    unrot*=np.rot90((rot7),k=1).T#
    unrot2 = np.copy(unrot*0)
    
    return unrot



########### Test ##########
## Produces the submission fila along with the final predictions
def mk_submission(size, size_ker, submission_file):
'''
INPUTS:
    size: int, the size of the resolution of the prediction (see isolate)
    size_ker: int, size of the convolution kernel (see isolate)
    submission_file: string, name of the submission file where to write the results.
OUTPUTS:
    None
'''
    image_name = []
    ## File where to save predictions

    ## Folder containingthe predictions
    folder = 'predictions'

    ## Iteration over each image of the test set
    for i in range(1,51):
        print(i)
        ## Name of the final prediction file to save (output)
        image_name.append(folder+'/prediction_' + '%.3d' % i  + '.png')
        ## loading the rotated images
        rot0 =mpimg.imread(folder+'/prediction_' + '%.3d' % i  + '_rota0.png')
        rot1 =mpimg.imread(folder+'/prediction_' + '%.3d' % i  + '_rota1.png')
        rot2 =mpimg.imread(folder+'/prediction_' + '%.3d' % i  + '_rota2.png')
        rot3 =mpimg.imread(folder+'/prediction_' + '%.3d' % i  + '_rota3.png')
        rot4 =mpimg.imread(folder+'/prediction_' + '%.3d' % i  + '_rota4.png')
        rot5 =mpimg.imread(folder+'/prediction_' + '%.3d' % i  + '_rota5.png')
        rot6 =mpimg.imread(folder+'/prediction_' + '%.3d' % i  + '_rota6.png')
        rot7 =mpimg.imread(folder+'/prediction_' + '%.3d' % i  + '_rota7.png')

        ## Recombinig the rotated images
        unrot =  unrotate(rot0, rot1, rot2, rot3, rot4, rot5, rot6, rot7)

        ## Kernel convolution
        unrot = isolate(unrot,size, size_ker)

        unrot = format_image(unrot)
        
        Image.fromarray(unrot).save(folder+'/prediction_' + '%.3d' % i  + '.png')

    ## Saving the submission
    sub.masks_to_submission(submission_file, *image_name)
    return 'sbravaradjan'
