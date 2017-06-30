#   FULLY CONVOLUTIONAL MODEL 
#
#   DESCRIPTION 
#       This scripts is able to train a fully convulational network or test it with new datagen
#       Set PHASE = "train" for model creation, training, and saving
#       Set PHASE = "test" to test the model on the test_set_images 
#           The first part of testing augment the test set with a factor of 32
#               - Creates 4 rotated versions of the test images
#               - Creates 4 rotated versions of the transpose of the test images
#               - Crops four 400*400 patches per rotated images 
#           The second part of testing predicts each rotated image
#           The third part is a post process phase during which for each original image the 32 predicted are manipulated to 
#           get back one prediction image of 608*608:
#               - The four patches are concatenated and only the max is taken from the overlapping regions producing 8 rotated images
#               - 8 rotated images are mapped back together using rotation and element-wise product 
#               - The resultant prediction is thresholded using the median 
#               - Then this prediction is convolved for line enhancement using a cross kernel  
#
#   DEPENDENCIES
#       Core
#           os, sys, getopt
#           
#       Graphic/images
#           matplotlib, PIL
#           
#       Machine Learning
#           scipy, tensorflow, keras 
#           
#       Custom 
#           mask_to_submission
#       
#   CONSTANTS 
#       PHASE ("train", "test")
#       SUBMISSION_FILE defaults to 'submission_fcn.csv'
#       PREDICTION_FOLDER defaults to 'predictions_fcn'
#       
#   
#   CONSOLE ARGUMENTS
#       model defaults to final_fcn_model.h5 
#   
from __future__ import division, print_function, absolute_import

import os
import sys, getopt

import matplotlib.image as mpimg
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 


from PIL import Image
import scipy
import scipy.signal as scp
import skimage.transform as ski

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape, Cropping2D, UpSampling2D
from keras.layers import Convolution2D, MaxPooling2D, merge, Input, Deconvolution2D
from keras.utils import np_utils
from keras.constraints import maxnorm
from keras.optimizers import SGD, Adam  
from keras.callbacks import ReduceLROnPlateau,TensorBoard
from keras import backend as K
from keras.models import load_model
from keras.regularizers import l2

# Set defualt tensor order to tf == [:,:,channels]
ordering = "tf" 
K.set_image_dim_ordering(ordering)

# Custom file from the course repo 
import mask_to_submission as sub

# CONSTANTS 
PHASE = "train"
model_string = "final_fcn_model.h5"

# training set directories
root_dir = "training/"
image_dir = root_dir + "images/"
gt_dir = root_dir + "groundtruth/" 

# prediction folder and file
SUBMISSION_FILE = 'submission_fcn.csv'
PREDICTION_FOLDER = 'predictions_fcn'


# TERMINAL ARGUMENTS 
def main(argv):
   phase = "train"
   model_string = "final_fcn_model.h5"
   try:
      opts, args = getopt.getopt(argv,"hp:m:",["phase=","model="])
   except getopt.GetoptError:
      print('run_FCN.py -p <phase> -m <model>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print('run_FCN.py -p <phase> -m <model>')
         sys.exit()
      elif opt in ("-p", "--phase"):
         phase = arg
      elif opt in ("-m", "--model"):
         model_string = arg
   return phase, model_string 

if __name__ == "__main__":
   PHASE, model_string = main(sys.argv[1:])

# FCN KERAS MODEL CONSTRUCTION
def build_fcn(X):  
    #
    #   DESCRIPTION
    #       KERAS FCN DEFINITION
    #       Using the shape of the input to setup the input layer we create a FCN with 2 skips 
    #       
    #   INPUTS
    #       X [number_of_images, 400, 400, channels] 
    #
    #   OUTPUTS 
    #       model uninstantiated Keras model 
    #
    img_rows, img_cols = 400, 400
    inputs = Input(shape=X.shape[1:])
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Convolution2D(32, 4, 4, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(64, 4, 4, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv3) # 50 50 

    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv4)  # 25 25

    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv5)  
    drop3 = Dropout(0.5)(pool5) 
    
    convpool3 = Convolution2D(60, 1, 1, activation='relu', border_mode='same')(pool3)
    convpool4 = Convolution2D(60, 1, 1, activation='relu', border_mode='same')(pool4)
    convdrop3 = Convolution2D(60, 1, 1, activation='relu', border_mode='same')(drop3)

    drop3x5 = UpSampling2D(size=(5, 5))(convdrop3)
    croppeddrop3x5 = Cropping2D(((5,5),(5,5)))(drop3x5) # 50 50
    pool4x2 = UpSampling2D(size=(2, 2))(convpool4) # 50 50
    fuse2 = merge([convpool3, pool4x2, croppeddrop3x5], mode='concat', concat_axis=-1) # 50 50 4224
    upscore3 = UpSampling2D(size=(8, 8))(fuse2) # F 8s 
    convscore3 = Convolution2D(1, 1, 1, activation='sigmoid')(upscore3)
    
    # Instantiate Model object 
    model = Model(input=inputs, output=convscore3)

    sgd = SGD(lr=1e-5, decay=2, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss=pixel_wise_loss, metrics=['mean_squared_error'])  

    #model.compile(loss='mean_squared_error', optimizer=sgd)
              
    return model

## CUSTOM LOSS FUNCTION 
def pixel_wise_coef(y_true, y_pred):
    #
    #   DESCRIPTION
    #       Computed a pixel wise distance from the true labels, 
    #       sum the absolute value and divide by the number of pixels to give a ratio 
    #   
    #   INPUTS 
    #       y_true, y_pred     
    #   OUTPUTS
    #       pixel_wise_coef between 0 and 1
    #
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    summed_error = tf.reduce_sum(tf.abs(y_true_f - y_pred_f)) 
    return tf.div(summed_error, 160000)


def pixel_wise_loss(y_true, y_pred):
    return pixel_wise_coef(y_true, y_pred)


def F1(tested, truth):
    tested = (tested-0.5)*2
    truth = (truth-0.5)*2
    truth[truth<=0] = -1.
    truth[truth>0] = 1.
    res  = tested+truth
    true_pos = np.size(np.where(res==2))/2.
    pos = np.size(np.where(truth ==1))/2.
    found_pos = np.size(np.where(tested ==1))/2.
    precision = true_pos/found_pos
    recall = true_pos/pos
    F1 = 2.*precision*recall/(precision+recall)
    return F1





# PRE-PROCESSING FUNCTIONS 

# TRAINING SET FUNCTIONS 
def get_images(images_directory, groundtruths_directory, num_images): 
    #
    #   DESCRIPTION 
    #       Loads each training image and its ground truth and creates tensors [numImages, 400, 400, 3]
    #   
    #   INPUTS 
    #       images_directory path to training images directory  
    #       groundtruths_directory path to the groundtruth images directory
    #       num_images number of images to load 
    #       
    #   OUTPUTS
    #       images, ground_truth two tensors 
    #
    images = []
    ground_truth = [] 
    for i in num_images:
        image_id = "satImage_%.3d" % i
        image_filename = image_id + ".png"
        image_path = images_directory + image_filename;
        groundtruth_image_path = groundtruths_directory + image_filename;

        if ((os.path.isfile(image_path))&(os.path.isfile(groundtruth_image_path))):
            print ('Loading ' + image_filename) 
            loaded_image = mpimg.imread(image_path)
            loaded_gt_image = mpimg.imread(groundtruth_image_path) 

            if ordering == "th":
                loaded_image = np.rollaxis(loaded_image,2) 

            images.append(loaded_image) 
            ground_truth.append(loaded_gt_image)
        else:
            print ('File ' + image_path + ' does not exist')

    return images, ground_truth
      
def extend_dataset(images, ground_truths):
    #
    #   DESCRIPTION 
    #       Doubles the dataset by transposing each image and its groundtruth and concatening it to the input dataset
    #
    #   INPUTS
    #       images [number_of_images, :,:, channels]
    #       ground_truths [number_of_images, :,:]
    #   
    #   OUTPUTS
    #       images, ground_truths concatenated with their transpose 
    num_images = len(images);
    for i in range(0, num_images+1): 
        if (ordering=="tf"):
            new_image = images[i].transpose([1,0,2])  
        else:
            new_image = images[i].transpose([0,2,1]) 

        new_gt = ground_truths[i].transpose([1,0])  
        images.append(new_image)
        ground_truths.append(new_gt)
        print("extend_dataset: distorted image %d" % i)
    return images, ground_truths

def mk_rotations(img):
    #
    #   DESCRIPTION
    #       This function create 8 roatation image fro an input image 4 rotation from the raw image and 4 rotation form the transposed 
    #   
    #   INPUTS
    #       img np.array 
    #       
    #   OUTPUTS
    #       rotated_image_img, img90, img180, img270, imgT, imgT90, imgT180,imgT270
    #
    #
    img90 = np.rot90(img)
    img180 = np.rot90(img,k=2)
    img270 = np.rot90(img,k=3)
    imgT = np.zeros(img.shape)
    if np.size(img.shape)>2:
        for i in range(3):
            imgT[:,:,i] =img[:,:,i].T
    else:
        imgT = img.T
    imgT90 = np.rot90(imgT)
    imgT180 = np.rot90(imgT, k=2)
    imgT270 = np.rot90(imgT, k=3)
    return img, img90, img180, img270, imgT, imgT90, imgT180,imgT270



def save_rotated_test_images():
    #
    #   DESCRIPTION 
    #       This function rotates the test image and create four patches of 400 * 400
    #       It then saves those 32 images in the test_set_images folder of each image 
    #   
    #
    
    # Loop over all images 
    for i in range(1,51):
        # Load image
        image = mpimg.imread('test_set_images/test_'+str(i)+'/test_'+str(i)+'.png')
        rotations = mk_rotations(image)
        rota_count = 0
        for rotation in rotations:
            patches = make_4_patch(rotation)
            patch_count = 0
            for patch in patches:
                patch = format_image(patch)
                Image.fromarray(patch).save('test_set_images/test_'+str(i)+'/Test_'+str(i)+'_rota'+str(rota_count)+'_patch'+str(patch_count)+'.png')
                patch_count += 1
            rota_count+=1


        print('Writing image ',i)

def make_4_patch(img, N1=400,N2=400):
    # 
    #   DESCRIPTION
    #       Creates 4 N1*N2 patches from an original image 
    # 
    #   INPUTS 
    #       img np.array
    #       N1  patch height defaults to 400
    #       N2 patch width
    #       
    #   OUTPUTS
    #       patch_top_left, patch_bottom_left, patch_top_right, patch_bottom_right
    #           4 patches np.array of [N1,N2,:] 
    # 
    sh = np.shape(img)

    if np.size(sh) ==2:
        n1,n2 = sh
        patch_top_left = img[:N1,:N2]
        patch_bottom_left = img[:N1,n2-N1:]
        patch_top_right = img[n1-N1:,:N2]
        patch_bottom_right = img[n1-N1:,n2-N2:]
    else:
        n1,n2,n = sh
        patch_top_left = img[:N1,:N2,:]
        patch_bottom_left = img[:N1,n2-N1:,:]
        patch_top_right = img[n1-N1:,:N2,:]
        patch_bottom_right = img[n1-N1:,n2-N2:,:]

    return patch_top_left, patch_bottom_left, patch_top_right, patch_bottom_right
    
def rebuild_4_patches(patch_top_left, patch_bottom_left, patch_top_right, patch_bottom_right,n1=608, n2 = 608):
    #
    #   DESCRIPTION
    #       Builds an image back grom 4 patches 
    #       Using max on the overlapping regions 
    #
    #   INPUTS 
    #       Patches 
    #       n1,n2 numeric output size 
    #       
    #   OUTPUTS
    #       res np.array rebuilded image [n1,n2,:]
    #       
    sh = np.shape(patch_top_left)
    res = []
    if np.size(sh) == 3:
        if sh[2] == 1:
            patch_top_left = patch_top_left[:,:, 0]
            patch_bottom_left = patch_bottom_left[:,:, 0]
            patch_top_right = patch_top_right[:,:, 0]
            patch_bottom_right = patch_bottom_right[:,:, 0] 
            sh = np.shape(patch_top_left)
        else:
            N1,N2,N3 = np.shape(patch_top_left)
            build = np.zeros((4,n1,n2,N3))
            build[0,:N1,:N2,:] = patch_top_left
            build[1,:N1,n2-N1:,:] = patch_bottom_left
            build[2,n1-N1:,:N2,:] = patch_top_right
            build[3,n1-N1:,n2-N2:,:] = patch_bottom_right
            res = np.max(build,0)


    if np.size(sh) ==2:
        N1,N2 = np.shape(patch_top_left)
        build = np.zeros((4,n1,n2))
        build[0,:N1,:N2] = patch_top_left
        build[1,:N1,n2-N1:] = patch_bottom_left
        build[2,n1-N1:,:N2] = patch_top_right
        build[3,n1-N1:,n2-N2:] = patch_bottom_right
        res = np.max(build,0)

    return res 

def predict_batch_test_images (model, batch_size = 1, max_image = 50):
    #
    #   DESCRIPTION 
    #       Generator function batching each of the 32 mapped image from one test image 
    #       Once the image have been loaded they are predicted using the specified Keras Model 
    #       Once predicted the generator finally yields the batch to be treated in a for loop in predict_and_rebuild
    #       
    #   INPUTS 
    #       model keras model 
    #       batch_size set to 1 
    #       max_image the max number of image loaded (50 test set)
    #
    #   OUTPUTS
    #       yield predictions a np.array of [:, 400, 400, 1]
    #
    images = np.zeros(shape=[8*4, 400, 400, 3], dtype=float) 
    
    for i in range(1,max_image+1):
        count = 0
        for rota_count in range(8):
            for patch_count in range(4):
                images[count, :,:,:] = mpimg.imread('test_set_images/test_'+str(i)+'/Test_'+str(i)+'_rota'+str(rota_count)+'_patch'+str(patch_count)+'.png')
                count += 1
        
        if (count == 32):
            preds = model.predict(images, batch_size = batch_size, verbose=1)
            yield preds

def predict_and_rebuild (model):
    #
    #   DESCRIPTION 
    #       MAIN PREDICTION FUNCTION 
    #       From a model loads all the test images using a batch generator called predict_batch_test_images
    #       It receives the prediction from that function and then rebuilds the the 8 rotated images from each of their 4 patch 
    #       Hence reducing the set from 32 to 8. 
    #       Images are saved in the specified folder. 
    #       The last reduction to only 1 final prediction image will be made by post_process().
    #       
    #   INPUTS
    #       model keras model 
    #       
    #
    folder = PREDICTION_FOLDER
    image_count = 1
    for predictions in predict_batch_test_images(model,1): 
        print("predict_and_rebuild: Predicted all rotations of image (count: "+str(image_count+1)+") - now reconstructing from patches.")
        count = 0 
        for rota_count in range(8): 
            reconstructed = rebuild_4_patches(predictions[count,:,:,0],predictions[count+1,:,:,0],predictions[count+2,:,:,0],predictions[count+3,:,:,0]) 
            reconstructed = format_image(reconstructed) 
          
            print('predict_and_rebuild: Reconstruction done. Saved at '+folder+'/prediction_' + '%.3d' % image_count  + '_rota'+str(rota_count)+'.png')
            Image.fromarray(reconstructed).save(folder+'/prediction_' + '%.3d' % image_count  + '_rota'+str(rota_count)+'.png')
            
            count += 4

        image_count += 1
 
# POST PROCESSING FUNCTIONS 
def post_process(nimage = 50, threshold_type = "median", convolution_patch_size = 8, kernel_size = 5):
    #
    #   DESCRIPTION
    #       This function is called after all predictions are made and patches are merged for each rotated image
    #       Images go through the following process: 
    #           - all the 8 rotations are merged back together forming one perdiction image 
    #           - it is median thresholded
    #           - and finally it is convolved by a cross kernel 
    #       Image is saved 
    #       All images are used to produce the final submission file 
    #       
    #    INPUTS 
    #       nimage number of image to process 
    #       threshold_type either median mean or percentile
    #       convolution_patch_size, kernel_size options of the cross kernel convolution 
    #
    image_names = []
    folder = PREDICTION_FOLDER
    submission_file = SUBMISSION_FILE 
    for i in range(1,nimage+1):
        # Load the rotated images 
        image_names.append(folder+'/prediction_' + '%.3d' % i  + '.png')
        rot0 =mpimg.imread(folder+'/prediction_' + '%.3d' % i  + '_rota0.png')
        rot1 =mpimg.imread(folder+'/prediction_' + '%.3d' % i  + '_rota1.png')
        rot2 =mpimg.imread(folder+'/prediction_' + '%.3d' % i  + '_rota2.png')
        rot3 =mpimg.imread(folder+'/prediction_' + '%.3d' % i  + '_rota3.png')
        rot4 =mpimg.imread(folder+'/prediction_' + '%.3d' % i  + '_rota4.png')
        rot5 =mpimg.imread(folder+'/prediction_' + '%.3d' % i  + '_rota5.png')
        rot6 =mpimg.imread(folder+'/prediction_' + '%.3d' % i  + '_rota6.png')
        rot7 =mpimg.imread(folder+'/prediction_' + '%.3d' % i  + '_rota7.png')
        
        # Get the unrotated image 
        unrot =  unrotate(rot0, rot1, rot2, rot3, rot4, rot5, rot6, rot7)
        unrot = threshold_image_by(unrot, threshold_type) 

        # Cross Kernel Convolution 
        # unrot = isolate(unrot,convolution_patch_size, kernel_size) 

        # Format prediction image and save 
        unrot = format_image(unrot)
        Image.fromarray(unrot).save(folder+'/prediction_' + '%.3d' % i  + '.png')

    # Once all prediction images have been reconstructed build the submission file     
    sub.masks_to_submission(submission_file, *image_names)

def threshold_image_by (img, type ="median", percentile = 70, min_val = 0, max_val = 1):
    #   
    #   DESCRIPTION
    #       This function threshold an image using three different threshold type:
    #           - "median"
    #           - "percentile"
    #           - "mean"
    #           
    #   INPUTS
    #       img np.array image to be thresholded 
    #       type string 
    #       percentile numeric used if type == "percentile"
    #       min_val min value 
    #       max_val max value 
    #       
    #   OUTPUTS
    #       thresholded_img
    #   
    threshold = np.mean(img)
    if type == "median":
        threshold = np.median(img)
    elif type == "percentile":
        threshold = np.percentile(img, 70)


    thresholded_img = np.copy(img)
    thresholded_img[np.where(img<=threshold)] = min_val
    thresholded_img[np.where(img>threshold)] = max_val

    return thresholded_img

# IMAGE PROCESSING FUNCTIONS 
def format_image(img):
    #   DESCRIPTION
    #       Transforms an image to be coded in uint8 from 0 to 255 in order to be saved by PIL.Image.save
    #   
    #   INPUTS
    #       img image to format 
    #       
    #   OUTPUTS
    #       scaled and formated imaged 
    #     
    img = img - np.min(img)
    img = img*255/np.max(img)
    return img.astype(np.uint8)

def isolate(img, size, size_ker):

    Id = -0.5*np.eye(size_ker)
    h = np.zeros((size_ker,size_ker))
    
    h[size_ker/2,:] = 1
    h[:,size_ker/2] = 1
 
    h[size_ker/2,size_ker/2] = 1
    
    h = h/np.float(np.sum(h))
    img = img/np.float(np.max(img))
    newimg = np.copy(img)*0
    n1,n2 = np.shape(img)
    N1,N2 = int(n1/size),int(n2/size)
    grid = np.zeros((N1,N2))
    print(np.shape(img), N1, size)
    
    for  i in range(N1):
        for j in range(N2):
            grid[i,j] = np.mean(img[i*(size):(i+1)*(size),j*(size):(j+1)*(size)])
    grid = (grid-0.5)*2
    newgrid = np.copy(grid)*0
    newgrid = scp.convolve2d(grid,h,mode = 'same', boundary = 'wrap')
    for  i in range(N1):
        for j in range(N2):
            newimg[i*(size):(i+1)*(size),j*(size):(j+1)*(size)] = newgrid[i,j] 
    t = 0
    newimg2 = np.copy(newimg)*0
    newimg2[newimg<=t]= -1
    newimg2[newimg>t] = 1
    
    return newimg2

def unrotate(rot0, rot1, rot2, rot3, rot4, rot5, rot6, rot7):
    #
    #   DESCRIPTION 
    #       Functions that merges the 8 mapped images as described in the beginning of the file back to the original format
    #       Uses element wise product  
    #
    #
    unrot = np.copy(rot0)
    unrot*=np.rot90((rot1),k=3)
    unrot*=np.rot90((rot2),k=2)
    unrot*=np.rot90((rot3),k=1) 

    unrot*=(rot4.T)
    unrot*=np.rot90((rot5),k=3).T
    unrot*=np.rot90((rot6),k=2).T 
    unrot*=np.rot90((rot7),k=1).T 
    
    return unrot

##                      ##
##                      ##
##      EXECUTION       ##
##                      ##
##                      ##

if PHASE == "train":
    # Load Data
    num_classes = 2 
    num_channels = 3 

    num_images = 100  

    images_batch, groundtruth_batch = get_images(image_dir,gt_dir, range(1,num_images+1))
    images_batch, groundtruth_batch = extend_dataset(images_batch, groundtruth_batch) 
    
    # Rename 
    X = np.array(images_batch)
    Y = np.array(groundtruth_batch)

    # resize to add one channel to ground truth
    Y_full = np.zeros((len(Y), Y.shape[1], Y.shape[2], 1))
    Y_full[:,:,:,0] = Y

    Y = Y_full 

    # Instantiate model 
    batch_size = 1
    model = build_fcn(X)

    # Start learning  
    nb_classes = 2
    nb_epoch = 10 

  
    # model = load_model(model_string, custom_objects={'pixel_wise_loss': pixel_wise_coef, "pixel_wise_coef": pixel_wise_coef})
    
    # Create a random image generator for the training set
    datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0.05,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.05,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,  # randomly flip images
                vertical_flip=True)  # randomly flip images

    # Compute quantities required for featurewise normalization 
    datagen.fit(X)

    # Reduces learning rate if mean_squared_error reaches a plateau for more than one epoch
    reduce_lr = ReduceLROnPlateau(monitor='mean_squared_error', factor=0.2, verbose=1,
                  patience=1, min_lr=1e-13) 
    
    model.fit_generator(datagen.flow(X, Y,
                        batch_size=batch_size),
                        samples_per_epoch=X.shape[0],
                        nb_epoch=nb_epoch,
                        callbacks=[reduce_lr])

    #model.fit(X,Y,batch_size=batch_size, nb_epoch=nb_epoch, callbacks=[reduce_lr, tensorboard]) 

    # # evaluate the model
    # scores = model.evaluate(X, Y)
    # print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    # saves a HDF5 file 'my_model.h5'
    model.save('newly_trained_fcn.h5')  
    

if PHASE == "test": 
    # Create a directory for predictions if does not exist 
    if not os.path.exists(PREDICTION_FOLDER):
        os.makedirs(PREDICTION_FOLDER)

    # Load the keras model
    model = load_model(model_string, custom_objects={'pixel_wise_loss': pixel_wise_coef, "pixel_wise_coef": pixel_wise_coef})
    
    # Expand the test set with 8 rotation * 4 patches (32x)
    save_rotated_test_images()

    # Predict and rebuilt only one image for each of the 8 rotation
    predict_and_rebuild(model) 

    # Finalize merging of the rotated images threshold it and convolve them using a cross kernel 
    # - save prediction images
    # - also saves the csv file 
    post_process()






