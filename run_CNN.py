import project2_run_win as prun
import Training_run as TR
import numpy as np


''' This file allows to reproduce the results pushed to kaggle as team Road runners

To run this code, there are two main variables to deal with: RESTORE and GENERATE

GENERATE: if set to one, generate will produce and save the images for the extended training set.
    If this is the first time you run this code, GENERATE has to be set to one. The rotated images will be produced in a matter of minutes

RESTORE: determines whether the code is to be used to train the network or to make predictions.
        If set to False, the network will train itself on the training set and save the model.
        If set to True, Predictions will be made on each image of the test set as well as on the rotated versions that have to be previously generated
            Once this is done, a post-processing step is ran and produces the submission file.

In case the user is interested only in reproducing the submission from the model, we davise to use the following configuration:
RESTORE = True
GENERATE = True
model_name = "./model_8_50epoch_0.86.ckpt"
submission_name = 'submission_reproduce.csv'

If one is interested in running the whole pipeline, we advise to run the routines twice. First using:
RESTORE = False
GENERATE = True
model_name = "./model_8_50epoch_0.86_reproduce.ckpt"
submission_name = 'submission_reproduce.csv' # Does not matter actually on the first run

Second using:
RESTORE = True
GENERATE = False
model_name = "./model_8_50epoch_0.86_reproduce.ckpt"
submission_name = 'submission_reproduce.csv' 

'''


### Most important variable!!!!!!!!!!!!!!!!
##      Set to False to start training the algorithm
##      To use a preexisting model for prediction, set to True
RESTORE = True

## Name of the model to generate if training, or to use if predicting
model_name = "./model_8_50epoch_0.86.ckpt"
GENERATE = False
## Name of the submission file. Necessary if RESTORE == true
submission_name = 'submission_reproduce.csv'




## Here is where the training or prediction is being performed 
TR.main(RESTORE, model_name, GENERATE)
## Post-processing
if RESTORE == True:
    prun.mk_submission(8, 5, submission_name)
