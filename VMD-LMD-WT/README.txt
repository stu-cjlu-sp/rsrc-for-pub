This is a brief overview of the files.

1. data_generation_VMD_LMD_WT.m: This file generates 12 types of radar signals and applies the VMD_LMD_WT multilayer denoising method to denoise the signals. It then generates CWD time-frequency images as training and testing datasets.

2. VMD_LMD_WT.m: This function performs multilayer signal denoising using a method combing VMD, LMD, and WT.

3. VMD.m: This function performs variational mode decomposition of the signal.

4. LMD.m: This function performs local mean decomposition of signals.

5. FTCWD.m: This function performs Choi-Williams distribution time-frequency transformations on signals.

6. train.py: This script loads the training dataset and trains the model.

7. test.py: This script loads the test dataset and performs the recognition of radar signals on the trained model. Results are finally obtained for the recognition accuracy and the confusion matrix at each signal-to-noise ratio.

8. CNN.py: This script file contains the CNN specific code to create the CNN model.
