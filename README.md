# LumiGAN
Trained CNN for feature extraction of luminescence images and GAN for reconstruction of luminescence images are stored in pickle files under LumiGAN/Models.

defect_localisation.py -->  Run localisation algorithm on inputted dataset using GAN model

loss_analysis.py --> Run efficiency loss estiamtation from set of images/defect location using GAN and CNN models

model_training.py --> Train GAN models


CNN-related training and fine-tuning can be done in https://github.com/acdc-pv-unsw/LumiNet

Packages required are shown in requirement.txt (pip install requirement.txt), and models with the trained CNN and without the ML regression are also saved ("_noML") for back-compatibility issues with sklearn.

# Paper: Automated efficiency loss analysis by luminescence image reconstruction using generative adversarial networks
https://doi.org/10.1016/j.joule.2022.05.001

Identifying solar cell efficiency shortfalls in production lines is crucial to troubleshoot and optimize manufacturing processes. With the adoption of luminescence imaging as a key end-of-line characterization tool, a wealth of information is available to evaluate cell performance and classify defects, suitable for user input-free deep-learning analysis. We propose an automated reconstruction method, based on state-of-the-art generative adversarial networks, to remove defective regions in luminescence images. The reconstructed and original images are compared to estimate the efficiency loss. The method is validated on intentionally damaged cells by reconstructing defect-free images and then predicting the efficiency loss. The method can differentiate between different types of defects and pinpoint the defects that lead to the highest efficiency shortfall, enabling manufacturers to troubleshoot production lines in a fast and cost-effective manner. The proposed approach unlocks the potential of luminescence imaging as an effective end-of-line characterization tool.
