# %%--  Imports
#   General
import pandas as pd
import numpy as np

#   Machine learning
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression, Lasso, LinearRegression, ElasticNet, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
#   Customs
from LumiGAN import *
from LumiGAN.utility import LoadObj, SaveObj

# %%-
# %%--  Prep data
workdir = "TEST\\"    #   ADD DIRECTORY PATH - Needs to have match.csv file with path to images and I-V data
Data = Datahandler(workdir=workdir)
seed = np.random.randint(9999)
save = True
Name = str(seed)+"_Baseline"
CNN_data_ratio = 300    #   Edit based on number of cell required for CNN and GAN training
Val_data_ratio = 50     #   Validation dataset number of cells
targetCol = "Eff_std_mmad"

All_df = Data.matchDf.copy(deep=True)
NR_df = All_df.loc[All_df['Labels']!=0] # if 'Labels' don't exist - edit based on Reject(R)/non-Reject(NR) [optional]
R_df = All_df.loc[All_df['Labels']==0]  # if 'Labels' don't exist - edit based on Reject(R)/non-Reject(NR) [optional]
Data.matchDf = NR_df.copy(deep=True)
CNN_df, ML_df = Data.splitData(mlFrac=CNN_data_ratio/len(NR_df),randomSeed=seed)
Data.matchDf = ML_df.copy(deep=True)
Val_df, ML_df = Data.splitData(mlFrac=Val_data_ratio/len(ML_df),randomSeed=seed)
Data.matchDf = NR_df.copy(deep=True)
MLR_df = ML_df.copy(deep=True).append(R_df.loc[R_df[targetCol]>19])    # 19 is criteria to edit [ based on Reject(R)/non-Reject(NR)]
# %%-

# %%--  GAN training
#   Edit the following parameter as required
img_size=256
mask_size=32
channels=1
params = {
    'subset_size':None,
    'test_batch_size':5,
    'batch_size':11,
    'epochs': 50,
    'frac_loss_img':0.15,
    'frac_loss_mask':0.15,
    'frac_loss_pixel':0.7,
    'random_seed':None,
    'print_freq':100,
    'mode':'GAN',
    'set_position':False,
    'mask_mode':'random',
    'transform':None,
    'transform_train':None,
    'generator':{
        'p2_deep':3,
        'p2_offset':6,
        'weights_init':True,
        'Optimizer':'Adam',
        'lr':0.00001,
        'b1':0.9,
        'b2':0.999,
        'eps':1e-8,
        'weight_decay':0,
    },
    'discriminator_img':{
        'p2_deep':3,
        'p2_offset':4,
        'weights_init':True,
        'Optimizer':'Adam',
        'lr':0.000001,
        'b1':0.9,
        'b2':0.999,
        'eps':1e-8,
        'weight_decay':0,
    },
    'discriminator_mask':{
        'p2_deep':3,
        'p2_offset':4,
        'weights_init':True,
        'Optimizer':'Adam',
        'lr':0.001,
        'b1':0.9,
        'b2':0.999,
        'eps':1e-8,
        'weight_decay':0,
    }
}
Data.matchDf = CNN_df.copy(deep=True)
GAN = Gcompute(Data,name=Name+"_GAN_"+str(img_size)+"_"+str(mask_size), save=save)
GAN.prepGDD(img_size=img_size, mask_size=mask_size, channels=channels, params=params)
GAN.trainGDD(params={'test_save_each_epoch':save})
# %%-

# %%--  CNN Loading
model_path = "Models\\LumiNet_VGG_RF_Eff_noML.pkl" #   ADD PATH HERE - Models can be loaded from https://github.com/acdc-pv-unsw/LumiNet
Obj = LoadObj(model_path)
TF_model = Obj['CNN']['model_extractor']
PARAMETERS = Obj['PARAMETERS']
PARAMETERS['SAVEFOLDER']=Data.pathDic['workdir']
PARAMETERS['SAVE']=save
#   Edit following parameters if required.
PARAMETERS['TARGET_COL']="Eff"
PARAMETERS['CNN']['N_EPOCHS']=25
PARAMETERS['CNN']['BATCH_SIZE']=11
PARAMETERS['ML']['MODEL']=('AdaBoost',AdaBoostRegressor(base_estimator = DecisionTreeRegressor(), n_estimators=100, loss='square'))
# %%-

# %%--  Fine tuning
Data.matchDf = CNN_df.copy(deep=True)
Data.matchDf=Data.matchDf.append(R_df)
Ptmodel = Ptcompute(Data,TF_model,name=Name+"_CNN_"+PARAMETERS['CNN']['MODEL'],save=save)
Ptmodel.subset_size = PARAMETERS['CNN']['SUBSET_SIZE']
Ptmodel.batch_size = PARAMETERS['CNN']['BATCH_SIZE']
Ptmodel.split_size = PARAMETERS['CNN']['SPLIT_FRAC']
Ptmodel.n_epochs = PARAMETERS['CNN']['N_EPOCHS']
Ptmodel.CM_fz = PARAMETERS['CNN']['CM_FZ']
Ptmodel.initTraining()
Ptmodel.trainModel(
    Ycol="Labels",
    transform=PARAMETERS['CNN']['TRANSFORM'],
    transformTrain=PARAMETERS['CNN']['TRANSFORM_AUG'],
    randomSeed=seed,
    split_randomSeed=seed,
    comment=""
    )
# %%-
# %%--  Feature extraction
CNN = Ptcompute.freezeCNN(PARAMETERS['CNN']['MODEL'],Ptmodel.model)
Xcols, ML_df = Ptcompute.extractFeature(TF_model,ML_df,PARAMETERS['CNN']['TRANSFORM'],batch_size=PARAMETERS['CNN']['BATCH_SIZE'])
# %%-
# %%--  Machine learning regression
Data.matchDf=ML_df.copy(deep=True)
Skmodel = Skcompute(Data,PARAMETERS['ML']['MODEL'][1],name=Name+"_ML_"+PARAMETERS['ML']['MODEL'][0], save=save)
Skmodel.initTraining()
Skmodel.subset_size = PARAMETERS['ML']['SUBSET_SIZE']
Skmodel.split_size = PARAMETERS['ML']['SPLIT_FRAC']
Skmodel.trainModel(
    Xcols=Xcols,
    Ycol=PARAMETERS['TARGET_COL'],
    predictType='Regression',
    randomSeed=seed,
    comment=""
)
# %%-
# %%--  Extract all features for dataset
Xcols, All_df = Ptcompute.extractFeature(TF_model,All_df,PARAMETERS['CNN']['TRANSFORM'],batch_size=PARAMETERS['CNN']['BATCH_SIZE'])
# %%-
# %%--  Save Models
Models={
    'Generator':GAN.d_prepGDD['model']['generator'],
    'Discriminator_img':GAN.d_prepGDD['model']['discriminator_img'],
    'Discriminator_mask':GAN.d_prepGDD['model']['discriminator_mask'],
    'CNN_classifier': Ptmodel.model,
    'CNN_extract': CNN,
    'ML':Skmodel.model,
    'Xcols':Xcols,
    'seed':seed,
}
if save: SaveObj(Models,workdir+"outputs\\"+Name+"_models.pkl")
# %%-
# %%--  Save objects [optional]
# CNN_d = Ptmodel.classResults[0]
# CNN_d['vocab'] = Ptmodel.vocab
# CNN_d['CM'] = Ptmodel.CM
# Long={
#     'GAN_p':GAN.d_prepGDD,
#     'GAN_t':GAN.d_trainGDD,
#     'CNN':CNN_d,
#     'ML':Skmodel.regResults[0],
# }
# if save: SaveObj(Long,workdir+"outputs\\"+Name+"_all.pkl")
# %%-
