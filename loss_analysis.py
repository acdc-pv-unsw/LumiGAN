# %%--  Imports
#   General
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imageio
import datetime

#   Machine learning
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression, Lasso, LinearRegression, ElasticNet, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from torchvision import transforms
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F

#   Customs
from LumiGAN import *
from LumiGAN.utility import LoadObj, SaveObj, print_dic

# %%-

# %%--  Load models
model_path = "Models\\LumiGAN_model.pkl"  #   ADD PATH HERE - Models can be loaded from https://github.com/acdc-pv-unsw/LumiNet
Models = LoadObj(model_path)
print_dic(Models)

All_df=pd.read_csv("TEST\\match.csv")  #   Load data here with path to images and I-V data
ML=('AdaBoost',AdaBoostRegressor(base_estimator = DecisionTreeRegressor(), n_estimators=100, loss='square'))
CNN_e=Models['CNN_extract']
GAN_g=Models['Generator']
seed=Models['seed']
# %%-
# %%--  Settings
workdir = "TEST\\"    #   ADD FOLDER PATH HERE - Data to be analysed by loss-analysis. Needs to have match.csv with image path and I-V data
Data = Datahandler(workdir=workdir)
#   Edit image and mask size to match training data
img_size=256
mask_size=32
channels=1
targetCol='Eff'
CNN_data_ratio = 300    #   Edit based on number of cell required for CNN and GAN training
Val_data_ratio = 50     #   Validation dataset number of cells
targetCol = "Eff_std_mmad"
extraction_batch_size=11

transform_GAN=transforms.Compose([
    transforms.Resize((img_size, img_size), Image.BICUBIC),
    transforms.Grayscale(num_output_channels=channels),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*channels, [0.5]*channels),
])
transform_CNN=transforms.Compose([
    transforms.Resize((224, 224), Image.BILINEAR),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])
# %%-
# %%--  Feature extraction and data separation
Xcols, All_df = Ptcompute.extractFeature(CNN_e,All_df,transform_CNN,batch_size=extraction_batch_size)
NR_df = All_df.loc[All_df['Labels']!=0] # if 'Labels' don't exist - edit based on Reject(R)/non-Reject(NR) [optional]
R_df = All_df.loc[All_df['Labels']==0]  # if 'Labels' don't exist - edit based on Reject(R)/non-Reject(NR) [optional]
Data.matchDf = NR_df.copy(deep=True)
CNN_df, ML_df = Data.splitData(mlFrac=CNN_data_ratio/len(NR_df),randomSeed=seed)
Data.matchDf = ML_df.copy(deep=True)
Val_df, ML_df = Data.splitData(mlFrac=Val_data_ratio/len(ML_df),randomSeed=seed)
Data.matchDf = NR_df.copy(deep=True)
MLR_df = ML_df.copy(deep=True).append(R_df.loc[R_df[targetCol]>19])    # 19 is criteria to edit [ based on Reject(R)/non-Reject(NR)]
# %%-
# %%--  Machine learning regression
Data.matchDf=MLR_df.copy(deep=True)
Skmodel = Skcompute(Data,ML[1],name="ML_"+ML[0], save=False)
Skmodel.initTraining()
Skmodel.trainModel(
    Xcols=Xcols,
    Ycol=targetCol,
    predictType='Regression',
    randomSeed=seed,
    comment=""
)
ML=Skmodel.model
# %%-
# %%--  Average increase of efficiency on dataset
df=Val_df.copy(deep=True)
eff_rec=[]
res=pd.DataFrame()
res['ori_ml']=ML.predict(df[np.array(Xcols)])
res['ori_true']=[e for e in df['Eff']]
ds=Dataset(np.array(df['path']),None,transform_GAN,mode='GAN',param_mode={"img_size":img_size,"mask_size":mask_size,"mask_mode":'random'})
dl=DataLoader(ds, batch_size=10, shuffle=False, num_workers=0)
for i, (imgs, masked_imgs, masked_parts, x1, y1) in enumerate(dl):
    print(i+1," of ",len(dl))
    gen_parts=GAN_g(masked_imgs)
    filled_imgs = masked_imgs.clone()
    for k,x,y in zip(range(masked_imgs.size()[0]),x1,y1):
        filled_imgs[k, :,x : x + mask_size, y : y + mask_size] = gen_parts[k,:,:,:]
    CNN_imgs=F.interpolate(filled_imgs, size=[224,224],mode='bilinear', align_corners=True).expand(-1,3,-1,-1)
    features=CNN_e(CNN_imgs)
    eff_rec+=ML.predict(features.detach().numpy()).tolist()

res['rec_ml']=eff_rec
res.to_csv(Data.pathDic['outputs']+"val_set_random_rec.csv", index=False)
# %%-
# %%--  Plot actual vs predicted
df=Val_df.copy(deep=True)
df['Reconstructed']=eff_rec
df['predict']=ML.predict(df[np.array(Xcols)])
df=df.loc[df["Eff_std_mmad"]>15]
plt.figure(figsize=(6,6))
ax1 = plt.gca()
ax1.set_xlabel('Actual value')
ax1.set_ylabel('Predicted value')
ax1.scatter(df['Eff_std_mmad'], df['predict'], c="C0", marker=".", label='Original')
ax1.scatter(df['Eff_std_mmad'], df['Reconstructed'], c="C4", marker=".", label='Reconstructed')
ax1.plot([np.min([np.min(df['Eff_std_mmad']),np.min(df['predict'])]),np.max([np.max(df['Eff_std_mmad']),np.max(df['predict'])])],[np.min([np.min(df['Eff_std_mmad']),np.min(df['predict'])]),np.max([np.max(df['Eff_std_mmad']),np.max(df['predict'])])], linewidth=1 ,linestyle="--",c="C3", label="y=x")
ax1.legend()
plt.show()
# %%-
# %%--  Plot Residuals
plt.figure(figsize=(6,6))
ax1 = plt.gca()
ax1.set_xlabel('Actual value')
ax1.set_ylabel('Residual value')
ax1.scatter(df['Eff_std_mmad'], [r-o for r,o in zip(df['Reconstructed'],df['Eff_std_mmad'])], c="C2", marker=".")
plt.show()
# %%-
# %%--  Random reconstruction efficiency increase - One cell
df=Val_df.copy(deep=True)
N=100
save=True
index=524    #    INPUT CELL ID HERE
res_2=pd.DataFrame()
res_2['ori_true']=[df['Eff'][index]]*N
param_mode={
    "img_size":img_size,
    "mask_size":mask_size,
    "mask_mode":'random',
    }
ds=Dataset([],None,transform_GAN,mode='GAN',param_mode=param_mode)
img_fix=transform_GAN(Image.open(df['path'][index]))
img_suc=transform_GAN(Image.open(df['path'][index]))
eff_fix=[]
eff_suc=[]
Recs={'0':img_suc.clone().squeeze(0).detach().numpy()}
for i in range(N):
    print("Run #",i)
    masked_img, masked_part, x1, y1 = ds.apply_random_mask(img_fix)
    img_rec=img_fix.clone()
    gen_part=GAN_g(masked_img.unsqueeze(0))
    img_rec[:,x1:x1+mask_size,y1:y1+mask_size]=gen_part.squeeze(0)
    eff_fix.append(ML.predict(CNN_e(F.interpolate(img_rec.unsqueeze(0), size=[224,224],mode='bilinear', align_corners=True).expand(-1,3,-1,-1)).detach().numpy())[0])


    masked_img, masked_part, x1, y1 = ds.apply_position_mask(img_suc,(x1,y1))
    gen_part=GAN_g(masked_img.unsqueeze(0))
    img_suc[:,x1:x1+mask_size,y1:y1+mask_size]=gen_part.squeeze(0)
    Recs[str(i+1)]=img_suc.clone().squeeze(0).detach().numpy()
    eff_suc.append(ML.predict(CNN_e(F.interpolate(img_suc.unsqueeze(0), size=[224,224],mode='bilinear', align_corners=True).expand(-1,3,-1,-1)).detach().numpy())[0])

res_2['rec_fix']=eff_fix
res_2['rec_suc']=eff_suc
gif=[]
for i in range(N+1):
    fig=plt.figure(figsize=(10,10))
    plt.imshow(Recs[str(i)], aspect='equal', cmap='Greys_r')
    plt.axis('off')
    fig.suptitle('Cell #'+str(index)+" - Rec. #"+str(i).zfill(3))
    if save: fig.savefig(Data.pathDic['figures']+"31-05_Cell-"+str(index)+"_Rec-"+str(i).zfill(3)+".png", transparent=True)
    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    gif.append(image_from_plot)
if save: imageio.mimsave(Data.pathDic['figures']+"31-05_Cell-"+str(index)+".gif",gif,format='GIF', fps=5)
if save: res_2.to_csv(Data.pathDic['outputs']+"31-05_Cell-"+str(index)+"_random_rec.csv", index=False)
# %%-
# %%-- [LONG] Statistics - Random reconstruction efficiency increase - ALL CELLS
df=Val_df.copy(deep=True)
N=100
record_img=np.unique([int(n)-1 for n in np.logspace(0,np.log10(N),25)])
save=True
param_mode={
    "img_size":img_size,
    "mask_size":mask_size,
    "mask_mode":'random',
    }
ds=Dataset([],None,transform_GAN,mode='GAN',param_mode=param_mode)
saveDic={}
imageDic={}
count=1
for index,row in df.iterrows():
    print('Cell #'+str(index), "\t %d of %d cells"%(count,len(df)))
    count+=1
    img_fix=transform_GAN(Image.open(row['path']))
    img_suc=transform_GAN(Image.open(row['path']))
    saveDic[index]={
        'eff_true':row['Eff'],
        'eff_fix':[],
        'eff_suc':[],
    }
    imageDic[index]={}
    for i in range(N):
        print("\t Run #",i)
        masked_img, masked_part, x1, y1 = ds.apply_random_mask(img_fix)
        img_rec=img_fix.clone()
        gen_part=GAN_g(masked_img.unsqueeze(0))
        img_rec[:,x1:x1+mask_size,y1:y1+mask_size]=gen_part.squeeze(0)
        saveDic[index]['eff_fix'].append(ML.predict(CNN_e(F.interpolate(img_rec.unsqueeze(0), size=[224,224],mode='bilinear', align_corners=True).expand(-1,3,-1,-1)).detach().numpy())[0])

        masked_img, masked_part, x1, y1 = ds.apply_position_mask(img_suc,(x1,y1))
        gen_part=GAN_g(masked_img.unsqueeze(0))
        img_suc[:,x1:x1+mask_size,y1:y1+mask_size]=gen_part.squeeze(0)
        saveDic[index]['eff_suc'].append(ML.predict(CNN_e(F.interpolate(img_suc.unsqueeze(0), size=[224,224],mode='bilinear', align_corners=True).expand(-1,3,-1,-1)).detach().numpy())[0])
        if i in record_img: imageDic[index][str(i)]=img_suc.clone().squeeze(0).detach().numpy()

timestamps=datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
if save: SaveObj(saveDic,Data.pathDic['outputs']+timestamps+"_test_set_random_rec.pkl")
if save:
    for key in imageDic.keys():
        SaveObj(imageDic[key],Data.pathDic['outputs']+"cell_succ_rec//"+timestamps+"_"+str(key)+"_test_set_random_rec_imgs.pkl")
# %%-
