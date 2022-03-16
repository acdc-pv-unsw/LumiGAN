# %%--  Imports
#   General
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import datetime
import ast

#   Machine learning
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression, Lasso, LinearRegression, ElasticNet, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
from torchvision import transforms
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

#   Customs
from LumiGAN import *
from LumiGAN.utility import LoadObj, SaveObj
# %%-
# %%--  Load Model and setting
workdir = "TEST\\"    #   ADD DIRECTORY PATH - Needs to have match.csv file with path to images and I-V data
model_path = "Models\\LumiGAN_model.pkl"  #   ADD PATH HERE - Models can be loaded from https://github.com/acdc-pv-unsw/LumiNet
Models = LoadObj(model_path)
Data = Datahandler(workdir=workdir)
#   Edit image and mask size to match training data
img_size=256
mask_size=32
channels=1
CNN_data_ratio = 300    #   Edit based on number of cell required for CNN and GAN training
Val_data_ratio = 50     #   Validation dataset number of cells
targetCol = "Eff_std_mmad"
extraction_batch_size=11
res={}
res['CNN_extract']=Models['CNN_extract']
res['CNN_classifier']=Models['CNN_classifier']
res['Generator']=Models['Generator']
res['Discriminator_img']=Models['Discriminator_img']
res['Discriminator_mask']=Models['Discriminator_mask']
res['Xcols']=Models['Xcols']
res['seed']=Models['seed']
SaveObj(res,model_path)
All_df=pd.read_csv("TEST\\match.csv")  #   Load data here with path to images and I-V data
ML=('AdaBoost',AdaBoostRegressor(base_estimator = DecisionTreeRegressor(), n_estimators=100, loss='square'))
CNN_e=Models['CNN_extract']
GAN_g=Models['Generator']
Xcols=Models['Xcols']
seed=Models['seed']

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
# %%--  Scan through images to identify patch with defect
df=NR_df.copy(deep=True)
df['predict']=ML.predict(df[np.array(Xcols)])
df=df.drop(columns=[c for c in Xcols])

corners=[(32*i,32*j) for i in range(8) for j in range(8)]
loss_dic={}
for c in corners: loss_dic['pos-'+str(c)]=[]
corners_name=['pos-'+str(c) for c in corners]
GAN_g.cuda()
param_mode={
    "img_size":img_size,
    "mask_size":mask_size,
    "mask_mode":'random',
    }
ds=Dataset([],None,transform_GAN,mode='GAN',param_mode=param_mode)
loss=torch.nn.SmoothL1Loss(reduction='mean')
bs=13
i=0
for k,row in df.iterrows():
    i+=1
    print("image %s of %s"%(i,len(df)))
    img=transform_GAN(Image.open(row['path']))
    img_to_cat=[]
    mask_part_ls=[]
    for pos in corners:
        masked_img, masked_part, x1, y1 = ds.apply_position_mask(img,pos)
        img_to_cat.append(masked_img.unsqueeze(0))
        mask_part_ls.append(masked_part.unsqueeze(0))
    batch_img=torch.cat(img_to_cat)
    batch_mask=torch.cat(mask_part_ls)

    dl=DataLoader(TensorDataset(batch_img, batch_mask), batch_size=bs, shuffle=False, num_workers=0)
    for ii, data in enumerate(dl):
        imgs,target= data
        imgs=imgs.cuda()
        gens=GAN_g(imgs)
        iii=0
        for g,t in zip(gens.cpu().detach(),target):
            loss_dic['pos-'+str(corners[iii+ii*bs])].append(loss(g,t).item())
            iii+=1
        del imgs,gens
        torch.cuda.empty_cache()
for k,v in loss_dic.items():
    df[k]=v

GAN_g.cpu()
df.to_csv(workdir+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")+"_loss_per_quadrant.csv",index=False)
# %%-
# %%--  Select cells to reconstruct
corners=[(32*i,32*j) for i in range(8) for j in range(8)]
corners_name=['pos-'+str(c) for c in corners]
skip_corners=['pos-(0, 0)',
 'pos-(0, 32)',
 'pos-(0, 64)',
 'pos-(0, 96)',
 'pos-(0, 128)',
 'pos-(0, 160)',
 'pos-(0, 192)',
 'pos-(0, 224)',
 'pos-(224, 0)',
 'pos-(224, 32)',
 'pos-(224, 64)',
 'pos-(224, 96)',
 'pos-(224, 128)',
 'pos-(224, 160)',
 'pos-(224, 192)',
 'pos-(224, 224)',
 'pos-(160, 224)',
 'pos-(160, 0)',
 'pos-(96, 224)',
 'pos-(128, 0)',
 'pos-(32, 224)',
 'pos-(64, 0)',
 'pos-(32, 0)',
]
check=[]
for c_n in corners_name:
    mean=df[c_n].mean()
    std=df[c_n].std()
    check.append(mean+5*std)
df['corners']=[[]]*len(df)
positions_check=[]
for i,row in df.iterrows():
    positions_check_loc=[]
    for c_n,c,ch in zip(corners_name,corners,check):
        if row[c_n]>ch:
            positions_check_loc.append(c)
    positions_check.append(positions_check_loc)
df['check']=[len(pos) for pos in positions_check]
df['corners']=positions_check
# %%-
# %%--  Reconstruct cells
df2=df.loc[df['check']>0].copy(deep=True)
param_mode={
    "img_size":img_size,
    "mask_size":mask_size,
    "mask_mode":'random',
    }
ds=Dataset([],None,transform_GAN,mode='GAN',param_mode=param_mode)
rec_eff=[]
GAN_g.cuda()
CNN_e.cuda()
i=0
for k,row in df2.iterrows():
    i+=1
    img=transform_GAN(Image.open(row['path']))
    gen_parts=[]
    corners_done=[]
    mask_corners=row['corners']
    np.random.shuffle(mask_corners)
    for pos in mask_corners:
        masked_img, masked_part, x1, y1 = ds.apply_position_mask(img,pos)
        for p,gen_part in zip(corners_done,gen_parts):
            x2,y2=p[0],p[1]
            masked_img[:,x2:x2+mask_size,y2:y2+mask_size]=gen_part.squeeze(0)
        gen_parts.append(GAN_g(masked_img.cuda().unsqueeze(0)).squeeze(0).cpu())
        corners_done.append(pos)
    img_rec=img.clone()
    for pos,gen_part in zip(mask_corners,gen_parts):
        x1,y1=pos[0],pos[1]
        img_rec[:,x1:x1+mask_size,y1:y1+mask_size]=gen_part.squeeze(0)
    eff_rec_ml=(ML.predict(CNN_e(F.interpolate(img_rec.cuda().unsqueeze(0), size=[224,224],mode='bilinear', align_corners=True).expand(-1,3,-1,-1)).cpu().detach().numpy())[0])
    rec_eff.append(eff_rec_ml)
    print("image %s of %s - [%s]: %.2F --> %.2F [%.3F] "%(i,len(df2),len(mask_corners),row['Eff'],eff_rec_ml,eff_rec_ml-row['Eff']))

df2['reconstructed']=rec_eff
df2['improvement']=[rec-ori for rec,ori in zip(df2['reconstructed'],df2['Eff'])]
df2['increase']=[i if i>0 else 0 for i in df2['improvement']]
df2.to_csv(workdir+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")+"_5s_withCorner_reconstruction_.csv",index=False)
# %%-
# %%--  Analysis of best reconstruction
dfr=df2.copy(deep=True)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
save=True
comment='reconstruction'
loopMax=10
param_mode={
    "img_size":img_size,
    "mask_size":mask_size,
    "mask_mode":'random',
    }
ds=Dataset([],None,transform_GAN,mode='GAN',param_mode=param_mode)
GAN_g.cuda()
CNN_e.cuda()
res={}
check=10
k=0
for i,row in dfr.iterrows():
    k+=1
    print("Cell #",k," of ",len(dfr))
    eff_true=row['Eff']
    img=transform_GAN(Image.open(row['path'])).squeeze(0).detach().numpy()
    eff_ml=(ML.predict(CNN_e(transform_CNN(Image.open(row['path'])).cuda().unsqueeze(0)).cpu().detach().numpy())[0])
    pass_eff_rec=0
    try:
        mask_corners=ast.literal_eval(row['corners'])
    except:
        mask_corners=row['corners']
    np.random.shuffle(mask_corners)
    img_rec=transform_GAN(Image.open(row['path']))
    gen_parts=[]
    corners_done=[]
    for pos in mask_corners:
        masked_img, masked_part, x1, y1 = ds.apply_position_mask(img_rec,pos)
        for p,gen_part in zip(corners_done,gen_parts):
            x2,y2=p[0],p[1]
            masked_img[:,x2:x2+mask_size,y2:y2+mask_size]=gen_part.squeeze(0)
        gen_parts.append(GAN_g(masked_img.cuda().unsqueeze(0)).squeeze(0).cpu())
        corners_done.append(pos)

    img_msk=img_rec.clone()
    for pos,gen_part in zip(mask_corners,gen_parts):
        x1,y1=pos[0],pos[1]
        img_rec[:,x1:x1+mask_size,y1:y1+mask_size]=gen_part.squeeze(0)
        img_msk[:,x1:x1+mask_size,y1:y1+mask_size]=1

    eff_rec=(ML.predict(CNN_e(F.interpolate(img_rec.unsqueeze(0), size=[224,224],mode='bilinear', align_corners=True).expand(-1,3,-1,-1).cuda()).cpu().detach().numpy())[0])
    if pass_eff_rec<eff_rec:
        pass_img_msk=img_msk.clone()
        pass_img_rec=img_rec.clone()
        pass_eff_rec=eff_rec
        pass_mask_corners=mask_corners

    img_msk=pass_img_msk.clone()
    img_rec=pass_img_rec.clone()
    eff_rec=pass_eff_rec
    mask_corners=pass_mask_corners

    img_rec=img_rec.squeeze(0).detach().numpy()
    img_msk=img_msk.squeeze(0).detach().numpy()
    res[str(k)]={
        'path':row['path'],
        'index':i,
        'img':img,
        'img_rec':img_rec,
        'img_msk':img_msk,
        'mask_corners':mask_corners,
        'eff_ml':eff_ml,
        'eff_rec':eff_rec,
        'eff_true':eff_true,
    }
    for j in range(1):
        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(12,6))
        axes[0].imshow(img, aspect='equal', cmap='Greys_r')
        axes[1].imshow(img_msk, aspect='equal', cmap='Greys_r')
        axes[2].imshow(img_rec, aspect='equal', cmap='Greys_r')
        axes[3].imshow(img-img_rec, aspect='equal', cmap='Greys_r')

        axes[0].axis('off')
        axes[1].axis('off')
        axes[2].axis('off')
        axes[3].axis('off')

        axes[0].set_title("Original")
        axes[1].set_title("Masked")
        axes[2].set_title("Reconstructed")
        axes[3].set_title("Difference with original")

        axes[0].annotate("True:\nML:\nError:\nRelative:",xy=(0.05,-0.37),xycoords='axes fraction', fontsize=14)
        axes[0].annotate("%.2F\n%.2F\n%.3F\n%.2F%%"%(eff_true,eff_ml,eff_ml-eff_true,100*(eff_ml-eff_true)/eff_true),xy=(0.5,-0.37),xycoords='axes fraction', fontsize=14)
        axes[2].annotate("ML:\nError to True:\nError to ML:",xy=(0.01,-0.27),xycoords='axes fraction', fontsize=14)
        axes[2].annotate("%.2F\n%.3F (%.2F%%)\n%.3F (%.2F%%)"%(eff_rec,eff_rec-eff_true,100*(eff_rec-eff_true)/eff_true,eff_rec-eff_ml,100*(eff_rec-eff_ml)/eff_ml),xy=(0.55,-0.27),xycoords='axes fraction', fontsize=14)
        # axes[3].annotate("Error to True:\nError to ML:",xy=(0.05,-0.17),xycoords='axes fraction', fontsize=14)
        # axes[3].annotate("%.3F (%.2F%%)\n%.3F (%.2F%%)"%(eff_rec-eff_true,100*(eff_rec-eff_true)/eff_true,eff_rec-eff_ml,100*(eff_rec-eff_ml)/eff_ml),xy=(0.7,-0.17),xycoords='axes fraction', fontsize=14)

        fig.suptitle("Cell # "+str(k)+" - index "+str(i), y=0.80)
        fig.subplots_adjust(wspace=0.05)
        if save: fig.savefig(workdir+"figures//"+timestamp+"_"+comment+"_Cell_"+str(i)+".png", transparent=True)
        plt.show()
        plt.close()
if save: SaveObj(res,workdir+"outputs//"+timestamp+"_"+comment+"_results.pkl")
# %%-
# %%--  Specific Cell Reconstruction
Cell_index=[] #Add cell index here
Corner_index=[
    [], #add list of (x,y) of patch corners for each cells
]
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
save=True
comment='specific_cells'
loopMax=100
param_mode={
    "img_size":img_size,
    "mask_size":mask_size,
    "mask_mode":'random',
    }
ds=Dataset([],None,transform_GAN,mode='GAN',param_mode=param_mode)
GAN_g.cuda()
CNN_e.cuda()
res={}
k=0
for i,row in dfr.iterrows():
    if i not in Cell_index:continue
    print("Cell #",i)
    eff_true=row['Eff']
    img=transform_GAN(Image.open(row['path'])).squeeze(0).detach().numpy()
    eff_ml=(ML.predict(CNN_e(transform_CNN(Image.open(row['path'])).cuda().unsqueeze(0)).cpu().detach().numpy())[0])
    pass_eff_rec=0
    mask_corners=Corner_index[k]
    np.random.shuffle(mask_corners)
    img_rec=transform_GAN(Image.open(row['path']))
    gen_parts=[]
    corners_done=[]
    for pos in mask_corners:
        masked_img, masked_part, x1, y1 = ds.apply_position_mask(img_rec,pos)
        for p,gen_part in zip(corners_done,gen_parts):
            x2,y2=p[0],p[1]
            masked_img[:,x2:x2+mask_size,y2:y2+mask_size]=gen_part.squeeze(0)
        gen_parts.append(GAN_g(masked_img.cuda().unsqueeze(0)).squeeze(0).cpu())
        corners_done.append(pos)

    img_msk=img_rec.clone()
    for pos,gen_part in zip(mask_corners,gen_parts):
        x1,y1=pos[0],pos[1]
        img_rec[:,x1:x1+mask_size,y1:y1+mask_size]=gen_part.squeeze(0)
        img_msk[:,x1:x1+mask_size,y1:y1+mask_size]=1

    eff_rec=(ML.predict(CNN_e(F.interpolate(img_rec.unsqueeze(0), size=[224,224],mode='bilinear', align_corners=True).expand(-1,3,-1,-1).cuda()).cpu().detach().numpy())[0])
    if pass_eff_rec<eff_rec:
        pass_img_msk=img_msk.clone()
        pass_img_rec=img_rec.clone()
        pass_eff_rec=eff_rec
        pass_mask_corners=mask_corners

    img_msk=pass_img_msk.clone()
    img_rec=pass_img_rec.clone()
    eff_rec=pass_eff_rec
    mask_corners=pass_mask_corners

    img_rec=img_rec.squeeze(0).detach().numpy()
    img_msk=img_msk.squeeze(0).detach().numpy()
    res[str(k)]={
        'path':row['path'],
        'index':i,
        'img':img,
        'img_rec':img_rec,
        'img_msk':img_msk,
        'mask_corners':mask_corners,
        'eff_ml':eff_ml,
        'eff_rec':eff_rec,
        'eff_true':eff_true,
    }
    for j in range(1):
        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(12,6))
        axes[0].imshow(img, aspect='equal', cmap='Greys_r')
        axes[1].imshow(img_msk, aspect='equal', cmap='Greys_r')
        axes[2].imshow(img_rec, aspect='equal', cmap='Greys_r')
        axes[3].imshow(img-img_rec, aspect='equal', cmap='Greys_r')

        axes[0].axis('off')
        axes[1].axis('off')
        axes[2].axis('off')
        axes[3].axis('off')

        axes[0].set_title("Original")
        axes[1].set_title("Masked")
        axes[2].set_title("Reconstructed")
        axes[3].set_title("Difference with original")

        axes[0].annotate("True:\nML:\nError:\nRelative:",xy=(0.05,-0.37),xycoords='axes fraction', fontsize=14)
        axes[0].annotate("%.2F\n%.2F\n%.3F\n%.2F%%"%(eff_true,eff_ml,eff_ml-eff_true,100*(eff_ml-eff_true)/eff_true),xy=(0.5,-0.37),xycoords='axes fraction', fontsize=14)
        axes[2].annotate("ML:\nError to True:\nError to ML:",xy=(0.01,-0.27),xycoords='axes fraction', fontsize=14)
        axes[2].annotate("%.2F\n%.3F (%.2F%%)\n%.3F (%.2F%%)"%(eff_rec,eff_rec-eff_true,100*(eff_rec-eff_true)/eff_true,eff_rec-eff_ml,100*(eff_rec-eff_ml)/eff_ml),xy=(0.55,-0.27),xycoords='axes fraction', fontsize=14)

        fig.suptitle("Cell # "+str(k)+" - index "+str(i), y=0.80)
        fig.subplots_adjust(wspace=0.05)
        if save: fig.savefig(workdir+"figures//"+timestamp+"_"+comment+"_Cell_"+str(i)+".png", transparent=True)
        plt.show()
        plt.close()
    k+=1
if save: SaveObj(res,workdir+"outputs//"+timestamp+"_"+comment+"_results.pkl")
# %%-
