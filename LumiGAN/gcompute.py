# %%--  Imports
#   General
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import datetime
import time
import imageio
import matplotlib.pyplot as plt
from PIL import Image


#   Machine learning
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms

#   Customs
from LumiGAN.logger import Logger
from LumiGAN.dataset import Dataset
from LumiGAN.ganmodel import Generator, Discriminator
from LumiGAN.matplotlibstyle import *
from LumiGAN.utility import printAttr, printTitle, movingaverage, SaveObj
# %%-

class Gcompute():
    #   Constants
    defaultParam={
        'prepGDD':{
            'subset_size':None,
            'test_batch_size':5,
            'batch_size':11,
            'epochs': 10,
            'frac_loss_img':0.1,
            'frac_loss_mask':0.1,
            'frac_loss_pixel':0.8,
            'random_seed':None,
            'print_freq':10,
            'mode':'GAN',
            'set_position':False,
            'mask_mode':'random',
            'labels':None,
            'transform':None,
            'transform_train':None,
            'pixelwise_loss':torch.nn.SmoothL1Loss(),
            'generator':{
                'p2_deep':3,
                'p2_offset':6,
                'weights_init':True,
                'Optimizer':'Adam',
                'lr':0.0001,
                'b1':0.5,
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
                'lr':0.0001,
                'b1':0.9,
                'b2':0.999,
                'eps':1e-8,
                'weight_decay':0,
            }
        },
        'trainGDD':{
            'show_epoch_test_plot':False,
            'show_final_epoch_test_plot':True,
            'test_fix_position':True,
            'test_save_each_epoch':False,
            'make_gif':True,
            'gif_fps':5,
            'loss_palette':['cornflowerblue','mediumorchid','forestgreen','firebrick'],
            'loss_alpha':0.2,
            'loss_ms':2,
            'avg_window':None,
        },
    }
    def __init__(self, datahandler, name : str,save : bool):
        ' Initialize attributes and trace files. Default values are saved here'
        #--------   Check for GPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.isGPU = torch.cuda.is_available()
        torch.backends.cudnn.benchmark = True

        #--------   Define files to save
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        tracefile = datahandler.pathDic['traces']+timestamp+"_"+"_trace_"+name+".txt"
        logger = Logger(tracefile)

        #--------   Attributes and default values
        self.name = name
        self.timestamp = timestamp
        self.dh = datahandler
        self.save = save
        self.tracefile = tracefile
        self.logger = logger

        #--------   Print name
        if self.save: self.logger.open()
        print(">"*80)
        print(" "*np.max([0,np.int((80-len(self.name))/2)])+self.name)
        print("<"*80)
        print("\n")

        #--------   Print attributes
        printTitle("Class attributes")
        attr = self.__dict__
        attr['Working directory']=self.dh.pathDic['workdir']
        printAttr(attr,skip=['logger','df'])
        if self.save: self.logger.close()

    def prepGDD(self, img_size, mask_size, channels, params={}):
        '''Load models, data and optimizer for training'''
        #--------   Load default parameters and check for additionnal parameters
        self.d_prepGDD={'parameters':Gcompute.defaultParam['prepGDD']}
        for key in params.keys(): self.d_prepGDD['parameters'][key]=params[key]
        self.d_prepGDD['parameters']['img_size']=img_size
        self.d_prepGDD['parameters']['mask_size']=mask_size
        self.d_prepGDD['parameters']['channels']=channels
        d_p=self.d_prepGDD['parameters']


        #--------   Log information
        if self.save: self.logger.open()
        if not d_p['random_seed']: d_p['random_seed']=np.random.randint(9999)
        printTitle('Hyper-parameters')
        printAttr(d_p,skip=['generator','discriminator_img','discriminator_mask','transform','transform_train'])

        printTitle('Data augmentation')
        if not d_p['transform']:
            d_p['transform']=transforms.Compose([
                transforms.Resize((img_size, img_size), Image.BICUBIC),
                transforms.Grayscale(num_output_channels=channels),
                transforms.ToTensor(),
                transforms.Normalize([0.5]*channels, [0.5]*channels),
            ])
        if not d_p['transform_train']: d_p['transform_train']=d_p['transform']
        printAttr({'transform':d_p['transform'],'transform_train':d_p['transform_train']})

        #--------   Dataloaders definition
        df = self.dh.matchDf.copy(deep=True)
        if d_p['subset_size']: df=df.sample(d_p['subset_size'], random_state = d_p['random_seed'])
        if not d_p['mode']=='GAN': raise ValueError('Dataset mode is not GAN')
        param_mode={
            'img_size':img_size,
            'mask_size':mask_size,
            'mask_mode':d_p['mask_mode']
        }
        if d_p['set_position']:
            param_mode['mask_mode']='position'
            param_mode['positions']={}
            for pos, path in zip(df['position'],df['path']): param_mode['positions'][path]=pos

        df_train, df_test = train_test_split(df,test_size = d_p['test_batch_size']/len(df),random_state=d_p['random_seed'])
        Train_set = Dataset(np.array(df_train['path']), d_p['labels'], d_p['transform_train'], mode='GAN',param_mode=param_mode)
        Test_set = Dataset(np.array(df_test['path']), d_p['labels'], d_p['transform'], mode='GAN',param_mode=param_mode)
        self.d_prepGDD['data']={}
        self.d_prepGDD['data']['df_train']=df_train['path']
        self.d_prepGDD['data']['df_test']=df_test['path']
        self.d_prepGDD['data']['dataloader_train'] = DataLoader(Train_set, batch_size=d_p['batch_size'], shuffle=True, num_workers=0)
        self.d_prepGDD['data']['dataloader_test'] = DataLoader(Test_set, batch_size=d_p['test_batch_size'], shuffle=False, num_workers=0)

        #--------   Model definition
        self.d_prepGDD['model'] = {}
        d_m = self.d_prepGDD['model']

        printTitle('Generator')
        printAttr(d_p['generator'])
        d_m['generator'] = Generator(
            img_size=d_p['img_size'],
            mask_size=d_p['mask_size'],
            channels=d_p['channels'],
            p2_deep=d_p['generator']['p2_deep'],
            p2_offset=d_p['generator']['p2_offset'],
            weights_init=d_p['generator']['weights_init'],
            )
        d_m['g_optim'] = torch.optim.Adam(
            d_m['generator'].parameters(),
            lr=d_p['generator']['lr'],
            betas=(d_p['generator']['b1'],d_p['generator']['b2']),
            eps=d_p['generator']['eps'],
            weight_decay=d_p['generator']['weight_decay'],
            )
        print(d_m['generator'])

        printTitle('Image discriminator')
        printAttr(d_p['discriminator_img'])
        d_m['discriminator_img'] = Discriminator(
            img_size=d_p['img_size'],
            channels=d_p['channels'],
            p2_deep=d_p['discriminator_img']['p2_deep'],
            p2_offset=d_p['discriminator_img']['p2_offset'],
            weights_init=d_p['discriminator_img']['weights_init'],
        )
        d_m['di_optim'] = torch.optim.Adam(
            d_m['discriminator_img'].parameters(),
            lr=d_p['discriminator_img']['lr'],
            betas=(d_p['discriminator_img']['b1'],d_p['discriminator_img']['b2']),
            eps=d_p['discriminator_img']['eps'],
            weight_decay=d_p['discriminator_img']['weight_decay'],
            )
        print(d_m['discriminator_img'])

        printTitle('Image mask')
        printAttr(d_p['discriminator_mask'])
        d_m['discriminator_mask'] = Discriminator(
            img_size=d_p['mask_size'],
            channels=d_p['channels'],
            p2_deep=d_p['discriminator_mask']['p2_deep'],
            p2_offset=d_p['discriminator_mask']['p2_offset'],
            weights_init=d_p['discriminator_mask']['weights_init'],
        )
        d_m['dm_optim'] = torch.optim.Adam(
            d_m['discriminator_mask'].parameters(),
            lr=d_p['discriminator_mask']['lr'],
            betas=(d_p['discriminator_mask']['b1'],d_p['discriminator_mask']['b2']),
            eps=d_p['discriminator_mask']['eps'],
            weight_decay=d_p['discriminator_mask']['weight_decay'],
            )
        print(d_m['discriminator_mask'])

        if self.save: self.logger.close()
        if self.isGPU:
            d_m['generator'].cuda()
            d_m['discriminator_img'].cuda()
            d_m['discriminator_mask'].cuda()
            d_p['pixelwise_loss'].cuda()

    def trainGDD(self, params={}):
        '''Execute training loop'''
        #--------   Load default parameters and check for additionnal parameters
        if not self.d_prepGDD: raise ValueError("prepGDD not run")
        self.d_trainGDD={'parameters':Gcompute.defaultParam['trainGDD']}
        for key in params.keys(): self.d_trainGDD['parameters'][key]=params[key]
        t_p=self.d_trainGDD['parameters']
        d_p=self.d_prepGDD['parameters']
        d_m=self.d_prepGDD['model']
        d_d=self.d_prepGDD['data']

        #--------   Prep saving files
        t_p['loss_file'] = self.dh.pathDic['figures']+self.timestamp+"_"+self.name+"_"+"loss-plot"+".png"
        t_p['test_file'] = self.dh.pathDic['figures']+self.timestamp+"_"+self.name+"_"+"test-img"+".png"
        t_p['gif_file'] = self.dh.pathDic['figures']+self.timestamp+"_"+self.name+"_"+"test-gif"+".gif"
        t_p['trace_file'] = self.tracefile
        t_p['model_file'] = self.dh.pathDic['models']+self.timestamp+"_"+self.name+"_"+"gdd"+".pkl"

        #--------   Log information
        if self.save: self.logger.open()

        #--------   Training Loop
        printTitle("Training verbose")
        self.d_trainGDD['losses']={
            'x':[],
            'gm_adv':[],
            'gi_adv':[],
            'g_pixel':[],
            'g_loss':[],
            'di_loss':[],
            'dm_loss':[],
        }
        t_l=self.d_trainGDD['losses']
        Tensor=torch.cuda.FloatTensor if self.isGPU else torch.FloatTensor
        epoch_start_time=time.time()
        gif=[]
        for epoch in range(d_p['epochs']):
            print(" ----Epoch {}----".format(epoch+1))
            start_time=time.time()
            for i, (imgs, masked_imgs, masked_parts, x1, y1) in enumerate(d_d['dataloader_train']):
                #  <subcell>    Generator training
                masked_imgs = Variable(masked_imgs.type(Tensor), requires_grad=False)
                masked_parts = Variable(masked_parts.type(Tensor), requires_grad=False)

                d_m['g_optim'].zero_grad()

                # Adversarial and pixelwise loss
                gen_parts = d_m['generator'](masked_imgs)
                filled_imgs = masked_imgs.clone()
                for k,x,y in zip(range(masked_imgs.size()[0]),x1,y1):
                    filled_imgs[k, :,y : y + d_p['mask_size'], x : x + d_p['mask_size']] = gen_parts[k,:,:,:]
                gm_adv = -torch.mean(d_m['discriminator_mask'](gen_parts))
                gi_adv = -torch.mean(d_m['discriminator_img'](filled_imgs))
                g_pixel = d_p['pixelwise_loss'](gen_parts, masked_parts)

                g_loss = d_p['frac_loss_img']*gi_adv +d_p['frac_loss_mask']*gm_adv+d_p['frac_loss_pixel']*g_pixel

                g_loss.backward()
                d_m['g_optim'].step()
                #  </subcell>
                #  <subcell>    Discriminator_img training
                # Define variable
                imgs = Variable(imgs.type(Tensor), requires_grad=True)
                masked_imgs = Variable(masked_imgs.type(Tensor), requires_grad=True)
                masked_parts = Variable(masked_parts.type(Tensor), requires_grad=True)

                d_m['di_optim'].zero_grad()

                # Real images
                real_validity = d_m['discriminator_img'](imgs)
                # Generate a batch of images
                gen_parts = d_m['generator'](masked_imgs)
                filled_imgs = masked_imgs.clone()
                for k,x,y in zip(range(masked_imgs.size()[0]),x1,y1):
                    filled_imgs[k, :,y : y + d_p['mask_size'], x : x + d_p['mask_size']] = gen_parts[k,:,:,:]
                fake_validity = d_m['discriminator_img'](filled_imgs)

                # Compute W-div gradient penalty
                real_grad_out = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
                real_grad = torch.autograd.grad(real_validity, imgs, real_grad_out, create_graph=True, retain_graph=True, only_inputs=True)[0]
                real_grad_norm = real_grad.view(real_grad.size(0), -1).pow(2).sum(1) ** (3)

                fake_grad_out = Variable(Tensor(filled_imgs.size(0), 1).fill_(1.0), requires_grad=False)
                fake_grad = torch.autograd.grad(fake_validity, filled_imgs, fake_grad_out, create_graph=True, retain_graph=True, only_inputs=True)[0]
                fake_grad_norm = fake_grad.view(fake_grad.size(0), -1).pow(2).sum(1) ** (3)

                div_gp = torch.mean(real_grad_norm + fake_grad_norm)

                # Adversarial loss
                di_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + div_gp

                di_loss.backward()
                d_m['di_optim'].step()

                #  </subcell>
                #  <subcell>    Discriminator_mask training
                # Define variable
                imgs = Variable(imgs.type(Tensor), requires_grad=True)
                masked_imgs = Variable(masked_imgs.type(Tensor), requires_grad=True)
                masked_parts = Variable(masked_parts.type(Tensor), requires_grad=True)

                d_m['dm_optim'].zero_grad()

                # Real images
                real_validity = d_m['discriminator_mask'](masked_parts)
                # Generate a batch of images
                gen_parts = d_m['generator'](masked_imgs)
                fake_validity = d_m['discriminator_mask'](gen_parts)

                # Compute W-div gradient penalty
                real_grad_out = Variable(Tensor(masked_parts.size(0), 1).fill_(1.0), requires_grad=False)
                real_grad = torch.autograd.grad(real_validity, masked_parts, real_grad_out, create_graph=True, retain_graph=True, only_inputs=True)[0]
                real_grad_norm = real_grad.view(real_grad.size(0), -1).pow(2).sum(1) ** (3)

                fake_grad_out = Variable(Tensor(gen_parts.size(0), 1).fill_(1.0), requires_grad=False)
                fake_grad = torch.autograd.grad(fake_validity, gen_parts, fake_grad_out, create_graph=True, retain_graph=True, only_inputs=True)[0]
                fake_grad_norm = fake_grad.view(fake_grad.size(0), -1).pow(2).sum(1) ** (3)

                div_gp = torch.mean(real_grad_norm + fake_grad_norm)

                # Adversarial loss
                dm_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + div_gp

                dm_loss.backward()
                d_m['dm_optim'].step()
                #  </subcell>
                #  <subcell>    Verbose
                t_l['x'].append(epoch+(i+1)/len(d_d['dataloader_train']))
                t_l['gm_adv'].append(gm_adv.item())
                t_l['gi_adv'].append(gi_adv.item())
                t_l['g_pixel'].append(g_pixel.item())
                t_l['g_loss'].append(g_loss.item())
                t_l['di_loss'].append(di_loss.item())
                t_l['dm_loss'].append(dm_loss.item())
                if (i % d_p['print_freq']==0) or  (i==len(d_d['dataloader_train'])-1):
                    print(
                        "\t %.2F %% \t Epoch %s/%d - Batch %s/%d took %.2F s \t [Di loss: %+.2E] [Dm loss: %+.2E] [G tot : %+.2E]"
                        % (100*(epoch*len(d_d['dataloader_train'])+i+1)/(d_p['epochs']*len(d_d['dataloader_train'])),str(epoch+1).zfill(int(1+np.floor(np.log10(d_p['epochs'])))), d_p['epochs'], str(i+1).zfill(int(1+np.floor(np.log10(len(d_d['dataloader_train']))))), len(d_d['dataloader_train']), time.time()-start_time, di_loss.item(), dm_loss.item(), g_loss.item())
                    )
                    start_time=time.time()
                #  </subcell>
                #  <subcell>    Clear memory
                del imgs, masked_imgs, masked_parts, filled_imgs, gen_parts
                del gm_adv, gi_adv, g_pixel, g_loss, di_loss, dm_loss, div_gp
                del real_validity, fake_validity, real_grad_out, real_grad, real_grad_norm, fake_grad_out, fake_grad, fake_grad_norm
                torch.cuda.empty_cache()
                #  </subcell>
            #  <subcell>    Update test set plot
            #    <ssubcell> Prep data
            imgs, masked_imgs, masked_parts, x1, y1  = next(iter(d_d['dataloader_test']))
            if t_p['test_fix_position']:
                if epoch==0:
                    xt,yt=x1,y1
                else:
                    x1,y1=xt,yt
                    masked_imgs=imgs.clone()
                for k,x,y in zip(range(masked_parts.size()[0]),x1,y1):
                    masked_parts[k,:,:] = imgs[k, :,x : x + d_p['mask_size'], y : y + d_p['mask_size']]
                for k,x,y in zip(range(imgs.size()[0]),x1,y1):
                        masked_imgs[k, :,x : x + d_p['mask_size'], y : y + d_p['mask_size']]=1

            masked_imgs = Variable(masked_imgs.type(Tensor))
            gen_masks = d_m['generator'](masked_imgs)
            if self.isGPU:
                with torch.no_grad():
                    masked_imgs = masked_imgs.detach().cpu()
                    gen_masks = gen_masks.detach().cpu()
            filled_imgs = imgs.clone()
            blank_imgs = imgs.clone()
            blank_imgs[:,:,:]=1
            for k,x,y in zip(range(gen_masks.size()[0]),x1,y1):
                filled_imgs[k, :,x : x + d_p['mask_size'], y : y + d_p['mask_size']]=gen_masks[k,:,:,:]
                blank_imgs[k, :,x : x + d_p['mask_size'], y : y + d_p['mask_size']]=0
            #    </ssubcell>
            #    <ssubcell> Plot
            n_imgs = d_p['test_batch_size']
            fig, axes = plt.subplots(nrows=n_imgs, ncols=3, figsize=(6,3*n_imgs), gridspec_kw={'width_ratios': [3,3,2]})
            for real, blank, filled, gen, mask, ax in zip(imgs, blank_imgs, filled_imgs, gen_masks, masked_parts, axes):
                ax[0].imshow(real.squeeze(), aspect='equal', cmap='Greys_r')
                ax[0].imshow(blank.squeeze(), aspect='equal', cmap='Greys_r',alpha = 0.1)
                ax[1].imshow(filled.squeeze(), aspect='equal', cmap='Greys_r')
                ax[2].imshow(mask.squeeze()-gen.squeeze(), aspect='equal', cmap='Greys_r')

                ax[0].axis('off')
                ax[1].axis('off')
                ax[2].axis('off')

            axes[0][0].set_title("Original", y=1.05)
            axes[0][1].set_title("Generated", y=1.05)
            axes[0][2].set_title("Difference", y=1.32)
            fig.suptitle(" Epoch %s/%d - %s"%(str(epoch+1).zfill(int(1+np.floor(np.log10(d_p['epochs'])))), d_p['epochs'], self.timestamp+"_"+self.name), y=1-2*n_imgs/100)
            fig.subplots_adjust(wspace=0.05, hspace=-n_imgs/10)
            if self.save and t_p['test_save_each_epoch']: fig.savefig(self.dh.pathDic['figures']+self.timestamp+"_"+self.name+"_test-img_epoch"+str(epoch+1).zfill(int(1+np.floor(np.log10(d_p['epochs']))))+".png",transparent=True)
            if self.save and (epoch==d_p['epochs']-1): fig.savefig(t_p['test_file'], transparent=True)
            if t_p['make_gif']:
                fig.canvas.draw()
                image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                gif.append(image_from_plot)
            if t_p['show_epoch_test_plot'] or (t_p['show_final_epoch_test_plot'] and (epoch==d_p['epochs']-1)): plt.show()
            plt.close()
            #    </ssubcell>
            #    <ssubcell> Clear memory
            del imgs, masked_imgs, masked_parts, filled_imgs, gen_masks, blank_imgs
            torch.cuda.empty_cache()
            #    </ssubcell>
            #  </subcell>

        #--------   Plot results
        #  <subcell>    Loss plot
        if not t_p['avg_window']: t_p['avg_window']=int(len(t_l['x'])/d_p['epochs'])
        win_slice=int(np.floor(t_p['avg_window']/2))

        fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10,6))
        axes[0].scatter(t_l['x'], t_l['gm_adv'], c=t_p['loss_palette'][0], alpha=t_p['loss_alpha'], s=t_p['loss_ms'])
        axes[0].scatter(t_l['x'], t_l['gi_adv'], c=t_p['loss_palette'][1], alpha=t_p['loss_alpha'], s=t_p['loss_ms'])
        axes[0].scatter(t_l['x'], t_l['g_pixel'], c=t_p['loss_palette'][2], alpha=t_p['loss_alpha'], s=t_p['loss_ms'])
        axes[0].scatter(t_l['x'], t_l['g_loss'], c=t_p['loss_palette'][3], alpha=t_p['loss_alpha'], s=t_p['loss_ms'])
        axes[0].plot(t_l['x'][win_slice-(1+t_p['avg_window'])%2:-win_slice], movingaverage(t_l['gm_adv'], t_p['avg_window'], 'valid'), c=t_p['loss_palette'][0],label='Mask GAN loss [%s]'%(d_p['frac_loss_mask']))
        axes[0].plot(t_l['x'][win_slice-(1+t_p['avg_window'])%2:-win_slice], movingaverage(t_l['gi_adv'], t_p['avg_window'], 'valid'), c=t_p['loss_palette'][1],label='Image GAN loss [%s]'%(d_p['frac_loss_img']))
        axes[0].plot(t_l['x'][win_slice-(1+t_p['avg_window'])%2:-win_slice], movingaverage(t_l['g_pixel'], t_p['avg_window'], 'valid'), c=t_p['loss_palette'][2],label='Pixelwise loss [%s]'%(d_p['frac_loss_pixel']))
        axes[0].plot(t_l['x'][win_slice-(1+t_p['avg_window'])%2:-win_slice], movingaverage(t_l['g_loss'], t_p['avg_window'], 'valid'), c=t_p['loss_palette'][3],label='Generator total loss')

        axes[1].scatter(t_l['x'], t_l['dm_loss'], c=t_p['loss_palette'][0], alpha=t_p['loss_alpha'], s=t_p['loss_ms'],)
        axes[1].scatter(t_l['x'], t_l['di_loss'], c=t_p['loss_palette'][1], alpha=t_p['loss_alpha'], s=t_p['loss_ms'])
        axes[1].plot(t_l['x'][win_slice-(1+t_p['avg_window'])%2:-win_slice], movingaverage(t_l['dm_loss'], t_p['avg_window'], 'valid'), c=t_p['loss_palette'][0],label='Mask Discriminator loss ')
        axes[1].plot(t_l['x'][win_slice-(1+t_p['avg_window'])%2:-win_slice], movingaverage(t_l['di_loss'], t_p['avg_window'], 'valid'), c=t_p['loss_palette'][1],label='Image Discriminator loss ')

        axes[0].set_ylabel('Generator loss')
        axes[1].set_ylabel('Discriminator loss')
        axes[1].set_xlabel('Epochs')
        axes[1].locator_params(axis='x', nbins=d_p['epochs']+1)
        lgn1=axes[0].legend(ncol=1,bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.,frameon=False)
        lgn2=axes[1].legend(ncol=1,bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., frameon=False)

        fig.suptitle("Loss plot - "+self.timestamp+"_"+self.name)
        fig.subplots_adjust(hspace=0.1)
        if self.save: fig.savefig(t_p['loss_file'],bbox_extra_artists=(lgn1,lgn2,fig.texts[0]), transparent=True, bbox_inches='tight')
        plt.show()
        #  </subcell>

        #--------   Save data and results
        if self.isGPU:
            d_m['generator'].cpu()
            d_m['discriminator_img'].cpu()
            d_m['discriminator_mask'].cpu()
            d_p['pixelwise_loss'].cpu()
            torch.cuda.empty_cache()
        if t_p['make_gif'] and self.save: imageio.mimsave(t_p['gif_file'],gif,format='GIF', fps=t_p['gif_fps'])
        if self.save: self.logger.close()
        if self.save:
            to_save=d_m
            to_save['losses']=t_l
            to_save['parameters']=d_p
            SaveObj(to_save, t_p['model_file'])
