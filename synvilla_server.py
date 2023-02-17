# -*- coding: utf-8 -*-

# Synvilla, a tool for progressive image evolution using stable diffusion
# Hannu Toyryla @htoyryla 2023

# Server

import os
import time
import random
import imageio
import numpy as np
import PIL
from PIL import Image, ImageEnhance
from skimage import exposure
import time

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from diffusers import AutoencoderKL, UNet2DConditionModel, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, LMSDiscreteScheduler, DDPMScheduler, DDIMScheduler, PNDMScheduler  
from transformers import CLIPTextModel, CLIPTokenizer

from configparser import SafeConfigParser

import inspect
import tqdm

import time

from synvilla_api import SynvillaAPI

with open('synvilla_version') as vf:
    version = vf.readline().replace("\n","")
    print(version)

def str2bool(v):
  return v.lower() in ("yes", "true", "1")


# class for reading configuration from file

class ReadConfig:

  def __init__(self, fn):

    conf = SafeConfigParser()
    conf.read(fn)

    self.steps = int(conf.get("run", "iters"))        # number of iterations on the same input
    self.text = conf.get("run", "prompt")             # initial text prompt
    self.fname = conf.get("run", "name")              # basename for saved images
    self.output_path = conf.get("run", "path")        # path for saving images
    self.modelpath = conf.get("run", "model")
    self.modeldir = conf.get("run", "modeldir")           # path for loading model
    self.seed = int(conf.get("run", "seed"))              # random seed
    self.tknzr = conf.get("run","tknzr",fallback="")
    self.sched = conf.get("run", "sched", fallback = "LMS")
    self.slices = int(conf.get("run", "slices", fallback = 2))
    self.guidance_scale = float(conf.get("run","g"))
    self.h = int(conf.get("image", "h"))   
    self.w = int(conf.get("image", "w"))
    self.maxnoise = int(conf.get("run", "maxnoise", fallback = 0.6))
    

# Now follows the server, as yet not a class

# image processing routines

def preprocess(image):
    global api
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    #image = image.convert("RGB")
    
    print(image.format, image.size, image.mode)

    if image.mode == "I":
        ima = np.array(image)
        ima = (ima / 65536).astype(np.uint8)
        print(ima.min(), ima.max(), ima.dtype.type)
        image = image.fromArray(ima)
    
    print("--------_",api.gamma, api.contrast)
    image = ImageEnhance.Brightness(image).enhance(api.gamma)
    image = ImageEnhance.Contrast(image).enhance(api.contrast)
    image = np.array(image)

    if image.ndim == 2: # handle BW image
        print(image.dtype)
        #if (image.dtype.type is np.uint32) or (image.dtype.type is np.int32): #catch uint32 BW images TODO ch3ck for image dtype
        #  image = image.astype(np.float32) / 8421504
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2) #image[:,:,np.newaxis]
    
    image = image.astype(np.float32) / 255.0
        
    image = image[None].transpose(0,3,1,2)
    image = torch.from_numpy(image)[:,:3,:,:]
    print(image.shape)
    print("new init image:", image.shape, image.min(), image.mean(), image.std(), image.max())    
    image = 2.0 * image - 1.0

    return image    
    
def numpy_to_pil(images):
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images

to_tensor_tfm = transforms.ToTensor()

def pil_to_latent(input_im):
  with torch.no_grad():
    latent = vae.encode(to_tensor_tfm(input_im).unsqueeze(0).to(torch_device)*2-1) # Note scaling
  return 0.18215 * latent.mode() # or .mean or .sample

def preprocess_mask(mask):
    global h, w
    mask = mask.convert("L")
    mask = mask.resize((w // 8, h // 8), resample=PIL.Image.NEAREST)
    mask = np.array(mask).astype(np.float32) / 255.0
    mask -= mask.min()
    mask /= mask.max() 
    mask = torch.from_numpy(mask)
    return mask.unsqueeze(0).unsqueeze(0)	    

# --------------------------------------------------------

# latents handling

def get_timesteps(s):
    global init_timestep, offset
    
    offset = scheduler.config.get("steps_offset", 0)
    init_timestep = int(num_inference_steps * s) + offset
    init_timestep = min(init_timestep, num_inference_steps)

    timesteps = scheduler.timesteps[-init_timestep]
    timesteps = torch.tensor([timesteps] * bs, device=device)
    
    t_start = max(num_inference_steps - init_timestep + offset, 0)    
    
    return timesteps, t_start


def get_latents(img = "", noisefactor=0):
  global generator, s, scheduler    
   
  print("get latents", img, s) 
  if img != "":
      print("new latents from image")
      init_image = Image.open(img) #.convert("RGB")
      init_image = init_image.resize((w, h))
      init_image = preprocess(init_image)
      
  
      # encode the init image into latents and scale the latents
      with torch.no_grad():
          init_latent_dist = vae.encode(init_image.to(device)).latent_dist
          init_latents = init_latent_dist.sample()
          init_latents = 0.18215 * init_latents
    
          #orig_latents = init_latents.clone()
  
          init_latents = torch.cat([init_latents]*bs)
  
          timesteps, t_start = get_timesteps(s)
          
          # add noise to latents using the timesteps
          noise = torch.randn(init_latents.shape, device=device, generator=generator)
          latents = scheduler.add_noise(init_latents, noise, timesteps).to(device).detach()
          print(s, t_start, latents.std())  
      #latents = latents * scheduler.sigmas[0] 
                     
  else:
      print("new random latents")
      latents = torch.randn(
          (bs, unet.in_channels, h // 8, w // 8), generator=generator, device=device
          )
      latents = latents * scheduler.init_noise_sigma       
    
  if noisefactor > 0:       
        print("adding noise to latents ",noisefactor)
        latents += torch.randn(latents.shape).to(device) * noisefactor 
  
      
  print("latents std ",latents.std())
  return latents           
    
    
# prompts to text embedding    
        
def get_embeddings_multi(prompt, negp):
    # prepare prompt: split into subprompts and their weights
    
    print("gem", prompt)

    plist = []    # a list for subprompts
    wlist = []    # a list for their weights
    wsum = 0

    parts = prompt.split("/") # split into subprompts at each /

    print("gem", parts)

    # separate text and weight for each subprompt
    
    for p in parts:
        if ":" in p:
          ps = p.split(":")
          plist.append(ps[0].strip())
          w = float(ps[1])
        else:
          plist.append(p)
          w = 10      
        wlist.append(w)
        wsum += w
        
    # normalize weights
    
    for i in range(0, len(wlist)):
      wlist[i] = wlist[i] / wsum
    
    print("gem", wlist)
    # text to tokens  

    # we process a list of all subprompts at the same time 
     
    text_tokens = tokenizer(plist, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")

    print("Tokens shape",text_tokens.input_ids.shape)

    # tokens to embedding

    with torch.no_grad():
      text_embeddings = text_encoder(text_tokens.input_ids.to(device))[0]
  
    print("gem", "Text embeddings shape ",text_embeddings.shape)  
  
    # encode empty prompt
    
    if negp == "":  
        tokens_length = text_tokens.input_ids.shape[-1]
        uncond_tokens = tokenizer(
            [""] * bs, padding="max_length", max_length=tokens_length, return_tensors="pt"
        )
        with torch.no_grad():
          uncond_embeddings = text_encoder(uncond_tokens.input_ids.to(device))[0] 
    else:
        tokens_length = text_tokens.input_ids.shape[-1]
        uncond_tokens = tokenizer(
            [negp] * bs, padding="max_length", max_length=tokens_length, return_tensors="pt"
        )
        with torch.no_grad():
          uncond_embeddings = text_encoder(uncond_tokens.input_ids.to(device))[0] 

    # store both embeddings

    text_embeddings = torch.cat([text_embeddings, uncond_embeddings], dim=0)    
    
    wlist.append(-1.)
    
    weights = np.asarray(wlist, dtype=np.float32)
    print(weights)
    pos_weights = torch.tensor(weights[weights > 0], device=device).reshape(-1, 1, 1, 1)
    pos_weights = pos_weights / pos_weights.sum()
    neg_weights = torch.tensor(weights[weights < 0], device=device).reshape(-1, 1, 1, 1)
    neg_weights = neg_weights / neg_weights.sum()
    
    return text_embeddings, weights, pos_weights, neg_weights 

# -----------------

# variables needes at init phase

cnf = ReadConfig("sds.ini")



modelpath = cnf.modelpath
tknzr = cnf.tknzr
sched = cnf.sched
slices = cnf.slices
steps = cnf.steps

seed = cnf.seed

bs = 1

device = "cuda"

# set up random number gen

if seed != 0:
    cseed = seed   
else:
    cseed = torch.Generator().seed()
    
generator = torch.Generator(device=device).manual_seed(cseed) 

# set up models

# VAE

print("setting up VAE")

vae = AutoencoderKL.from_pretrained(
        modelpath, subfolder="vae", use_auth_token=False
)

vae.eval()
vae.cuda()

# Unet

print("setting up UNET")

unet = UNet2DConditionModel.from_pretrained(
        modelpath, subfolder="unet", use_auth_token=False
)
unet.eval()
unet.cuda()

# text encoding

print("setting up text encoder")

if tknzr != "":
        tokenizer = CLIPTokenizer.from_pretrained(tknzr+"/tokenizer/")
        print("Loading tokenizer from "+tknzr)
elif modelpath:
        tokenizer = CLIPTokenizer.from_pretrained(modelpath, subfolder="tokenizer")
        print("Loading tokenizer from "+modelpath)

if tknzr != "":
    text_encoder = CLIPTextModel.from_pretrained(tknzr+"/text_encoder/").cuda()
else:
    text_encoder = CLIPTextModel.from_pretrained(modelpath, subfolder="text_encoder", use_auth_token=False).cuda()

# noise scheduler

if sched == "LMS":
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
elif sched == "DDIM":
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
elif sched == "DDPM":
    scheduler = DDPMScheduler() #beta_start=0.0001, beta_end=0.02, beta_schedule="linear", num_train_timesteps=1000)
#elif sched == "PNDM":
#    scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
elif sched == "EULERv":    
    scheduler = EulerDiscreteScheduler.from_pretrained(modelpath, subfolder="scheduler", prediction_type="v_prediction")
elif sched == "EULER":    
    scheduler = EulerDiscreteScheduler.from_pretrained(modelpath, subfolder="scheduler")
elif sched == "EULERA":    
    scheduler = EulerAncestralDiscreteScheduler.from_pretrained(modelpath, subfolder="scheduler")
else:
    print("Unknown scheduler ",sched)
    exit()    

eta = 0
accepts_eta = "eta" in set(inspect.signature(scheduler.step).parameters.keys())
extra_step_kwargs = {}
if accepts_eta:
    extra_step_kwargs["eta"] = eta

if isinstance(unet.config.attention_head_dim, int):
                # half the attention head size is usually a good trade-off between
                # speed and memory
                slice_size = unet.config.attention_head_dim // slices
else:
                # if `attention_head_dim` is a list, take the smallest head size
                slice_size = min(unet.config.attention_head_dim)

unet.set_attention_slice(slice_size)

num_inference_steps = steps #500 

scheduler.set_timesteps(num_inference_steps)


# --------------------------------------------------------

# main variables

text = cnf.text
print("ini", text)
ntext = ""
mline = ""
nmline = ""

h = cnf.h
w = cnf.w
guidance_scale = cnf.guidance_scale

output_path = cnf.output_path
fname = cnf.fname  

bg_w = 1.
fg_w = 1. 

s = 0.5
blend = 0
gamma = 1
contrast = 1

init_timestep = 0
offset = 0      
mask = None
image = ""    
      
# calculated values

# initial latents, either random or from init image

if image != "":
  latents = get_latents(img=image).to(device)
else:
  latents = get_latents().to(device)

# not used yet

mlatents = latents.clone() 
      
if image != "":
  t_start = max(num_inference_steps - init_timestep + offset, 0)    
else:
  t_start = 0
    
total = steps - t_start
  
saved_latents = latents.clone()

text_embs, weights, pos_weights, neg_weights = get_embeddings_multi(text, ntext)
mask_embs, mweights, mpos_weights, mneg_weights = get_embeddings_multi(mline, nmline)

'''''
mask_embs = None
mweights = None
mpos_weights = None
mneg_weights = None
'''

# temp

usemask = False

#-------------------------------------------------

# set up server API

api = SynvillaAPI()

# set values for the API

api.version = version
api.text = text
print("api", api.text)
api.ntext = ntext
api.beta = s
api.guidance = guidance_scale
api.steps = steps
api.newsteps = steps # to ensure some random value will not replace the value of steps soon
api.extranoise = 0
api.seed = seed 
api.blend = blend
api.gamma = gamma
api.maxnoise = 0.6
api.total = 0
api.bg_w = bg_w
api.fg_w = 1
api.h = h
api.w = w
api.contrast = contrast
api.schedlist = ["LMS", "EULER", "EULERA", "EULERv", "DDIM", "DDPM"]

# get a list of available models

if cnf.modeldir != None:
    dc = os.listdir(cnf.modeldir)
    ml = []
    for d in dc:
        mn = d.split("/")[-1]
        ml.append(mn)
    api.modellist = ml    

modelname = modelpath.split("/")[-1] 
api.status = "Using model: "+modelname+", scheduler: "+sched
api.status2 = ""
api.sched = cnf.sched
api.model = modelname

#--------------------------------------------------

# needed to pass a reference to server api to Flask

def getAPI():

    return api
   

# actual server loop
      
def server():
    global generator, t, latents, mlatents, image, text, ntext, mline, nmline 
    global scheduler, text_embs, mask_embs, ctr, s, total, num_inference_steps, steps, timesteps
    global guidance_scale, beta, weights, pos_weights, neg_weights
    global change_i, bg_w, fg_w, seed, generator, blend, s  
    global mask_embs, mweights, mpos_weights, mneg_weights, extranoise
    global h, w, sched, modelname, unet, tokenizer, text_encoder, vae, cnf, gamma, usemask
    
    print("starting SD server")

    # starting t for diffusion

    if image != "":
        t_start = max(num_inference_steps - init_timestep + offset, 0)    
    else:
        t_start = 0

    # total number of diffusion steps

    total = num_inference_steps - t_start
    api.total = total

    # not used yet
    
    mweights = []

    # placeholder for latent blending

    last_latents = None

    while True: 
      print("new loop starts", text, bg_w, fg_w)
      sname = time.strftime("%d%m%Y%H%M%S")    
      ctr = 0
      api.ctr = 0
      
      if last_latents != None:
          orig_latents = last_latents.clone()
      else:      
          orig_latents = latents.clone()
      #if api.inpaint:
      #  noise = torch.randn(orig_latents.shape, device=device, generator=generator)
      #  orig_latents = scheduler.add_noise(orig_latents, noise, timesteps).to(device).detach()
      
      with torch.no_grad():
        i = 0
        
        tsteps = scheduler.timesteps
        print(">>>>>>>>>", t_start, len(tsteps), text)
        t_index = t_start
        
        while t_index < len(tsteps) - 1:        
              bg_w = api.bg_w
        
              t_index = t_start + i
              t = tsteps[t_index]  
              print("***", t_index, t.item(), latents.std().item(), mlatents.std().item(), bg_w)
              
              # prepare current latent for UNET      

              latent_model_input = torch.cat([latents] * len(weights))
        
              # adjust latents according to sigmas (current noise level)
        
              latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        
              # estimate the noise for each subprompt (+ the uncond prompt with a negative weight)
           
              noise_preds = []
              for latent_in, text_embedding_in in zip(
                 torch.chunk(latent_model_input, chunks=latent_model_input.shape[0], dim=0),
                 torch.chunk(text_embs, chunks=text_embs.shape[0], dim=0)):
                 
                 noise_preds.append(unet(latent_in, t, encoder_hidden_states=text_embedding_in).sample)
              noise_preds = torch.cat(noise_preds, dim=0)
        
        
              # adjust noise estimate for text guidance

              # noise estimate for prompts with negative weight
              noise_preds_uncond = (noise_preds[weights < 0.] * neg_weights).sum(dim=0, keepdims=True)
        
              # noise estimate for prompts with positive weight
              noise_preds_text = (noise_preds[weights >= 0] * pos_weights).sum(dim=0, keepdims=True)
        
              # noise estimate with guidance
              noise_pred = noise_preds_uncond + guidance_scale * (noise_preds_text - noise_preds_uncond)

              # estimate denoised latent
       
              latents_ = scheduler.step(noise_pred, t, latents, **extra_step_kwargs)
              
              latents = latents_.prev_sample.detach()
              
              # get final image estimate
              
              lats_ = latents_.pred_original_sample.detach()
              
              
              #print("usemask", usemask, len(mweights))
              
              if usemask and len(mweights) > 0:  # we have both a mask and prompt(s) for it
                  print("mask on")
                  
                  # prepare current latent for UNET      
        
                  mlatent_model_input = torch.cat([mlatents] * len(mweights))
        
                  # adjust latents according to sigmas (current noise level)
                  #sigma = scheduler.sigmas[t_index]
        
                  mlatent_model_input = scheduler.scale_model_input(mlatent_model_input, t)                  
                  
                  # diffuse with mask embeddings
                  # estimate the noise for each subprompt (+ the uncond prompt with a negative weight)
           
                  mnoise_preds = []
                  for mlatent_in, mask_embedding_in in zip(
                     torch.chunk(mlatent_model_input, chunks=mlatent_model_input.shape[0], dim=0),
                     torch.chunk(mask_embs, chunks=mask_embs.shape[0], dim=0)):
                     #print(latent_in.shape, text_embedding_in.shape)
                     mnoise_preds.append(unet(mlatent_in, t, encoder_hidden_states=mask_embedding_in).sample)
                  mnoise_preds = torch.cat(mnoise_preds, dim=0)
                  
                  # adjust noise estimate for text guidance

                  # noise estimate for prompts with negative weight
                  mnoise_preds_uncond = (mnoise_preds[mweights < 0.] * mneg_weights).sum(dim=0, keepdims=True)
        
                  # noise estimate for prompts with positive weight
                  mnoise_preds_text = (mnoise_preds[mweights >= 0] * mpos_weights).sum(dim=0, keepdims=True)
        
                  # noise estimate with guidance
                  mnoise_pred = mnoise_preds_uncond + guidance_scale * (mnoise_preds_text - mnoise_preds_uncond)
                  # estimate denoised latent
       
                  mlatents_ = scheduler.step(mnoise_pred, t, mlatents, **extra_step_kwargs)
              
                  #latents_ = scheduler.step(noise_pred, t, latents, **extra_step_kwargs)              
                  mlatents = mlatents_.prev_sample.detach()
        
                  mlats = mlatents_.pred_original_sample.detach()              
        
                  if api.inpaint == 0:
                      lats_ =  (1 - mask) * lats_ + mask * mlats
                  else:
                      lats_ = (1 - mask) * orig_latents + mask * mlats      
                  
              # latent blending to smooth out transition when continuing from generated image 
              
              if api.blend > 0 and last_latents != None:
                  bf = 1 - ctr / api.blend
                  if bf > 0:
                      lats_ = bf * last_latents + (1 - bf) * lats_
                  else:
                      last_latents = lats_.clone()        
              else:
                  last_latents = lats_.clone()               
              
             
              # save an image from current latents
              
              lats_ = 1 / 0.18215 * lats_
  
              oimg = vae.decode(lats_.to(vae.dtype)).sample
              oimg = (oimg / 2 + 0.5).clamp(0, 1)
              oimg = oimg.cpu().permute(0, 2, 3, 1).detach().numpy()
              oimg = numpy_to_pil(oimg)[0]
              oimg.save("static/temp.jpg")
              os.rename("static/temp.jpg", "static/result.jpg")
              if output_path != "":
                  oimg.save(output_path+"/"+fname+"-"+sname+"-"+str(ctr)+".png")
              ctr += 1
              api.ctr = ctr
      
              i += 1 
              
              # check for api changed relevant while diffusion is still running
              
              gamma = api.gamma
              extranoise = api.extranoise
              
              changenotices = []
              
              if api.wh_changed:
                  h = api.h
                  w = api.w
                  changenotices.append("change to "+str(h)+"x"+str(w))

              if api.sched_changed:
                  changenotices.append("schedule change to "+api.sched)

              if api.model_changed:
                  changenotices.append("model change to "+api.model)
                  
              if len(changenotices) > 0:
                  api.status2 = "Pending " + ",".join(changenotices)      
                  
              if api.newImg:
                  break
                  
              # if reset pressed, reset latents, optimizer and prompt embedding
              if api.resetL:           
                  break   
                    
              changed, text, nprompt, mline, nmline = api.getChanges()  # check if prompt changed
              if changed:
                 text_embs, weights, pos_weights, neg_weights = get_embeddings_multi(text, nprompt)
                 mask_embs, mweights, mpos_weights, mneg_weights = get_embeddings_multi(mline, nmline)
            
                 changed = False
                 api.newprompt = False # mark change handled to api
                 
              if api.guidance != guidance_scale:
                  guidance_scale = api.guidance      
                  
              if api.seed != seed:
                  seed = api.seed     
                  if seed != 0:
                      cseed = seed   
                  else:
                      cseed = torch.Generator().seed()
    
        # now the diffusion cycle is finished, let's do a thorough check of api changes

        changenotices = []
        api.status2 = ""
               
        # loop until a new diffusion cycle is requested       
               
        while True:  
            gamma = api.gamma
            extranoise = api.extranoise
            
            # image size change
            
            if api.wh_changed:
                h = api.h
                w = api.w
                api.wh_changed = False
                changenotices.append("image size to "+str(h)+"x"+str(w))
            
            # scheduler change

            if api.sched_changed:
                modelpath = cnf.modeldir + "/" + api.model
                sched = api.sched
                
                if sched == "LMS":
                    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
                elif sched == "DDIM":
                    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
                elif sched == "DDPM":
                    scheduler = DDPMScheduler(beta_start=0.0001, beta_end=0.02, beta_schedule="linear", num_train_timesteps=1000)
                #elif sched == "PNDM":
                #    scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)    
                elif sched == "EULERv":    
                    scheduler = EulerDiscreteScheduler.from_pretrained(modelpath, subfolder="scheduler", prediction_type="v_prediction")
                elif sched == "EULERA":    
                    scheduler = EulerAncestralDiscreteScheduler.from_pretrained(modelpath, subfolder="scheduler")
                elif sched == "EULER":    
                    scheduler = EulerDiscreteScheduler.from_pretrained(modelpath, subfolder="scheduler")
                    
                scheduler.set_timesteps(num_inference_steps)
                changenotices.append("sched to "+sched)    
                api.sched_changed = False    
            
            if len(changenotices) > 0:
                api.status2 = "Changes made: "+",".join(changenotices)
                
                
             # model change
             
             # note that this may take long if the model is not yet in cache
             # therefore we update the status to the client between operations
                 
            if api.model_changed:
                modelpath = cnf.modeldir + "/" + api.model
                changenotices.append("LOADING MODEL "+api.model+" ...vae")
                api.status2 = "Changes made: "+",".join(changenotices)
                
                vae = AutoencoderKL.from_pretrained(
                        modelpath, subfolder="vae", use_auth_token=False
                )

                vae.eval()
                vae.cuda()

                changenotices[-1] = "LOADING MODEL "+api.model+" ...unet"
                api.status2 = "Changes made: "+", ".join(changenotices)

                unet = UNet2DConditionModel.from_pretrained(
                        modelpath, subfolder="unet", use_auth_token=False
                )
                unet.eval()
                unet.cuda()

                changenotices[-1] = "LOADING MODEL "+api.model+" ...text encoder"              
                api.status2 = "Changes made: "+", ".join(changenotices)
                

                tokenizer = CLIPTokenizer.from_pretrained(modelpath, subfolder="tokenizer")                
                text_encoder = CLIPTextModel.from_pretrained(modelpath, subfolder="text_encoder", use_auth_token=False).cuda()    

                text_embs, weights, pos_weights, neg_weights = get_embeddings_multi(text, nprompt)
                mask_embs, mweights, mpos_weights, mneg_weights = get_embeddings_multi(mline, nmline)
                   
                api.model_changed = False

            modelname = api.model
            api.status = "Using model: "+modelname+", scheduler: "+sched
                
            # time slider moved    
                                    
            if api.newsteps != steps:
               print("steps ->",api.newsteps)    
               steps = api.newsteps
               num_inference_steps = steps    
               scheduler.set_timesteps(num_inference_steps)
               offset = scheduler.config.get("steps_offset", 0)
               init_timestep = int(num_inference_steps * s) + offset
               init_timestep = min(init_timestep, num_inference_steps)

               timesteps = scheduler.timesteps[-init_timestep]
               timesteps = torch.tensor([timesteps] * bs, device=device)
            
            # if NEW, CONT or SEND is pressed or prompt changed
            
            changed, text, nprompt, mline, nmline = api.getChanges()
            
            api.status2 = ""
            if (api.resetL or api.newImg):
                break
                
            time.sleep(1)    
           
        
        print("////", latents.std(), changed, api.resetL, api.newImg, api.beta, s)
        
        # handle prompt change
        
        if changed:
            text_embs, weights, pos_weights, neg_weights = get_embeddings_multi(text, nprompt)
            print(mline, nmline)
            mask_embs, mweights, mpos_weights, mneg_weights = get_embeddings_multi(mline, nmline)
            changed = False
            api.newprompt = False  # mark change done to api
            
        # handle push change
            
        if api.beta != s:
            s = api.beta
                        
        if s < 1:
                offset = scheduler.config.get("steps_offset", 0)
                init_timestep = int(num_inference_steps * s) + offset
                init_timestep = min(init_timestep, num_inference_steps)

                timesteps = scheduler.timesteps[-init_timestep]
                timesteps = torch.tensor([timesteps] * bs, device=device)
                

                # add noise to latents using the timesteps
                noise = torch.randn(latents.shape, generator=generator, device=device) #, dtype=latents_dtype)
                latents_ = scheduler.add_noise(latents, noise, timesteps)
                
                latents = bg_w * latents_
                mlatents = fg_w * latents_.clone()
                    
                t_start = max(num_inference_steps - init_timestep + offset, 0) 
        else:
                latents_ = get_latents()
                latents = bg_w * latents_
                mlatents = fg_w * latents_.clone()
                t_start = 0
                
        total = num_inference_steps - t_start  
        api.total = total  
                
        # handle NEW change        
                
        if api.resetL: 
            print("handling resetL")   
            latents_ = get_latents()
            latents = bg_w * latents_
            mlatents = fg_w * latents_.clone()
            ctr = 0   # reset iterations counterk
            api.ctr = 0
            t_start = 0
            api.resetL = False
            total = steps - t_start
            api.total = total
            
        # handle init image SEND
            
        if api.newImg:
            image = "startimg.jpg"
            latents_ = get_latents(img=image, noisefactor=api.extranoise*0.6)
            latents = bg_w * latents_
            mlatents = fg_w * latents_.clone()
            api.newImg = False
            total = steps - t_start
            api.total = total    

        # handle mask image SEND
            
        if api.newMask:
            img = Image.open("./maskimg.jpg")
            mask = preprocess_mask(img).cuda()
            print("mask set on")
            usemask = True
            api.newMask = False
            
        if api.resetMask:
            usemask = False
            api.resetMask = False    
            
        bg_w = api.bg_w    
        fg_w = api.fg_w    
        api.total = total
        api.ctr = ctr    
        

