import threading


# ---------------------------------------------------

# class for API

class SynvillaAPI:
    
  _instance = None
  _lock = threading.Lock()

  def __new__(cls):
         if cls._instance is None: 
             with cls._lock:
                 # Another thread could have created the instance
                 # before we acquired the lock. So check that the
                 # instance is still nonexistent.
                 if not cls._instance:
                     cls._instance = super().__new__(cls)
         return cls._instance
  

             
  def __init__(self):    
      self.status = ""
      self.status2 = ""
      self.version = ""
      
      self.text = "" #text 
      self.ntext = "" #nprompt
      self.mtext = ""
      self.nmtext = ""
      
      self.resetL = False # reset latents
      self.newImg = False # new init image
      self.newMask = False
      self.resetMask = False
      self.resetImg = False # reset init image

      s = 0.5
      self.beta = s
      self.guidance = 0 #newLR = None
      self.steps = 50
      self.newsteps = self.steps
      
      self.ctr = None # iteration number
      self.extranoise = 0
      self.change_i = None
      self.seed = 0 #newseed = 0
      self.seedlock = False

      self.blend = 0
      self.mline = ""
      self.nmline = ""
      self.inpaint = 0
      
      #simg = False
      
      self.gamma = 1
      self.contrast = 1
      self.newprompt = False

      self.maxnoise = 0.6
      self.total = 0
      self.bg_w = 1
      self.fg_w = 1
      
      self.h = 768
      self.w = 768
      self.schedlist = []
      self.modelist = []
      self.model = ""
      self.sched = "" 
      
      self.wh_changed = False
      self.model_changed = False
      self.sched_changed = False
 
  def getDataObj(self):      
     
     d = type('', (), {})()
     d.text = self.text
     d.ntext = self.ntext
     d.mtext = self.mtext
     d.nmtext = self.nmtext
     d.i = self.ctr
     d.n = self.total
     d.beta = self.beta
     d.steps = self.newsteps
     d.lr = self.guidance
     d.seed = self.seed
     #d.fg = self.fg_w
     d.bgw = self.bg_w
     d.fgw = self.fg_w
     d.blend = self.blend
     d.gamma = self.gamma
     d.noise = self.extranoise
     d.status = self.status
     d.status2 = self.status2
     d.version = self.version
     
     # todo, include these only when needed
     d.modellist = self.modellist
     d.schedlist = self.schedlist
     d.model = self.model
     d.sched = self.sched
     d.h = self.h
     d.w = self.w
     d.inpaint = self.inpaint
     d.contrast = self.contrast
     
 
     #print("getobj", d.text)
 
     return d
 
  def setSettings(self, h, w, m, s):
      if (h != self.h) or (w != self.w):
          self.wh_changed = True
          self.h = h
          self.w = w
          
      if (m != self.model):
          self.model_changed = True
          self.model = m
          
      if (s != self.sched):
          self.sched_changed = True
          self.sched = s
          
      return             
  

  def setLock(self, s):
    if int(s) == 0:
        self.seedlock = False
    else:
        self.seedlock = True
    return    

  def setSeed(self, s):
    self.seed = int(s)
    return

  def setBlend(self, s):
    self.blend = int(s)
    print("blend set to ",s)
    return

  def setInpaint(self, s):
    self.inpaint = int(s)

  def setGamma(self, s):
    self.gamma = float(s)
    return

  def setContrast(self, s):
    self.contrast = float(s)
    return

  def resetLats(self):
    self.resetL = True
    print("set lats to be reseted")
    return

  def setText(self, text, ntext, mtext, nmtext):
    print("new prompt:", text, "neg:", ntext)
    print("new mask prompt:", mtext, "neg:", nmtext)
    self.text = text
    self.ntext = ntext
    self.mtext = mtext
    self.nmtext = nmtext
    self.newprompt = True
    return    

  def setBeta(self, val):
    print("new beta:", val, " s=",self.beta)
    self.beta = float(val)
    return    

  def setSteps(self, val):
    print("new steps:", val)
    self.newsteps = int(val)
    return    

  '''''    
  def setIter(val):
    print("jump to iter:", val)
    self.change_i = int(val)
    return    
  '''
        
  def setLR(self, val):
    print("new g:", val)
    self.guidance = float(val)
    return        
    
  def setNoise(self, val):
    print("extra noise:", val)
    self.extranoise = float(val)
    return        

  def setBgw(self, val):
    print("attenuate:", val)
    self.bg_w = float(val)
    return 

  def setFgw(self, val):
    print("fwg:", val)
    self.fg_w = float(val)
    return 
 
    
  def setImg(self):
    print("img received")
    self.newImg = True
    return

  
  def setMask(self):
    print("mask received")
    self.newMask = True
    return

  def clearMask(self):
    global resetMask
    print("setting mask to none")
    self.resetMask = True
    return


  def getChanges(self):
    go = self.newprompt
    #self.newprompt = False    
    return go, self.text, self.ntext, self.mtext, self.nmtext 


# ---------------------------------------------------------