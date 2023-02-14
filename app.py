from flask import Flask
from flask import render_template, request, jsonify, send_from_directory
from synvilla_server import server, getAPI   
#from synvilla_api import SynvillaAPI
from flask import render_template, request, jsonify
app = Flask(__name__)
import os


import threading, logging

def pmx_server():
    print("Thread sdserver starting")
    server()
    print("Thread sdserver finishing")
 
pmx = threading.Thread(target=server, daemon = True)
pmx.start()
api = getAPI()
   
@app.route('/data')
def gdataobj():
    d = api.getDataObj()
    d.text = cleant(d.text)
    d.ntext = cleant(d.ntext)
    #print("APPo", d.text)
    d.h = str(d.h)
    d.w = str(d.w)
    d = d.__dict__
    return jsonify(d)   

def cleant(t):
    if t == None:
        t = ""
    t = t.replace("<","&lt;").replace(">","&gt;")
    return t

@app.route('/')
@app.route('/index')
def client():
    return send_from_directory('templates', 'client2b.html')
    
@app.route('/img')
@app.route('/client')
def img():
   #text, np, i, total, beta, steps, lr, seed, _, _ = getData()
   o = a.getDataObj()
   ifn  = 'result.jpg?'+str(i.i)
   #print(o.text, o.i, o.total)
   c = cleant(o.text)+' ('+str(o.i)+ "/" + str(o.total)+')'
   html = render_template('img.html', caption=c, ifn=ifn, prompt=o.text)
   return html

    
@app.route('/prompts', methods=['POST'])
def prompts():
    #print("APP", request.json)
    text = request.json['prompt'] 
    ntext = request.json['nprompt']
    mtext = request.json['mprompt']  
    mntext = request.json['nmprompt'] 
    
    #print("APP",text,ntext, mtext, mntext)
    
    api.setText(text, ntext, mtext, mntext)
    return cleant(text)    
    
@app.route('/nexts', methods=['POST'])
def nexts():
    text = request.json['prompt'] 
    ntext = request.json['nprompt']
    mtext = request.json['mprompt']
    nmtext = request.json['nmprompt']
    #print("APP next",text,ntext, mtext, mntext)
    api.setText(text, ntext, mtext, nmtext)
    os.system("cp static/result.jpg startimg.jpg")
    api.setImg()
    return cleant(text)    

@app.route('/reset',methods=['POST'])
def reset():
   api.resetLats()
   return "ok"

@app.route('/resetmask',methods=['POST'])
def resetm():
   api.clearMask()
   return "ok"
   
'''''

@app.route('/resetbmask',methods=['POST'])
def resetbm():
   api.resetBMask()
   return "ok"
  
@app.route('/pause',methods=['POST'])
def pausef():
   pause()
   return "ok"

@app.route('/resume',methods=['POST'])
def resumef():
   resume()
   return "ok"
'''
@app.route('/steps', methods=['POST'])
def noise():
    n = request.json['steps']
    api.setSteps(int(n))
    return "ok"   

@app.route('/beta', methods=['POST'])
def beta():
    b = request.json['beta']
    #print("APP beta in ",b)
    api.setBeta(float(b))
    return "ok"   

@app.route('/lr', methods=['POST'])
def lrset():
    lr = float(request.json['lr'])
    api.setLR(lr)
    return "ok"   
    
@app.route('/noise', methods=['POST'])
def enoise():
    n = request.json['noise']
    api.setNoise(float(n))
    return "ok"       

@app.route('/bgw', methods=['POST'])
def bgw():
    n = request.json['bgw']
    api.setBgw(float(n))
    return "ok"       

@app.route('/fgw', methods=['POST'])
def fgw():
    n = request.json['fgw']
    api.setFgw(float(n))
    return "ok"       

@app.route('/iter', methods=['POST'])
def iter():
    it = request.json['iter']
    #api.setIter(int(it))
    return "ok"       

@app.route('/seed', methods=['POST'])
def seed():
    s = request.json['seed']
    api.setSeed(int(s))
    return "ok"  

@app.route('/seedlock', methods=['POST'])
def seedlock():
    s = request.json['lock']
    #api.setLock(int(s))
    return "ok"  

@app.route('/gamma', methods=['POST'])
def gamma():
    g = request.json['gamma']
    api.setGamma(float(g))
    return "ok"  

@app.route('/contrast', methods=['POST'])
def contrast():
    c = request.json['contrast']
    api.setContrast(float(c))
    return "ok"  


@app.route('/blend', methods=['POST'])
def blend():
    b = request.json['blend']
    api.setBlend(int(b))
    return "ok"  
 
@app.route('/inpaint', methods=['POST'])
def inpaint():
    b = request.json['inpaint']
    api.setInpaint(int(b))
    return "ok"  
       
@app.route('/startimg',methods=[ 'POST'])
def uploadImg():
    isthisFile=request.files.get('file')
    if not isthisFile:
        return "nok"
    else:
        print(request)    
    print(isthisFile)
    print(isthisFile.filename)
    isthisFile.save("./startimg.jpg") #+isthisFile.filename) 
    api.setImg()
    return "ok"   

@app.route('/maskimg',methods=[ 'POST'])
def uploadMask():
    isthisFile=request.files.get('file')
    print(isthisFile)
    print(isthisFile.filename)
    isthisFile.save("./maskimg.jpg") #+isthisFile.filename) 
    api.setMask()
    return "ok"   

'''''    
@app.route('/bmask',methods=[ 'POST'])
def uploadBMask():
    isthisFile=request.files.get('file')
    print(isthisFile)
    print(isthisFile.filename)
    isthisFile.save("./bmaskimg.jpg") #+isthisFile.filename) 
    #api.setBMask()
    return "ok"
'''
        
@app.route('/settings', methods=['POST'])
def settings():
    h = request.json['h']
    w = request.json['w']
    m = request.json['model']
    s = request.json['sched']
    api.setSettings(int(h), int(w), m, s)
    return "ok"      


if __name__ == '__main__':
  app.run(host='0.0.0.0')


