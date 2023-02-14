# synvilla_private

![Näyttökuva 2023-2-14 kello 9 42 25](https://user-images.githubusercontent.com/15064373/218671231-405729d9-065d-4282-a724-6a60d0c31df1.png)


## A responsive progressive image synth

Synvilla is designed to be progressive in the sense of "happening or developing gradually or in stages". The image is maybe never finished, you can continue to make it evolve further and further. You can also change direction at any time, making it responsive. Technically, the idea is simple, but in practice the experience of using Synvilla is quite different from a typical "type prompt, press button and wait" image synth. A limited number of controls, when applied to an evolving image time after time, results in a myriad of possible routes for the image to develop and change. 

## Installation
```
# basic install

conda create -n synvilla python=3.9
conda activate synvilla
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install diffusers transformers accelerate scipy safetensors

conda install imageio
conda install scikit-image 
conda install tqdm
conda install ftfy
conda install regex

# flask server

sudo apt install python3-flask
conda install flask

# uwsgi, if server to be accessed from the web

conda install uwsgi -c conda-forge
```

## Server operation

**Prepare ini file:**

Example:

```
[proc]

[image]
h = 768
w = 768

[run]
iters = 25
path = 
name = 
prompt = an abstract watercolor
g = 7.5
model = ./models/stable-diffusion-v1-4
modeldir = ./models
tknzr = 
seed = 0

sched = EULER
```

Explanation:

```
h, w:    height and width of the generated image 
iters:   number of iterations*
path:    set to a path to store all generated frames (a lot!), leave empty otherwise
name:    base for the name of stored frames
prompt:  initial prompt*
g:       initial guidance*
model:   path to folder of the SD model to be used
tknzr:   path to a tokenizer, if different from the model
seed:    initial seed, use 0 for random
```
Settings marked * are initial values for settings which can then be changed from the UI

**Start Flask server (for access from LAN only):**

In the synvilla folder:

```
conda activate synvilla
python app.py
```

**models**

Download models from Huggingface (see links below) and place them in ./models

https://huggingface.co/CompVis/stable-diffusion-v1-4
https://huggingface.co/runwayml/stable-diffusion-v1-5
https://huggingface.co/stabilityai/stable-diffusion-2
https://huggingface.co/stabilityai/stable-diffusion-2-1
https://huggingface.co/22h/vintedois-diffusion-v0-1


## Use

Start the client on your browser by going localhost:5000 or url-of-your-server:5000 if you access from another computer.

**Text to image**
  
  ![Näyttökuva 2023-1-30 kello 18 16 32](https://user-images.githubusercontent.com/15064373/215532700-0d290af2-63c4-4a60-b104-d7bd18d8edd5.png)

Type a prompt and press **Prompts**. Press **New**.

Use **Download** button at any time to save the current image in the downloads folder of your computer.

Continue by changing the prompt and pressing **New** to start from scratch, or **Cont** to continue from the current image using the new prompt.

Try using a negative prompt, too.

In addition to simple prompts, yiu can also split the prompt to subprompts, each with a different weight. Use the following format:

Sunset at the north pole: 50 / a watercolor in an optimistic style:70 / an abstract feeling of grey longing: 40

Be careful not to omit anything (no syntax checking implemented yet)

With **Time** you can make the image development faster (left) or slower with possibly better quality (right)    
  
**Image to image**
  
  ![Näyttökuva 2023-1-30 kello 18 16 40](https://user-images.githubusercontent.com/15064373/215532796-dc291e7d-45ae-4e70-841b-138560496d21.png)

Use the init image selector to select an image. Set your prompts and press **Prompt**.

Adjust **Push** to set the balance between the init image (slider to the left) and the prompt (slider to the right).

Press **Img** to start. 

Again, you can for instance change the prompt and press **Cont** to continue from the current image. Or press **Send** to start again from the selected init image. 
  
**Brightness** can be used to adjust the brightness of the init image. **Add noise** will add some random variation to the init image. It also helps if you have white or black areas in your image: added noise helps them develop too.
  
**Other controls**

  ![Näyttökuva 2023-1-30 kello 18 17 18](https://user-images.githubusercontent.com/15064373/215532919-fc964448-d07f-46e8-b6eb-8497b8e25d6b.png)

**Complexity** can be adjusted to the left to reduce the complexity of the image. 

**Blend** can be useful if Synvilla is used to generate live visuals. If set to more than zero, transition between the previous and new generation round (after pressing New, Cont or Img) will use cross-blending to produce a smoother look.
  
**Info**
  
  ![Näyttökuva 2023-1-30 kello 18 17 30](https://user-images.githubusercontent.com/15064373/215533016-344cfddf-3c24-4e7b-a763-cbbda51195e1.png)
  
Below the image frame you will the current prompts, as well as the number of the current iteration vs. the total number of iterations.
  
  ![Näyttökuva 2023-1-30 kello 18 17 35](https://user-images.githubusercontent.com/15064373/215533054-96086cf5-b0a8-42ed-9d78-ac09a1ed428b.png)
  
Above the image you see a status line, showing which model and scheduler is being used.
  
**Settings**  
 
Pressing Settings will open a popup allowing you to change image dimensions, model and scheduler used. Changes will usually not take effect until the current image has fully evolved (i.e. current iteration is equal to the number of total iterations, as in 50/50). Model change, especially, can take quite long, and you should observe the status line above the image frame to see when the operation has finished.   

You can also open a panel for using a mask (e.g. for inpainting). 

## Using a mask

A mask is a black and white image which can be used to restrict changes to a part of the canvas. In "inpainting", a part of the image is made to develop while the rest of the image is kept unchanged. You can also use a mask to make two areas develop simultaneously but accorging to separate prompts. 

In the Synvilla UI, the mask panel is initially hidden to keep everything look simple. To enable masking, open settings and select "open mask panel".

![Näyttökuva 2023-2-3 kello 11 34 37](https://user-images.githubusercontent.com/15064373/216565003-3590b4af-5959-43fb-9c1c-6c30f0c5c1fb.png)

Check inpaint if you want the rest of the canvas remain unchanged. Otherwise the whole canvas will evolve, with the mask determining which part will use which prompt.

To enable masking, type a mask prompt and select a mask image from your computer. A mask image should be monochrome, white for the areas to follow mask prompts, black for the areas not to be changed (or to follow main prompts).

Remember to press PROMPTS when you want your changed prompts to take effect. There are two PROMPTS buttons for convenience, both will send all prompts.

You can now use CONT, NEW and IMG as before, the mask will just restrict the changes to a specific area as described above.

To stop masking, press x in the mask image selector.

## Acknowledgements

This software is deeply dependent on Diffusers library (https://github.com/huggingface/diffusers) and the code of the server engine makes extensive use of techniques learned from the various diffusers pipelines. 


