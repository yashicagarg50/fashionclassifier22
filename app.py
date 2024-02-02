import gradio as gr

from fastai.vision.all import *
import gradio as gr
def style(x): return x[0].isupper()


learn=load_learner('https://drive.google.com/file/d/1jjGAKMHxvL7xJIHXFXLBULOmln9eoavW/view?usp=sharing')

categories = ('active wear','bussiness wear','ethnic wear','goth fashion')
def classify_image(img):
    
    pred, idx, probs = learn.predict(img)
    return dict(zip(categories,map (float,probs)))

image = gr.inputs. Image (shape= (192, 192))
label = gr.outputs. Label()
# examples = ['/kaggle/input/ip-pics/goth.jpg','/kaggle/input/bizzzzzzzzzz/bizzzzzzz.jpg','/kaggle/input/piccccc/ethnic.jpg' ]
intf = gr. Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
intf.launch(inline=False)
