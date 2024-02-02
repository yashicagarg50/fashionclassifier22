from collections import Counter, defaultdict, namedtuple, OrderedDict
from collections.abc import Iterable


from fastai.vision.all import *
import gradio as gr
def style(x): return x[0].isupper()


learn = load_learner('model.pkl')

categories = ('active wear','bussiness wear','ethnic wear','goth fashion')

def classify_image(img):
    pred, idx, probs = learn.predict(img)
    return dict(zip(categories,map (float,probs)))

image = gr.inputs.Image(shape=(192, 192))
label = gr.outputs.Label()

# intf = gr. Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)

intf = gr.Interface(fn=classify_image,inputs=gr.Image(type="pil"),outputs=gr.Label(),examples=['business.jpg','goth.jpeg'])
intf.launch(inline=False)
