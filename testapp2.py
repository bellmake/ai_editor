import streamlit as st
from PIL import Image
from PIL import ImageOps

import torch
import numpy as np
import cv2
from matplotlib import pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

from streamlit_drawable_canvas import st_canvas

def main():
    st.title('welcome to the segmentation world')
    torch.cuda.empty_cache()
    with st.sidebar:
        upload = st.file_uploader('insert image',type=['png','jpg'])
        
        sam_button = st.button('generate mask')
        
    if upload:
        image = Image.open(upload)
        image = ImageOps.exif_transpose(image)
        
        image = np.array(image)
        
        # if 'image' not in st.session_state:
            # st.session_state['image'] = image
        st.session_state['image'] = image
        
        # st.image(image, caption='success') # image show
        # st.write(image.size)
        st.image(st.session_state['image'])
        
    if sam_button:
        generator = get_sam()
        
        
        
if __name__ == '__main__':
    main()