import streamlit as st
from PIL import Image
from PIL import ImageOps

from streamlit_drawable_canvas import st_canvas

def main():
    st.title('welcome to the segmentation world')
    with st.sidebar:
        upload = st.file_uploader('insert image',type=['png','jpg'])
        
    if upload:
        image = Image.open(upload)
        image = ImageOps.exif_transpose(image)
        if 'image' not in st.session_state:
            st.session_state['image'] = image
        
        st.image(image, caption='success') # image show
        st.write(image.size)
        
        
if __name__ == '__main__':
    main()