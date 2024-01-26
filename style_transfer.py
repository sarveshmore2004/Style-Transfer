import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import io

st.set_page_config(
    page_title="StyleTransfer",
    # layout="wide",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    st.write("***Click the X button above to close the instructions guide.***")
    st.divider()
    st.title("**Quick Start Guide**")
    st.write("1. ***Upload Images***: Select your content and style images.")
    st.write("2. ***Automatic Generation***: Wait as we transform your images.")
    st.write("3. ***Select and Download***: Choose your favorite and download.")
    st.header("**Example :**")
    st.write("*Input Images :*")
    st.image(Image.open("content.jpg"), caption="Content Image", use_column_width=True)
    st.image(Image.open("style.jpg"), caption="Style Image", use_column_width=True)
    st.write("")
    st.write("*Output Generated :*")
    st.image(Image.open("output.png"), caption="Stylized Image", use_column_width=True)


def load_img(img_path):
    img = img_path.getvalue()
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis, :]

    return img


def crop_to_square(image):
    shape = tf.shape(image)
    size = tf.minimum(shape[1], shape[2])
    offset_height = (shape[1] - size) // 2
    offset_width = (shape[2] - size) // 2
    image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, size, size)
    return image


def generate_img(content_path, style_path , max_dim = 1024):
    content_img = load_img(content_path)
    style_img = load_img(style_path)


    scale = max_dim / max(content_img.shape)
    if scale < 1:
        content_img = tf.image.resize(content_img, (round(content_img.shape[1] * scale), round(content_img.shape[2] * scale)))
    # print(content_img.shape)
    style_img = crop_to_square(style_img)
    style_img = tf.image.resize(style_img, (256, 256))
    # content_img = tf.image.resize(content_img, (256, 256))
    # st.image(np.squeeze(content_img))
    # st.image(np.squeeze(style_img))
    model = st.session_state['model']
    stylized_image = model(tf.constant(content_img), tf.constant(style_img))[0]
    return stylized_image


def image_to_bytes(img_array):
    img = Image.fromarray(np.uint8(img_array * 255))
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    return img_buffer.getvalue()


st.title('Style Transfer')
st.text('Imagine Your Photos in Creative Styles')
# st.write("")
st.divider()
hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
if 'model' not in st.session_state:
    with st.spinner("Loading . . ."):
        st.session_state['model'] = hub.load(hub_handle)

if 'key1' not in st.session_state:
    st.session_state['key1'] = 1
# model = hub.load(hub_handle)


col1 , col2 = st.columns(2)
with col1:
    content_img = st.file_uploader("***Upload Content Image***")
    if content_img is not None:
        st.write("*Uploaded Content Image:*")
        st.image(content_img , caption="Content Image" ,use_column_width=True)

with col2:
    style_img = st.file_uploader("***Upload Style Image***")
    if style_img is not None:
        st.write("*Uploaded Style Image:*")
        st.image(style_img , caption="Style Image" , use_column_width=True)

if content_img is None or style_img is  None:
    st.error("!!! Upload Both Images to see the Results")
if content_img is not None and style_img is not None:
    # print(content_img)
    stylized_images = []
    with st.spinner('Creating your stylized artwork...'):
        stylized_images.append(generate_img(content_img, style_img , 1024))
        stylized_images.append(generate_img(content_img, style_img , 764))
        stylized_images.append(generate_img(content_img, style_img , 512))
        stylized_images.append(generate_img(content_img, style_img , 256))

    st.divider()
    st.header('***Here are your stylized images!***')
    st.write("")
    for r in range(2):
        cols = st.columns(2)
        for i in range(2):
            with cols[i]:
                st.write("")
                st.write("")
                result_img = np.squeeze(stylized_images[2*r+i])
                st.image(result_img , use_column_width=True)
                st.download_button(
                    label=f"***Download V{2*r+i+1}***",
                    data=image_to_bytes(result_img),
                    file_name=f"stylized_image_{2*r+i+1}.png",
                    key=st.session_state['key1'],
                    mime="image/png"
                )
                st.session_state['key1'] += 1


st.markdown('---')
st.caption('WhatsApp Chat Analyzer v1.0 | Developed by Sarvesh_More')