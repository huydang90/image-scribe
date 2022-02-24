import io
import streamlit as st
import numpy as np
import pandas as pd
import torch
from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel
from tqdm.auto import tqdm

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
TEXT_EMBED_URL = "https://www.dropbox.com/s/5gvkjyjkolehnn9/text_embeds.npy?dl=1"
CAPTION_URL = "https://www.dropbox.com/s/n6s30qh1ldycko7/url2caption.csv?dl=1"
DEFAULT_IMAGE = "https://images.unsplash.com/photo-1490730141103-6cac27aaab94?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1470&q=80"


@st.cache(hash_funcs={CLIPModel: lambda _: None, CLIPProcessor: lambda _: None})
def load_model():
    # wget csv file with captions
    captions = pd.read_csv(CAPTION_URL)
    # wget text embeddings of above
    response = requests.get(TEXT_EMBED_URL)
    text_embeddings = torch.FloatTensor(np.load(io.BytesIO(response.content)), allow_pickle=True,fix_imports=True,encoding='latin1')
    # huggingface model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", from_tf=False).eval()
    for p in model.parameters():
        p.requires_grad = False
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    return model, processor, captions, text_embeddings


def get_image(url, model, processor):
    image = Image.open(requests.get(url, stream=True).raw)
    image_inputs = processor(images=image, return_tensors="pt",)
    image_inputs = {k: v.to(device) for k, v in image_inputs.items()}

    img_features = model.get_image_features(
        pixel_values=image_inputs["pixel_values"],
        output_attentions=False,
        output_hidden_states=False,
    )

    img_len = torch.sqrt((img_features ** 2).sum(dim=-1, keepdims=True))
    img_features = img_features / img_len
    return image, img_features


def get_best_captions(img_features, text_features, captions):
    similarity = img_features @ text_features.T
    img2text_similarity = similarity.softmax(dim=-1)
    _, idx = img2text_similarity.topk(dim=-1, k=5)

    st.write("## The closes captions are:")
    for caption in captions.loc[idx.cpu().numpy().ravel(), "caption"].values:
        st.write(caption)


model, processor, captions, text_embeddings = load_model()
st.header("Image Descriptor")
st.sidebar.subheader("Description")
st.sidebar.write("This is a simple prototype using the Hugging Face Transformers - CLIP model, which can be instructed in natural language to predict the most relevant text snippet, given an image.")
st.sidebar.subheader("Instruction")
st.sidebar.write("You can paste any link to a photo that you can find online into the box and press enter. The model will try to match what is in the image with a few description.")

url = st.text_input("Insert url of image", value=DEFAULT_IMAGE)
image, img_features = get_image(url, model, processor)
st.image(image)

get_best_captions(img_features, text_embeddings, captions)
