import os
import torch
import clip
from PIL import Image
import streamlit as st
from tqdm import tqdm
from datetime import datetime
import shutil
import pickle
from googletrans import Translator

# Page config
st.set_page_config(layout="wide")
st.sidebar.title("AI-PhotoSearch")

# --- Custom CSS for equal height input and button ---
st.markdown("""
    <style>
    .equal-height-row {
        display: flex;
        align-items: stretch !important;
        gap: 1em;
        margin-bottom: 1em;
    }
    .equal-height-row > div {
        flex: 1 1 0%;
        display: flex;
        align-items: stretch !important;
    }
    .equal-height-row .stButton button {
        height: 100%;
        min-height: 0;
        padding-top: 0;
        padding-bottom: 0;
    }
    .equal-height-row .stTextInput > div > div {
        height: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# Constants
VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.cr2', '.nef', '.arw', '.dng', '.psd')
CONFIG_FILE = "last_used_paths.pkl"
MAX_HISTORY = 5
CACHE_DIR = "cache"

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def save_last_folder(path):
    if not path:
        return
    history = load_last_folders()
    if path in history:
        history.remove(path)
    history.insert(0, path)
    history = history[:MAX_HISTORY]
    with open(CONFIG_FILE, "wb") as f:
        pickle.dump(history, f)

def load_last_folders():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "rb") as f:
            return pickle.load(f)
    return []

def clear_saved_paths():
    if os.path.exists(CONFIG_FILE):
        os.remove(CONFIG_FILE)

def get_cache_file(folder):
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"clip_cache_{os.path.basename(folder)}.pkl")

def load_or_update_cache(folder):
    CACHE_FILE = get_cache_file(folder)

    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            cache = pickle.load(f)
    else:
        cache = {"paths": [], "features": []}

    all_paths = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(VALID_EXTENSIONS):
                all_paths.append(os.path.join(root, f))

    current_paths_set = set(all_paths)
    cached_paths_set = set(cache["paths"])

    removed_paths = cached_paths_set - current_paths_set
    if removed_paths:
        keep_indices = [i for i, p in enumerate(cache["paths"]) if p in current_paths_set]
        cache["paths"] = [cache["paths"][i] for i in keep_indices]
        cache["features"] = [cache["features"][i] for i in keep_indices]

    new_paths = [p for p in all_paths if p not in cached_paths_set]
    if new_paths:
        st.sidebar.info(f"🔍 Indexing new images: {len(new_paths)}")
        for path in tqdm(new_paths):
            try:
                image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
                with torch.no_grad():
                    feat = model.encode_image(image)
                    feat = (feat / feat.norm(dim=-1, keepdim=True)).cpu()
                cache["paths"].append(path)
                cache["features"].append(feat)
            except Exception as e:
                continue

    if not cache["features"]:
        st.error("❌ No suitable images found.")
        st.stop()

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)

    all_features = torch.cat(cache["features"], dim=0)
    return cache["paths"], all_features

# --- Sidebar interface ---
st.sidebar.header("Image source")

# Get the first existing path from history, else empty
last_folders = load_last_folders()
default_folder = ""
for path in last_folders:
    if os.path.exists(path):
        default_folder = path
        break
IMAGE_FOLDER = st.sidebar.text_input("Folder path:", value=default_folder)

st.sidebar.markdown("<div style='width:100%'>", unsafe_allow_html=True)

if IMAGE_FOLDER and os.path.exists(IMAGE_FOLDER):
    save_last_folder(IMAGE_FOLDER)
else:
    st.warning("⚠️ Specify the path to the folder with images. Right-click the folder → “Copy as path”")
    st.stop()

image_paths, image_features = load_or_update_cache(IMAGE_FOLDER)

st.sidebar.header("Search settings")
translator = Translator()

query_clip = st.sidebar.text_input("Image description:", "")
translated = ""
if query_clip:
    try:
        detected = translator.detect(query_clip)
        src_lang = detected.lang
        dest_lang = "en" if src_lang == "ru" else "ru"
        translated = translator.translate(query_clip, src=src_lang, dest=dest_lang).text
        st.sidebar.markdown(f"🌐 **{translated}**")
    except Exception as e:
        translated = query_clip

query_name = st.sidebar.text_input("File name:", "")
num_columns = st.sidebar.slider("Scale", 1, 8, 3)
max_results = st.sidebar.slider("Number of results", 5, 1000, 30, step=5)
show_all = st.sidebar.checkbox("Show all")

# --- Show images without search ---
if not query_clip and not query_name:
    if show_all:
        num_show = len(image_paths)
    else:
        num_show = min(max_results, len(image_paths))
    cols = st.columns(num_columns)
    for i in range(num_show):
        img_path = image_paths[i]
        img_name = os.path.basename(img_path)
        with cols[i % num_columns]:
            st.image(img_path, use_container_width=True, caption=img_name)

# --- Search by file name (substring, no CLIP) ---
if not query_clip and query_name:
    filtered = [
        (i, os.path.basename(p))
        for i, p in enumerate(image_paths)
        if query_name.lower() in os.path.basename(p).lower()
    ]
    if show_all:
        show_inds = [i for i, _ in filtered]
    else:
        show_inds = [i for i, _ in filtered][:max_results]
    cols = st.columns(num_columns)
    selected_images = []
    for idx, i in enumerate(show_inds):
        img_path = image_paths[i]
        img_name = os.path.basename(img_path)
        with cols[idx % num_columns]:
            st.image(img_path, use_container_width=True, caption=img_name)
            selected_images.append(img_path)
    st.markdown("---")
    st.markdown('<div class="equal-height-row">', unsafe_allow_html=True)
    custom_folder = st.text_input("Save found images to", value="search_results", key="save_input_1", label_visibility="visible")
    save_clicked = st.button("Save", key="save_button_1")
    st.markdown('</div>', unsafe_allow_html=True)
    if save_clicked:
        save_path = os.path.join(custom_folder, datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(save_path, exist_ok=True)
        for src in selected_images:
            try:
                shutil.copy2(src, save_path)
            except Exception as e:
                continue
        st.success(f"✅ Images saved to: `{save_path}`")

# --- Search by description (CLIP) or joint CLIP+name ---
if query_clip:
    try:
        translated = translator.translate(query_clip, src='auto', dest='en').text
    except Exception as e:
        translated = query_clip

    with torch.no_grad():
        text_tokens = clip.tokenize([translated]).to(device)
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarities_clip = (image_features @ text_features.T).squeeze()

    filename_scores = []
    for path in image_paths:
        name = os.path.basename(path).lower()
        score = 1.0 if query_name and query_name.lower() in name else 0.0
        filename_scores.append(score)
    filename_scores = torch.tensor(filename_scores)
    similarities = similarities_clip + filename_scores

    if show_all:
        top_k = len(similarities)
    else:
        top_k = min(max_results, len(similarities))
    top_indices = similarities.topk(top_k).indices

    cols = st.columns(num_columns)
    selected_images = []
    for idx, i in enumerate(top_indices):
        img_path = image_paths[i]
        img_name = os.path.basename(img_path)
        with cols[idx % num_columns]:
            st.image(img_path, use_container_width=True, caption=img_name)
            selected_images.append(img_path)

    st.markdown("---")
    st.markdown('<div class="equal-height-row">', unsafe_allow_html=True)
    custom_folder = st.text_input("Where to save the found images", value="search_results", key="save_input_2", label_visibility="visible")
    save_clicked = st.button("Save", key="save_button_2")
    st.markdown('</div>', unsafe_allow_html=True)
    if save_clicked:
        save_path = os.path.join(custom_folder, datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(save_path, exist_ok=True)
        for src in selected_images:
            try:
                shutil.copy2(src, save_path)
            except Exception as e:
                continue
        st.success(f"✅ Images saved to: `{save_path}`")
