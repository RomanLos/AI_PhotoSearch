import os
import torch
import open_clip
from PIL import Image
import streamlit as st
from torchvision import transforms
import pickle
import time
from googletrans import Translator
import numpy as np

st.set_page_config(layout="wide")
st.sidebar.title("AI-PhotoSearch")

# --- CLIP ---
SUPPORTED_ARCHS = ["ViT-B-32", "ViT-B-16", "ViT-L-14", "ViT-L-14-336"]
MODELS_DIR = "models"

def get_available_models(models_dir, archs):
    models = []
    for arch in archs:
        pt_path = os.path.join(models_dir, f"{arch}.pt")
        st_path = os.path.join(models_dir, f"{arch}.safetensors")
        if os.path.isfile(pt_path):
            models.append((arch, pt_path))
        elif os.path.isfile(st_path):
            models.append((arch, st_path))
    return models

available_models = get_available_models(MODELS_DIR, SUPPORTED_ARCHS)

if not available_models:
    st.sidebar.error("No supported CLIP models found in the models folder!")
    st.stop()

arch_names = [m[0] for m in available_models]
selected_arch = st.sidebar.selectbox("Select a CLIP model", arch_names)
model_path = dict(available_models)[selected_arch]
clip_model_name = selected_arch

device = "cuda" if torch.cuda.is_available() else "cpu"

# Показываем информацию об устройстве в sidebar
if device == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    st.sidebar.success(f"🚀 GPU: {gpu_name}")
else:
    st.sidebar.warning("⚠️ Running on CPU (slower)")

# --- CLIP модель ---
try:
    model, _, preprocess = open_clip.create_model_and_transforms(
        clip_model_name,
        pretrained=model_path
    )
    model = model.to(device)
except Exception as e:
    st.sidebar.error(f"CLIP model loading error {clip_model_name}: {e}")
    st.stop()

# --- DINOv2-B (ViT-B/14) ---
DINOV2_MODEL_PATH = os.path.join("models", "dinov2_vitb14_pretrain.pth")
if not os.path.isfile(DINOV2_MODEL_PATH):
    st.sidebar.error(f"DINOv2-B model not found! Please download dinov2_vitb14_pretrain.pth and place it in models/")
    st.stop()

@st.cache_resource(show_spinner=True)
def load_dino():
    weights = torch.load(DINOV2_MODEL_PATH, map_location="cpu")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    model.load_state_dict(weights)
    return model.eval()

dinov2 = load_dino()
dinov2 = dinov2.to(device)  # Явно перемещаем DINOv2 на GPU
dino_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_dino_cache_file(folder, model_name="dinov2_vitb14"):
    os.makedirs("cache", exist_ok=True)
    base = os.path.basename(os.path.abspath(folder))
    return os.path.join("cache", f"dino_embeddings_{model_name}_{base}.pt")

@st.cache_data(show_spinner=False)
def index_images_with_dino_cache(folder):
    cache_file = get_dino_cache_file(folder)
    
    # Получаем все текущие файлы
    all_files = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                all_files.append(os.path.join(root, f))
    
    # Загружаем существующий кеш или создаем новый
    if os.path.exists(cache_file):
        data = torch.load(cache_file)
        cached_files = data.get("files", [])
        cached_embeddings = data.get("embeddings", [])
    else:
        cached_files = []
        cached_embeddings = []
    
    current_files_set = set(all_files)
    cached_files_set = set(cached_files)
    
    # Удаляем из кеша файлы, которых больше нет
    removed_files = cached_files_set - current_files_set
    if removed_files:
        keep_indices = [i for i, f in enumerate(cached_files) if f in current_files_set]
        cached_files = [cached_files[i] for i in keep_indices]
        if len(cached_embeddings) > 0:
            cached_embeddings = [cached_embeddings[i] for i in keep_indices]
    
    # Находим новые файлы для индексации
    new_files = [f for f in all_files if f not in cached_files_set]
    
    if new_files:
        st.sidebar.info(f"🔍 Indexing new images with DINOv2: {len(new_files)}")
        progress_bar = st.sidebar.progress(0, text="")
        
        new_embeddings = []
        batch_size = 8  # Размер пакета для DINOv2
        
        for i in range(0, len(new_files), batch_size):
            batch_files = new_files[i:i+batch_size]
            batch_tensors = []
            valid_files = []
            
            # Подготавливаем пакет изображений
            for path in batch_files:
                try:
                    img = Image.open(path).convert("RGB")
                    tensor = dino_transform(img)
                    batch_tensors.append(tensor)
                    valid_files.append(path)
                except Exception as e:
                    continue
            
            if batch_tensors:
                # Обрабатываем весь пакет сразу
                batch_tensor = torch.stack(batch_tensors).to(device)
                with torch.no_grad():
                    batch_embs = dinov2(batch_tensor)
                    batch_embs = batch_embs / batch_embs.norm(dim=-1, keepdim=True)
                
                # Сохраняем результаты
                for j, path in enumerate(valid_files):
                    new_embeddings.append(batch_embs[j].cpu())
                    cached_files.append(path)
            
            progress_bar.progress(min(i + batch_size, len(new_files)) / len(new_files), 
                                text=f"Indexed {min(i + batch_size, len(new_files))} out of {len(new_files)}")
        
        progress_bar.empty()
        
        # Объединяем старые и новые эмбеддинги
        if len(cached_embeddings) > 0 and len(new_embeddings) > 0:
            # Есть и старые, и новые
            if isinstance(cached_embeddings, torch.Tensor):
                # Старые уже в виде тензора
                new_tensor = torch.stack(new_embeddings)
                embeddings_tensor = torch.cat([cached_embeddings, new_tensor], dim=0)
            else:
                # Старые в виде списка тензоров
                old_tensor = torch.stack(cached_embeddings) if isinstance(cached_embeddings, list) else cached_embeddings
                new_tensor = torch.stack(new_embeddings)
                embeddings_tensor = torch.cat([old_tensor, new_tensor], dim=0)
        elif len(new_embeddings) > 0:
            # Только новые
            embeddings_tensor = torch.stack(new_embeddings)
        elif len(cached_embeddings) > 0:
            # Только старые
            embeddings_tensor = cached_embeddings if isinstance(cached_embeddings, torch.Tensor) else torch.stack(cached_embeddings)
        else:
            # Ничего нет
            embeddings_tensor = torch.empty((0, 768))
    else:
        # Новых файлов нет
        if len(cached_embeddings) > 0:
            embeddings_tensor = cached_embeddings if isinstance(cached_embeddings, torch.Tensor) else torch.stack(cached_embeddings)
        else:
            embeddings_tensor = torch.empty((0, 768))
    
    # Если нет эмбеддингов, возвращаем пустые списки
    if embeddings_tensor.numel() == 0:
        return [], torch.empty((0, 768))  # 768 - размерность DINOv2
    
    # Сохраняем обновленный кеш
    torch.save({"files": cached_files, "embeddings": embeddings_tensor}, cache_file)
    
    return cached_files, embeddings_tensor

# --- Основной код CLIP ---
VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.cr2', '.nef', '.arw', '.dng', '.psd')
CONFIG_FILE = "last_used_paths.pkl"
MAX_HISTORY = 5
CACHE_DIR = "cache"

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

def get_cache_file(folder, model_name):
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"clip_cache_{model_name}_{os.path.basename(folder)}.pkl")

def load_or_update_cache(folder, model_name):
    CACHE_FILE = get_cache_file(folder, model_name)

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
        progress = st.sidebar.progress(0, text="Indexing...")
        batch_size = 4
        total = len(new_paths)
        for i in range(0, total, batch_size):
            batch_paths = new_paths[i:i+batch_size]
            images = []
            img_names = []
            for p in batch_paths:
                try:
                    img = preprocess(Image.open(p).convert("RGB")).unsqueeze(0)
                    images.append(img)
                    img_names.append(p)
                except Exception as e:
                    continue
            if not images:
                continue
            images_tensor = torch.cat(images, dim=0).to(device)
            with torch.no_grad():
                feats = model.encode_image(images_tensor)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            for j, path in enumerate(img_names):
                cache["paths"].append(path)
                cache["features"].append(feats[j:j+1].cpu())
            progress.progress(min(i + batch_size, total) / total, text=f"Indexed {min(i + batch_size, total)} out of {total}")
        progress.empty()

    if not cache["features"]:
        st.error("❌ No suitable images found.")
        st.stop()

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)

    all_features = torch.cat(cache["features"], dim=0)
    return cache["paths"], all_features

# --- Интерфейс ---
st.sidebar.header("Image source")
last_folders = load_last_folders()

# Если есть история, показываем selectbox как у CLIP
if last_folders:
    existing_paths = [path for path in last_folders if os.path.exists(path)]
    if existing_paths:
        # Добавляем опцию для ввода нового пути
        folder_options = ["Browse new folder..."] + existing_paths
        
        # По умолчанию выбираем первый существующий путь
        default_index = 1 if len(folder_options) > 1 else 0
        
        selected_option = st.sidebar.selectbox("Select folder:", folder_options, index=default_index)
        
        if selected_option == "Browse new folder...":
            IMAGE_FOLDER = st.sidebar.text_input("", "", placeholder="Paste folder path here...")
        else:
            IMAGE_FOLDER = selected_option
    else:
        # Нет существующих путей в истории
        IMAGE_FOLDER = st.sidebar.text_input("Folder path:", "", placeholder="Paste folder path here...")
else:
    # Нет истории совсем
    IMAGE_FOLDER = st.sidebar.text_input("Folder path:", "", placeholder="Paste folder path here...")

st.sidebar.markdown("<div style='width:100%'>", unsafe_allow_html=True)
if IMAGE_FOLDER and os.path.exists(IMAGE_FOLDER):
    save_last_folder(IMAGE_FOLDER)
else:
    st.warning("⚠️ Specify the path to the folder with images. Right-click the folder → \"Copy as path\"")
    st.stop()

image_paths, image_features = load_or_update_cache(IMAGE_FOLDER, clip_model_name)
image_features = image_features.to(device)
image_features = image_features.float()

# Создаем словарь для быстрого поиска индексов
path_to_index = {path: i for i, path in enumerate(image_paths)}

st.sidebar.header("Search settings")

# Кеш для переводов
if 'translation_cache' not in st.session_state:
    st.session_state.translation_cache = {}

def get_translation(text):
    """Кешированный перевод"""
    if not text:
        return ""
    
    if text in st.session_state.translation_cache:
        return st.session_state.translation_cache[text]
    
    try:
        translator = Translator()
        detected = translator.detect(text)
        src_lang = detected.lang
        dest_lang = "en" if src_lang == "ru" else "ru"
        translated = translator.translate(text, src=src_lang, dest=dest_lang).text
        st.session_state.translation_cache[text] = translated
        return translated
    except Exception as e:
        st.session_state.translation_cache[text] = text
        return text

query_clip = st.sidebar.text_input("Image description:", "")
translated = ""
if query_clip:
    translated = get_translation(query_clip)
    st.sidebar.markdown(f"🌐 **{translated}**")

query_name = st.sidebar.text_input("File name:", "")
num_columns = st.sidebar.slider("Scale", 1, 8, 3)
max_results = st.sidebar.slider("Number of results", 5, 1000, 30, step=5)
show_all = st.sidebar.checkbox("Show all")

# --- DINOv2-B image search ---
st.sidebar.subheader("Image Search")
dino_ref = st.sidebar.file_uploader("Upload an image to find similar ones", type=['jpg', 'jpeg', 'png', 'bmp', 'webp'])

# Превью референса
if dino_ref is not None:
    ref_img = Image.open(dino_ref)
    st.sidebar.image(ref_img, caption="Reference", use_container_width=True)

# --- Главный вывод: мультимодальный поиск ---
if dino_ref is not None and IMAGE_FOLDER and os.path.exists(IMAGE_FOLDER):
    dino_files, dino_embs = index_images_with_dino_cache(IMAGE_FOLDER)
    dino_img = Image.open(dino_ref).convert("RGB")
    tensor = dino_transform(dino_img).unsqueeze(0).to(device)  # Перемещаем на GPU
    with torch.no_grad():
        query_emb = dinov2(tensor)
        query_emb = query_emb / query_emb.norm(dim=-1, keepdim=True)
    dino_similarities = torch.nn.functional.cosine_similarity(query_emb.cpu(), dino_embs)

    # Топ по DINO: 5*max_results или всё если show_all
    N = len(dino_similarities) if show_all else min(max_results * 5, len(dino_similarities))
    topk = dino_similarities.topk(N)
    filtered_files = [dino_files[idx] for idx in topk.indices]

    filtered_embs = []
    filtered_paths = []
    dino_scores = []
    for i, f in enumerate(filtered_files):
        if f in path_to_index:  # O(1) поиск вместо O(n)
            idx = path_to_index[f]  # O(1) получение индекса
            filtered_embs.append(image_features[idx])
            filtered_paths.append(f)
            dino_scores.append(dino_similarities[topk.indices[i]].item())
    if len(filtered_embs) > 0:
        filtered_embs = torch.stack(filtered_embs).to(device)
    else:
        filtered_embs = torch.empty((0, model.visual.output_dim)).to(device)

    # Если есть текстовый запрос
    if query_clip and len(filtered_embs) > 0:
        # Используем уже переведенный текст
        translated_en = translated if translated else query_clip
        if not translated_en.replace(" ", "").isascii():
            # Если всё еще не английский, переводим
            translator = Translator()
            translated_en = translator.translate(query_clip, src='auto', dest='en').text
        with torch.no_grad():
            text_tokens = open_clip.tokenize([translated_en]).to(device)
            text_features = model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.float()
            text_similarities = (filtered_embs @ text_features.T).squeeze()

        text_scores = text_similarities.cpu().numpy()
        dino_scores = np.array(dino_scores)
        sort_idx = np.lexsort((-dino_scores, -text_scores))
        K = len(sort_idx) if show_all else min(max_results, len(sort_idx))
        final_indices = sort_idx[:K]
        final_files = [filtered_paths[i] for i in final_indices]
    else:
        # Без текста — просто по DINO
        K = len(filtered_paths) if show_all else min(max_results, len(filtered_paths))
        final_files = filtered_paths[:K]

    cols = st.columns(num_columns)
    for idx, img_path in enumerate(final_files):
        img_name = os.path.basename(img_path)
        with cols[idx % num_columns]:
            st.image(img_path, use_container_width=True, caption=img_name)

elif query_clip:
    t_start = time.time()
    # Используем уже переведенный текст
    translated_en = translated if translated else query_clip
    if not translated_en.replace(" ", "").isascii():
        # Если всё еще не английский, переводим
        translator = Translator()
        translated_en = translator.translate(query_clip, src='auto', dest='en').text

    with torch.no_grad():
        text_tokens = open_clip.tokenize([translated_en]).to(device)
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.float()
        similarities_clip = (image_features @ text_features.T).squeeze()

    filename_scores = []
    for path in image_paths:
        name = os.path.basename(path).lower()
        score = 1.0 if query_name and query_name.lower() in name else 0.0
        filename_scores.append(score)
    filename_scores = torch.tensor(filename_scores, device=device, dtype=torch.float32)
    similarities = similarities_clip + filename_scores

    if show_all:
        top_k = len(similarities)
    else:
        top_k = min(max_results, len(similarities))
    top_indices = similarities.topk(top_k).indices

    t_end = time.time()
    elapsed = t_end - t_start
    st.session_state['last_search_time'] = elapsed

    cols = st.columns(num_columns)
    selected_images = []
    for idx, i in enumerate(top_indices):
        img_path = image_paths[i]
        img_name = os.path.basename(img_path)
        with cols[idx % num_columns]:
            st.image(img_path, use_container_width=True, caption=img_name)
            selected_images.append(img_path)

elif query_name:
    t_start = time.time()
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

else:
    t_start = time.time()
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
    t_end = time.time()
    elapsed = t_end - t_start
    st.session_state['last_search_time'] = elapsed