from flask import Flask, request, render_template, jsonify
import os
from PIL import Image
import numpy as np
import torch
import faiss
import pickle
from transformers import CLIPModel, CLIPProcessor
import base64
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app = Flask(__name__)

# CLIP model setup
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model.to(device)

# Paths
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
DATA_FOLDER = 'data'
os.makedirs(DATA_FOLDER, exist_ok=True)

# Define categories and their dataset paths
CATEGORIES = {
    "rings": r"C:\Users\Aavej\QHills\Datasets\Ring iamges\Datasets\All_Rings",
    "bangles": r"C:\Users\Aavej\QHills\Datasets\Bangles_new",
    "noserings": r"C:\Users\Aavej\QHills\Datasets\nose",
    "earrings": r"C:\Users\Aavej\QHills\Datasets\Unique_Earrings",
    "mangalsutras": r"C:\Users\Aavej\QHills\Datasets\Mangalsutra"
}

indexes = {}
features_dict = {}
image_paths_dict = {}

def load_images(folder_path):
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                   if f.endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    logger.info(f"Loaded {len(image_paths)} images from {folder_path}")
    if image_paths:
        logger.debug(f"First image path: {image_paths[0]}")
    return image_paths

def extract_clip_features(image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
    return features.squeeze().cpu().numpy()

def build_faiss_index(image_paths, category):
    logger.info(f"Building FAISS index for {category}...")
    features = np.array([extract_clip_features(Image.open(img).convert("RGB")) 
                        for img in image_paths], dtype=np.float32)
    
    with open(os.path.join(DATA_FOLDER, f"{category}_features.pkl"), "wb") as f:
        pickle.dump(features, f)
    with open(os.path.join(DATA_FOLDER, f"{category}_image_paths.pkl"), "wb") as f:
        pickle.dump(image_paths, f)
    
    dimension = features.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(features)
    faiss.write_index(index, os.path.join(DATA_FOLDER, f"{category}_faiss_index.bin"))
    logger.info(f"FAISS index for {category} built and saved")
    return index, features

def load_faiss_index(category):
    index_path = os.path.join(DATA_FOLDER, f"{category}_faiss_index.bin")
    features_path = os.path.join(DATA_FOLDER, f"{category}_features.pkl")
    image_paths_path = os.path.join(DATA_FOLDER, f"{category}_image_paths.pkl")
    
    if all(os.path.exists(p) for p in [index_path, features_path, image_paths_path]):
        logger.info(f"Loading FAISS index for {category}...")
        index = faiss.read_index(index_path)
        with open(features_path, "rb") as f:
            features = pickle.load(f)
        with open(image_paths_path, "rb") as f:
            image_paths = pickle.load(f)
        if image_paths and os.path.exists(image_paths[0]):
            logger.info(f"Loaded {len(image_paths)} image paths for {category}")
            return index, features, image_paths
        else:
            logger.warning(f"Invalid image paths for {category}, rebuilding...")
            return None, None, None
    logger.info(f"No index found for {category}, will build new one")
    return None, None, None

# Load or build indexes for all categories at startup
for category, folder in CATEGORIES.items():
    if not os.path.exists(folder):
        logger.error(f"Dataset folder {folder} for {category} does not exist!")
        continue
    index, features, image_paths = load_faiss_index(category)
    if index is None:
        image_paths = load_images(folder)
        if not image_paths:
            logger.error(f"No images found in {folder} for {category}!")
            continue
        index, features = build_faiss_index(image_paths, category)
    indexes[category] = index
    features_dict[category] = features
    image_paths_dict[category] = image_paths

def get_similar_images(query_image, category, top_k=5):
    if category not in indexes or indexes[category] is None:
        logger.error(f"No valid index for category {category}")
        return []
    query_embedding = extract_clip_features(query_image).astype(np.float32).reshape(1, -1)
    distances, indices = indexes[category].search(query_embedding, top_k + 1)
    recommended_images = [image_paths_dict[category][i] for i in indices[0][1:] 
                          if i < len(image_paths_dict[category])]
    logger.info(f"Found {len(recommended_images)} similar images for {category}")
    return recommended_images[:top_k]

def image_to_base64(img_path):
    try:
        with open(img_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except FileNotFoundError as e:
        logger.error(f"Image not found: {img_path}")
        raise e

@app.route('/')
def index_page():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    category = request.form.get('category')
    if not category or category not in CATEGORIES:
        logger.error(f"Invalid or missing category: {category}")
        return jsonify({'error': 'Invalid or missing category'}), 400
    if 'image' not in request.files:
        logger.error("No image uploaded")
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        logger.error("No image selected")
        return jsonify({'error': 'No image selected'}), 400
    
    img_path = os.path.join(UPLOAD_FOLDER, 'query_image.jpg')
    file.save(img_path)
    query_image = Image.open(img_path).convert("RGB")
    
    try:
        recommended_images = get_similar_images(query_image, category)
        if not recommended_images:
            return jsonify({'error': f'No recommendations found for {category}'}), 404
        query_base64 = image_to_base64(img_path)
        recommended_base64 = [image_to_base64(img) for img in recommended_images]
        
        return jsonify({
            'query_image': query_base64,
            'recommended_images': recommended_base64
        })
    except FileNotFoundError as e:
        logger.error(f"Error in image processing: {str(e)}")
        return jsonify({'error': f'Image file not found: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)