from flask import Flask, request, jsonify, render_template, redirect, url_for
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search
from torch.nn.functional import cosine_similarity
import torch, pandas as pd, spacy, os, ast, zipfile

# === CONFIGURATION ===
repo_id = "dsivaram/recipe-api"  # REPLACE THIS with your actual Hugging Face dataset repo ID

# === DOWNLOAD DATA & MODEL FROM HUGGINGFACE ===
print("🔄 Downloading data from Hugging Face...")

recipes_path = hf_hub_download(repo_id=repo_id, filename="RAW_recipes.csv", repo_type="dataset")
interactions_path = hf_hub_download(repo_id=repo_id, filename="RAW_interactions.csv", repo_type="dataset")
embedding_path = hf_hub_download(repo_id=repo_id, filename="recipe_embeddings.pt", repo_type="dataset")
model_zip_path = hf_hub_download(repo_id=repo_id, filename="bert_recipe_model.zip", repo_type="dataset")

# === EXTRACT MODEL ===
model_dir = "bert_recipe_model"
if not os.path.exists(model_dir):
    with zipfile.ZipFile(model_zip_path, 'r') as zip_ref:
        zip_ref.extractall(model_dir)

# === INIT FLASK ===
app = Flask(__name__)
model_dir = "bert_recipe_model/"+model_dir
# === LOAD MODELS AND DATA ===
print("📦 Loading model and data...")
model = SentenceTransformer(model_dir)
recipe_embeddings = torch.load(embedding_path, map_location=torch.device('cpu'))

recipes = pd.read_csv(recipes_path)
interactions = pd.read_csv(interactions_path)

interactions = interactions.groupby('recipe_id')['rating'].mean().reset_index()
recipes = recipes.merge(interactions, how='left', left_on='id', right_on='recipe_id')
recipes['rating'] = recipes['rating'].fillna(recipes['rating'].mean())

# === NLP UTIL ===
nlp = spacy.load("en_core_web_sm")

def extract_keywords(user_input):
    doc = nlp(user_input.lower())
    return list(set(
        token.lemma_ for token in doc
        if token.pos_ in {"NOUN", "ADJ", "PROPN"} and not token.is_stop and token.is_alpha
    ))

def preprocess_query_tags(tag_list):
    return ' '.join([tag.replace('-', ' ').lower() for tag in tag_list])

# === ROUTES ===

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    if request.method == 'GET':
        user_input = request.args.get("input", "")
        cv_tags = request.args.get("tags", "")
        is_api = False
    else:
        if request.is_json:
            user_input = request.json.get("input", "")
            cv_tags = ""
            is_api = True
        else:
            user_input = request.form.get("input", "")
            cv_tags = request.form.get("tags", "")
            is_api = False

    if not user_input and not cv_tags:
        return "<h3>No input or tags provided for recommendation.</h3>", 400

    full_query = f"{cv_tags} {user_input}".strip()
    tags = extract_keywords(full_query)
    query = preprocess_query_tags(tags)
    query_embedding = model.encode(query, convert_to_tensor=True)
    results = semantic_search(query_embedding, recipe_embeddings, top_k=5)[0]

    return_results = []
    for hit in results:
        corpus_id = hit['corpus_id']
        return_results.append({
            "name": recipes.iloc[corpus_id]["name"],
            "id": int(recipes.iloc[corpus_id]["id"]),
            "rating": int(recipes.iloc[corpus_id]["rating"]),
            "score": float(hit["score"])
        })

    if is_api:
        return jsonify(return_results)
    else:
        return render_template("results.html", results=return_results, user_input=user_input, tags=cv_tags)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/nlp')
def nlp_search():
    tags = request.args.get("tags", "")
    return render_template('nlp_search.html', tags=tags)

@app.route('/recipe/<int:recipe_id>')
def recipe_detail(recipe_id):
    recipe_row = recipes[recipes['id'] == recipe_id]
    if recipe_row.empty:
        return "<h3>Recipe not found.</h3>", 404
    recipe = recipe_row.iloc[0].to_dict()

    try:
        steps_raw = recipe.get('steps', '')
        steps_list = ast.literal_eval(steps_raw) if steps_raw else []
        recipe['steps'] = [step.strip().strip('"').strip("'") for step in steps_list]
    except Exception:
        recipe['steps'] = []

    return render_template('recipe_detail.html', recipe=recipe, back_url=request.referrer or '/')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Render provides PORT env variable
    app.run(debug=False, host="0.0.0.0", port=port)
