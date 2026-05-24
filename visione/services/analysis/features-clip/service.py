import argparse
import functools
import logging
import os
from pathlib import Path

from flask import Flask, request, jsonify
import numpy as np
import requests
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModel

from visione.vram_aware_loader import load_hf_model_with_offload, get_offload_dir, probe_vram


# setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')

# Optimize PyTorch for efficiency
torch.set_num_threads(4)
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

# create the Flask app
app = Flask(__name__)


class CLIPTextEncoder():
    def __init__(self, model_handle, offload_dir=None):
        offload_dir = offload_dir or get_offload_dir()

        # Estimate size by model variant name
        if 'large' in model_handle or 'H-14' in model_handle or 'vit-l' in model_handle.lower():
            model_size_gb = 3.5
        elif 'huge' in model_handle or 'G-14' in model_handle:
            model_size_gb = 7.0
        else:
            model_size_gb = 1.5

        strategy = probe_vram(model_size_gb)
        cache = "/cache/huggingface"
        self.model = load_hf_model_with_offload(
            from_pretrained_fn=lambda **kw: AutoModel.from_pretrained(model_handle, cache_dir=cache, **kw),
            model_size_gb=model_size_gb,
            offload_dir=offload_dir,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_handle, cache_dir=cache)

        # Determine active device for output tensors
        self._cpu_out = strategy in ('cpu', 'disk')

    def get_text_embedding(self, text, normalized=False):
        with torch.no_grad():
            inputs = self.tokenizer(text, padding=True, return_tensors="pt")
            # Move inputs to the model's first parameter device
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            text_features = self.model.get_text_features(**inputs)
            if normalized:
                text_features = F.normalize(text_features, dim=-1)
            text_features = text_features.cpu().numpy().squeeze()
        return text_features

@app.route('/ping', methods=['GET'])
def ping():
    return "pong"

@app.route('/get-text-feature', methods=['GET'])
def get_text_features():
    text = request.args.get("text")
    logging.info('Received text: {}'.format(text))
    text_feature = qe.get_text_embedding(text, normalized=args.normalized)
    out = jsonify(text_feature.tolist())
    return out

@app.route('/get-image-feature', methods=['GET'])
def extract_image_feature_by_url():
    # url = request.args.get("url")
    return "Not Implemented", 501

@app.route('/get-image-feature', methods=['POST'])
def extract_image_feature_by_file():
    # file = request.files['file']
    return "Not Implemented", 501

# deprecated, kept for backward compatibility of 'core' service
@app.route('/text-to-image-search', methods=['GET'])
def text_to_image_search():
    text = request.args.get("text")
    k = request.args.get("k", type=int, default=10000)
    logging.info('Received text: {}'.format(text))
    text_feature = qe.get_text_embedding(text, normalized=args.normalized)

    response = requests.post('http://faiss-index-manager:8080/search', json={
        'type': features_name,
        'feature_vector': text_feature.tolist(),
        'k': k,
    }).content

    return response

# deprecated, kept for backward compatibility of 'core' service
@app.route('/internal-image-search', methods=['GET'])
def internal_image_search():
    img_id = request.args.get("imgId")
    k = request.args.get("k", type=int, default=10000)

    response = requests.post('http://faiss-index-manager:8080/search', json={
        'type': features_name,
        'query_id': img_id,
        'k': k,
    }).content

    return response


if __name__ == '__main__':
    default_model_handle = os.environ['MODEL_HANDLE']
    features_name = os.environ['FEATURES_NAME']

    parser = argparse.ArgumentParser(description='Service for query feature extraction for CLIP models.')

    parser.add_argument('--host', default='0.0.0.0', help="IP address to use for binding")
    parser.add_argument('--port', default='8080', help="Port to use for binding")
    parser.add_argument('--model-handle', default=default_model_handle, help='hugging face handle of the CLIP model')
    parser.add_argument('--no-normalized', action='store_false', dest='normalized', default=True, help='Whether to normalize features or not')
    parser.add_argument('--offload-dir', default=None, help='directory for disk-based model offloading')
    parser.add_argument('--vram-threshold', type=float, default=None, help='VRAM fraction threshold before offloading')

    args = parser.parse_args()

    # Propagate CLI overrides into env vars the loader reads
    if args.vram_threshold is not None:
        os.environ['VISIONE_VRAM_THRESHOLD'] = str(args.vram_threshold)

    # init the query encoder
    qe = CLIPTextEncoder(args.model_handle, offload_dir=args.offload_dir)

    # run the flask app
    app.run(debug=False, host=args.host, port=args.port)