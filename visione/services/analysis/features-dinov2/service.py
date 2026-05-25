import argparse
import logging
import os
import tempfile
import shutil
import urllib.request

from flask import Flask, request, jsonify
from PIL import Image
import torch
from torch.nn import functional as F
from torchvision import transforms

from visione.vram_aware_loader import load_hub_model_with_offload, get_offload_dir, probe_vram


torch.set_num_threads(4)
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')

# create the Flask app
app = Flask(__name__)


class DinoV2Extractor():
    def __init__(self, model_type='dinov2_vits14', gpu=False, offload_dir=None):
        offload_dir = offload_dir or get_offload_dir()

        if not gpu:
            os.environ.setdefault('VISIONE_OFFLOAD_MODE', 'cpu')

        size_map = {
            'dinov2_vits14': 0.35,
            'dinov2_vitb14': 0.85,
            'dinov2_vitl14': 1.2,
            'dinov2_vitg14': 4.2,
        }
        size_gb = size_map.get(model_type, 1.2)
        probe_vram(size_gb)  # log chosen strategy

        torch.hub.set_dir("/cache/torch")
        self.model = load_hub_model_with_offload(
            hub_load_fn=lambda: torch.hub.load(
                '/usr/src/dinov2',
                model_type,
                source='local',  # use pre-cloned repo, no GitHub network call
                pretrained=True,
            ),
            model_size_gb=size_gb,
            offload_dir=offload_dir,
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def extract_from_pil(self, pil_image, normalized=False):
        with torch.no_grad():
            inputs = self.transform(pil_image).unsqueeze(0).to(self.device)
            image_features = self.model(inputs)
            if normalized:
                image_features = F.normalize(image_features, dim=-1)

        image_features = image_features.cpu().numpy().squeeze()
        return image_features

    def extract_from_path(self, image_path):
        image = Image.open(image_path).convert('RGB')
        features = self.extract_from_pil(image)
        return features

    def extract_from_url(self, image_url):
        with urllib.request.urlopen(image_url) as response:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                shutil.copyfileobj(response, tmp)
                tmp_path = tmp.name
        try:
            features = self.extract_from_path(tmp_path)
        finally:
            os.unlink(tmp_path)
        return features


@app.route('/ping', methods=['GET'])
def ping():
    return "pong"


@app.route('/extract', methods=['GET'])
@app.route('/get-image-feature', methods=['GET'])
def extract_from_url():
    url = request.args.get("url")
    app.logger.info(f'Received URL: {url}')
    feature = extractor.extract_from_url(url).squeeze()
    out = jsonify(feature.tolist())
    return out


@app.route('/extract', methods=['POST'])
@app.route('/get-image-feature', methods=['POST'])
def extract_from_image():
    image = Image.open(request.files['image']).convert('RGB')
    feature = extractor.extract_from_pil(image).squeeze()
    out = jsonify(feature.tolist())
    return out


# deprecated, just for backward compatibility of 'core' service
@app.route('/gem', methods=['GET'])
def extract_quant_from_url():
    import requests
    url = request.args.get("url")
    app.logger.info(f'Received URL: {url}')
    feature_vector = extractor.extract_from_url(url).squeeze()
    app.logger.info(f'Feature Vector: {feature_vector[:5]} ...')
    str_doc = requests.post('http://str-feature-encoder:8080/encode', json={
        'type': 'dinov2',
        'feature_vector': feature_vector.tolist(),
    }).content

    str_doc = str_doc.decode('utf8')
    app.logger.info(str_doc[:25] + " ...")

    return jsonify(str_doc)


if __name__ == '__main__':
    default_model = os.environ['MODEL']
    features_name = os.environ['FEATURES_NAME']
    default_gpu = torch.cuda.is_available() and torch.cuda.device_count() > 0

    parser = argparse.ArgumentParser(description='Service for query feature extraction for CLIP models.')

    parser.add_argument('--host', default='0.0.0.0', help="IP address to use for binding")
    parser.add_argument('--port', default='8080', help="Port to use for binding")
    parser.add_argument('--model', default=default_model, choices=('dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14'), help='Model to use')
    parser.add_argument('--no-normalized', action='store_false', dest='normalized', default=True, help='Whether to normalize features or not')
    parser.add_argument('--gpu', action='store_true', default=default_gpu, help='Whether to use GPU')
    parser.add_argument('--offload-dir', default=None, help='directory for disk-based model offloading')
    parser.add_argument('--vram-threshold', type=float, default=None, help='VRAM fraction threshold before offloading')

    args = parser.parse_args()

    if args.vram_threshold is not None:
        os.environ['VISIONE_VRAM_THRESHOLD'] = str(args.vram_threshold)

    # init the extractor
    extractor = DinoV2Extractor(model_type=args.model, gpu=args.gpu, offload_dir=args.offload_dir)

    # run the flask app
    app.run(debug=False, host=args.host, port=args.port)