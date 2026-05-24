import argparse
import itertools
import logging
import os

import more_itertools
from PIL import Image
import torch
import open_clip

from visione.extractor import BaseExtractor
from visione.vram_aware_loader import (
    load_model_with_offload,
    suggest_batch_size,
    probe_vram,
    get_offload_dir,
)


loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in loggers:
    logger.setLevel(logging.WARNING)


class ImageListDataset(torch.utils.data.Dataset):
    def __init__(self, paths, processor):
        self.paths = paths
        self.processor = processor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = Image.open(path)
        image_pt = self.processor(image)
        return image_pt


class WrapIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, iterable, batch_size, processor, preload=False):
        self.iterable = iterable
        self.batch_size = batch_size
        self.processor = processor

        if preload:
            self.iterable = list(self.iterable)

    def process(self, item):
        images = [Image.open(i) for i in item]
        images_pt = self.processor(images=images, return_tensors="pt")
        return images_pt

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        itr = self.iterable
        if worker_info is not None:
            itr = itertools.islice(self.iterable, worker_info.id, None, worker_info.num_workers)

        itr = more_itertools.chunked(itr, self.batch_size)
        itr = map(self.process, itr)
        yield from itr


class OpenCLIPExtractor(BaseExtractor):

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument('--model-handle', default=os.environ['MODEL_HANDLE'], help='model handle')
        parser.add_argument('--batch-size', default=1, type=int, help='batch size')
        parser.add_argument('--num-workers', default=4, type=int, help='number of workers')
        super(OpenCLIPExtractor, cls).add_arguments(parser)

    def __init__(self, args):
        super(OpenCLIPExtractor, self).__init__(args)
        self.device = None
        self.model = None
        self.processor = None

    def setup(self):
        if self.model is None:
            # Apply CLI overrides
            if self.args.vram_threshold is not None:
                os.environ['VISIONE_VRAM_THRESHOLD'] = str(self.args.vram_threshold)
            offload_dir = self.args.offload_dir or get_offload_dir()

            if not self.args.gpu:
                os.environ.setdefault('VISIONE_OFFLOAD_MODE', 'cpu')

            handle = self.args.model_handle
            os.makedirs('/cache/open_clip', exist_ok=True)

            # open_clip is not HF — use generic loader
            def _load():
                model, _, processor = open_clip.create_model_and_transforms(
                    handle, cache_dir='/cache/open_clip'
                )
                return model

            # Estimate size from model name
            if 'ViT-H' in handle or 'ViT-L' in handle:
                size_gb = 3.5
            elif 'ViT-G' in handle or 'ViT-bigG' in handle:
                size_gb = 7.0
            else:
                size_gb = 1.5

            strategy = probe_vram(size_gb)
            self._strategy = strategy

            self.model = load_model_with_offload(_load, model_size_gb=size_gb, offload_dir=offload_dir)
            self.model.eval()

            # We need the preprocessor separately (no side effects in _load closure)
            _, _, self.processor = open_clip.create_model_and_transforms(
                handle, cache_dir='/cache/open_clip', pretrained=False
            )

            # Determine active device
            self.device = next(self.model.parameters()).device

    def extract(self, image_paths):
        records = list(self.extract_iterable(image_paths))
        return records

    def extract_iterable(self, image_paths):
        self.setup()  # lazy load model

        # if we must preload, we might as well use a standard dataset
        image_paths = list(image_paths)
        dataset = ImageListDataset(image_paths, self.processor)

        effective_batch = suggest_batch_size(self.args.batch_size, getattr(self, '_strategy', 'cuda'))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=effective_batch, num_workers=self.args.num_workers)

        with torch.no_grad():
            for images_pt in dataloader:
                images_pt = images_pt.to(self.device)
                images_features = self.model.encode_image(images_pt).float()
                records = [{'feature_vector': f.tolist()} for f in images_features.cpu().numpy()]
                yield from records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract features from an Open CLIP model')
    OpenCLIPExtractor.add_arguments(parser)
    args = parser.parse_args()
    extractor = OpenCLIPExtractor(args)
    extractor.run()