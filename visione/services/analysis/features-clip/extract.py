import argparse
import itertools
import logging
import os

import more_itertools
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

from visione.extractor import BaseExtractor
from visione.vram_aware_loader import (
    load_hf_model_with_offload,
    suggest_batch_size,
    probe_vram,
    get_offload_dir,
)

os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in loggers:
    logger.setLevel(logging.INFO)


class ImageListDataset(torch.utils.data.Dataset):
    def __init__(self, paths, processor):
        self.paths = paths
        self.processor = processor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = Image.open(path)
        image_pt = self.processor(images=[image], return_tensors="pt")
        return image_pt

    @staticmethod
    def collate_fn(batch):
        return {k: torch.concat([item[k] for item in batch]) for k in batch[0].keys()}


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


class CLIPExtractor(BaseExtractor):

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument('--model-handle', default=os.environ['MODEL_HANDLE'], help='hugging face handle of the CLIP model')
        parser.add_argument('--batch-size', default=1, type=int, help='batch size')
        parser.add_argument('--num-workers', default=4, type=int, help='number of workers')
        super(CLIPExtractor, cls).add_arguments(parser)

    def __init__(self, args):
        super(CLIPExtractor, self).__init__(args)
        self.device = None
        self.model = None
        self.processor = None

    def setup(self):
        if self.model is None:
            # Apply CLI overrides to env vars so the loader reads them
            if self.args.vram_threshold is not None:
                os.environ['VISIONE_VRAM_THRESHOLD'] = str(self.args.vram_threshold)
            offload_dir = self.args.offload_dir or get_offload_dir()

            if not self.args.gpu:
                # GPU flag not set — force CPU mode
                os.environ.setdefault('VISIONE_OFFLOAD_MODE', 'cpu')

            strategy = probe_vram(self._model_size_gb())
            self._strategy = strategy

            cache = "/cache/huggingface"
            handle = self.args.model_handle
            self.model = load_hf_model_with_offload(
                from_pretrained_fn=lambda **kw: CLIPModel.from_pretrained(handle, cache_dir=cache, **kw),
                model_size_gb=self._model_size_gb(),
                offload_dir=offload_dir,
            )
            self.processor = CLIPProcessor.from_pretrained(handle, cache_dir=cache)

            # Adjust device reference for tensor moves inside extract_iterable
            if hasattr(self.model, 'device'):
                self.device = self.model.device
            else:
                self.device = next(self.model.parameters()).device

    def _model_size_gb(self) -> float:
        """Rough VRAM estimate based on model variant."""
        handle = getattr(self.args, 'model_handle', '')
        if 'large' in handle or 'H-14' in handle or 'vit-l' in handle.lower():
            return 3.5
        if 'huge' in handle or 'G-14' in handle:
            return 7.0
        return 1.5  # ViT-B default

    def extract(self, image_paths):
        batch_size = len(image_paths)
        records = list(self.extract_iterable(image_paths, batch_size))
        return records

    def extract_iterable(self, image_paths):
        self.setup()  # lazy load model

        # FIXME: iterable dataset has problems with h5py in multiprocessing, it only works with preload=True
        # dataset = WrapIterableDataset(image_paths, batch_size, self.processor, preload=True)
        # dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=24)

        # if we must preload, we might as well use a standard dataset
        image_paths = list(image_paths)
        dataset = ImageListDataset(image_paths, self.processor)

        # Reduce batch size when running on CPU or disk offload
        effective_batch = suggest_batch_size(self.args.batch_size, getattr(self, '_strategy', 'cuda'))

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=effective_batch,
            num_workers=self.args.num_workers,
            collate_fn=ImageListDataset.collate_fn
        )

        with torch.no_grad():
            for images_pt in dataloader:
                images_pt = {k: v.to(self.device) for k, v in images_pt.items()}
                images_features = self.model.get_image_features(**images_pt)
                records = [{'feature_vector': f.tolist()} for f in images_features.cpu().numpy()]
                yield from records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract features from a CLIP model')
    CLIPExtractor.add_arguments(parser)
    args = parser.parse_args()
    extractor = CLIPExtractor(args)
    extractor.run()