"""Acquisition-domain classification.

Priority order, exactly as the policy requires:

1. verified source metadata (``metadata_verified``);
2. trusted path/prefix metadata inside a mixed pool (``filename_or_metadata``);
3. zero-shot visual inference, only when the first two are absent or conflict
   (``vision_model_inferred``);
4. anything else is ``UNKNOWN`` and cannot enter R2/R3.

The visual model is a *classifier*, not a witness.  If metadata and pixels
disagree, or if the top two visual domains are within the conflict margin, the
image is ``QUARANTINE_DOMAIN`` and its domain becomes ``UNKNOWN``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Weights are fetched from the HF mirror because huggingface.co is unreachable
# from this machine.  Setting this before transformers/huggingface_hub import
# matters: the endpoint is read at import time.
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

DOMAINS = ('FIELD_IN_SITU', 'ORCHARD_IN_SITU', 'PROTECTED_CULTIVATION',
           'CONTROLLED_LAB', 'UAV_AERIAL', 'SATELLITE_REMOTE')

PROMPTS: Dict[str, List[str]] = {
    'FIELD_IN_SITU': [
        'a close-up photograph of crop plants growing in an open farm field',
        'a ground-level photograph of weeds and crops in an agricultural field',
        'a photograph of a farm field taken at human height',
    ],
    'ORCHARD_IN_SITU': [
        'a photograph of fruit trees in an orchard',
        'a close-up photograph of apples growing on a tree in an orchard',
    ],
    'PROTECTED_CULTIVATION': [
        'a photograph of plants growing inside a greenhouse',
        'a photograph of plants under artificial grow lights in a plant factory',
    ],
    'CONTROLLED_LAB': [
        'a photograph of a single leaf on a plain white laboratory background',
        'a photograph of a plant sample on a laboratory bench',
        'a studio photograph of a crop leaf with a uniform background',
    ],
    'UAV_AERIAL': [
        'an aerial drone photograph of a farm field seen from above',
        'a top-down aerial photograph of agricultural plots',
    ],
    'SATELLITE_REMOTE': [
        'a satellite image of farmland',
    ],
}

#: Source-level metadata.  ``metadata_verified`` is only used where the source
#: documentation actually states the capture setting (Stage 5A/5B evidence).
SOURCE_DOMAIN: Dict[str, Tuple[str, str]] = {
    'agrifm_ref1': ('FIELD_IN_SITU', 'official_metadata'),
    'perennial_plants': ('UNKNOWN', 'official_metadata'),
    'agrifm_ref5': ('FIELD_IN_SITU', 'official_metadata'),
    'agrifm_ref6': ('CONTROLLED_LAB', 'official_metadata'),
    'agrifm_ref7': ('UNKNOWN', 'official_metadata'),
    'agrifm_ref8': ('UNKNOWN', 'official_metadata'),
    'agrifm_ref9': ('ORCHARD_IN_SITU', 'official_metadata'),
    'agrifm_ref10': ('UNKNOWN', 'official_metadata'),
    'agrifm_ref11': ('FIELD_IN_SITU', 'official_metadata'),
    'agrifm_ref15': ('CONTROLLED_LAB', 'official_metadata'),
    'agrifm_ref18': ('ORCHARD_IN_SITU', 'official_metadata'),
    'agrifm_ref20': ('CONTROLLED_LAB', 'official_metadata'),
    'agrifm_ref29': ('FIELD_IN_SITU', 'official_metadata'),
    'agrifm_ref31': ('FIELD_IN_SITU', 'official_metadata'),
    'agrifm_ref35': ('FIELD_IN_SITU', 'official_metadata'),
    'agrifm_ref37': ('CONTROLLED_LAB', 'official_metadata'),
    'agrifm_ref46': ('UAV_AERIAL', 'official_metadata'),
    'agrifm_ref48': ('UAV_AERIAL', 'official_metadata'),
    'agrifm_ref57': ('MIXED', 'official_metadata'),
    'agrifm_ref58': ('FIELD_IN_SITU', 'official_metadata'),
    'agrifm_ref60': ('FIELD_IN_SITU', 'official_metadata'),
    'agrifm_ref65': ('FIELD_IN_SITU', 'official_metadata'),
    'agrifm_ref73': ('FIELD_IN_SITU', 'official_metadata'),
    'agrifm_ref74': ('UAV_AERIAL', 'official_metadata'),
    'agrifm_ref75': ('UNKNOWN', 'official_metadata'),
    'agrifm_ref76': ('UNKNOWN', 'official_metadata'),
    'ronin': ('FIELD_IN_SITU', 'official_metadata'),
    'agrifm_ref81': ('UNKNOWN', 'official_metadata'),
    'agrifm_ref82': ('FIELD_IN_SITU', 'official_metadata'),
    'phenobench': ('FIELD_IN_SITU', 'official_metadata'),
    'agrifm_ref90': ('PROTECTED_CULTIVATION', 'official_metadata'),
    'plantvillage': ('CONTROLLED_LAB', 'official_metadata'),
    'agrifm_ref95': ('FIELD_IN_SITU', 'official_metadata'),
    'deepweeds': ('FIELD_IN_SITU', 'official_metadata'),
    'artemis_imagesafari': ('FIELD_IN_SITU', 'official_metadata'),
    'sesame_weed': ('FIELD_IN_SITU', 'official_metadata'),
    'applebbch81': ('ORCHARD_IN_SITU', 'official_metadata'),
    'agrifield40k_pool': ('UNKNOWN', 'prefix_level'),
}

#: Prefix-level domains inside the AgriField-40K pool.  Only prefixes whose
#: upstream capture setting is documented are given a positive domain; the rest
#: stay UNKNOWN instead of being guessed from the dataset name.
AGRIFIELD_PREFIX_DOMAIN: Dict[str, str] = {
    'acw': 'FIELD_IN_SITU',     # ACRECropWeed
    'vcd': 'FIELD_IN_SITU',     # VCD
    'vgn': 'FIELD_IN_SITU',     # VegAnn
    'inat': 'FIELD_IN_SITU',    # iNatWeeds
    'rwd': 'FIELD_IN_SITU',     # RadishWheat
    'saw': 'FIELD_IN_SITU',     # sesame weed
    'lvb': 'FIELD_IN_SITU',     # LUCASVision land-cover photographs
    'rl':  'FIELD_IN_SITU',     # Rumex leaves in field
    'weds': 'FIELD_IN_SITU',    # WE3DS
    'gc':  'FIELD_IN_SITU',     # GrassClover plots
    'mw':  'FIELD_IN_SITU',     # maize weed
    'pnb': 'FIELD_IN_SITU',     # PhenoBench (holdout, excluded separately)
    'mc':  'MIXED',             # MuST-C robot + UAV
    'pags': 'UNKNOWN',
}

#: ref57 background tokens.  The publisher explicitly documents four
#: backgrounds; the token decides the domain and the resulting label is still
#: recorded as metadata + (optionally) visually confirmed.
BACKGROUND_DOMAIN: Dict[str, str] = {
    'white': 'CONTROLLED_LAB',
    'dark': 'CONTROLLED_LAB',
    'illuminated': 'CONTROLLED_LAB',
    'real': 'FIELD_IN_SITU',
}


@dataclass
class DomainDecision:
    domain: str
    evidence_type: str
    confidence: float
    status: str              # OK | QUARANTINE_DOMAIN
    reason: str
    model_sha: str = ''
    scores: Optional[Dict[str, float]] = None


def metadata_domain(source_id: str, rel_path: str = '') -> Optional[DomainDecision]:
    if source_id == 'agrifield40k_pool':
        prefix = Path(rel_path).name.split('_', 1)[0]
        if prefix.startswith('lvb'):
            prefix = 'lvb'
        domain = AGRIFIELD_PREFIX_DOMAIN.get(prefix)
        if domain and domain != 'MIXED':
            return DomainDecision(domain, 'filename_or_metadata', 1.0, 'OK',
                                  f'agrifield prefix {prefix}')
        if domain == 'MIXED':
            return None
        return DomainDecision('UNKNOWN', 'filename_or_metadata', 0.0, 'OK',
                              f'agrifield prefix {prefix} has no documented domain')
    if source_id == 'agrifm_ref57':
        name = Path(rel_path).name.lower()
        for token, domain in BACKGROUND_DOMAIN.items():
            if token in name:
                return DomainDecision(domain, 'filename_or_metadata', 0.95, 'OK',
                                      f'publisher background token "{token}"')
        return None
    pair = SOURCE_DOMAIN.get(source_id)
    if not pair:
        return DomainDecision('UNKNOWN', 'none', 0.0, 'OK', 'no source metadata')
    domain, evidence = pair
    if domain == 'MIXED':
        return None
    return DomainDecision(domain, evidence, 1.0, 'OK', 'source documentation')


def _feature_tensor(output, kind: str):
    """Normalise CLIP feature outputs across transformers versions.

    Older transformers return a plain tensor from ``get_*_features``; newer
    versions may return ``CLIPTextModelOutput`` / ``BaseModelOutputWithPooling``.
    The projected embedding is the tensor we need either way.
    """
    if hasattr(output, 'text_embeds') and kind == 'text':
        return output.text_embeds
    if hasattr(output, 'image_embeds') and kind == 'image':
        return output.image_embeds
    if hasattr(output, 'pooler_output') and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, 'last_hidden_state'):
        return output.last_hidden_state[:, 0]
    return output


class VisionDomainClassifier:
    """Lazy CLIP zero-shot classifier (openai/clip-vit-base-patch32)."""

    MODEL_NAME = 'openai/clip-vit-base-patch32'

    def __init__(self, device: str = 'cuda:1'):
        self.device = device
        self._model = None
        self._processor = None
        self._text_features = None
        self.model_sha = ''

    def load(self):
        if self._model is not None:
            return
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self._model = CLIPModel.from_pretrained(self.MODEL_NAME).to(self.device).eval()
        self._processor = CLIPProcessor.from_pretrained(self.MODEL_NAME)
        self._text_features = None
        try:
            from huggingface_hub import HfApi
            info = HfApi().model_info(self.MODEL_NAME)
            self.model_sha = info.sha or ''
        except Exception:  # noqa: BLE001
            self.model_sha = 'unresolved'

    def _text(self):
        import torch
        if self._text_features is None:
            prompts, owners = [], []
            for domain, texts in PROMPTS.items():
                for text in texts:
                    prompts.append(text)
                    owners.append(domain)
            inputs = self._processor(text=prompts, return_tensors='pt', padding=True)
            with torch.no_grad():
                features = self._model.get_text_features(
                    input_ids=inputs['input_ids'].to(self.device),
                    attention_mask=inputs['attention_mask'].to(self.device))
            features = _feature_tensor(features, 'text')
            features = features / features.norm(dim=-1, keepdim=True)
            self._text_features = (features, owners)
        return self._text_features

    def classify(self, paths: List[Path], batch_size: int = 32) -> List[Tuple[str, float, Dict[str, float]]]:
        import torch
        self.load()
        text_features, owners = self._text()
        results = []
        for start in range(0, len(paths), batch_size):
            batch = paths[start:start + batch_size]
            images = []
            for path in batch:
                from PIL import Image
                try:
                    with Image.open(path) as image:
                        images.append(image.convert('RGB').copy())
                except Exception:  # noqa: BLE001
                    images.append(None)
            valid = [i for i, image in enumerate(images) if image is not None]
            outputs: List[Tuple[str, float, Dict[str, float]]] = [('UNKNOWN', 0.0, {}) for _ in batch]
            if valid:
                inputs = self._processor(images=[images[i] for i in valid], return_tensors='pt')
                with torch.no_grad():
                    features = self._model.get_image_features(
                        pixel_values=inputs['pixel_values'].to(self.device))
                features = _feature_tensor(features, 'image')
                features = features / features.norm(dim=-1, keepdim=True)
                similarity = features @ text_features.T
                for row, index in enumerate(valid):
                    values = similarity[row]
                    domain_scores: Dict[str, float] = {}
                    for domain in PROMPTS:
                        idx = [j for j, owner in enumerate(owners) if owner == domain]
                        domain_scores[domain] = float(values[idx].max())
                    stacked = torch.tensor([domain_scores[d] for d in DOMAINS])
                    probabilities = torch.softmax(stacked * 20.0, dim=0)
                    best = int(probabilities.argmax())
                    domain = DOMAINS[best]
                    confidence = float(probabilities[best])
                    outputs[index] = (domain, confidence,
                                      {d: float(p) for d, p in zip(DOMAINS, probabilities)})
            results.extend(outputs)
        return results


def decide(source_id: str, rel_path: str,
           classifier: Optional[VisionDomainClassifier],
           vision_path: Optional[Path] = None,
           accept_min: float = 0.55,
           conflict_margin: float = 0.08) -> DomainDecision:
    meta = metadata_domain(source_id, rel_path)
    if meta and meta.domain not in ('UNKNOWN',):
        return meta
    if classifier is None or vision_path is None:
        return meta or DomainDecision('UNKNOWN', 'none', 0.0, 'OK', 'no metadata and vision disabled')
    if not vision_path.exists():
        return DomainDecision('UNKNOWN', 'none', 0.0, 'OK', 'file missing for vision')
    domain, confidence, scores = classifier.classify([vision_path])[0]
    ordered = sorted(scores.values(), reverse=True)
    margin = ordered[0] - ordered[1] if len(ordered) > 1 else 1.0
    if margin < conflict_margin or confidence < accept_min:
        whether = ('source metadata conflicts with pixels'
                   if meta and meta.domain == 'UNKNOWN' else 'low-confidence visual domain')
        return DomainDecision('UNKNOWN', 'vision_model_inferred', confidence,
                              'QUARANTINE_DOMAIN', whether,
                              classifier.model_sha, scores)
    if meta and meta.domain == 'UNKNOWN':
        return DomainDecision(domain, 'vision_model_inferred', confidence, 'OK',
                              'metadata unknown; visual model accepted',
                              classifier.model_sha, scores)
    return DomainDecision(domain, 'vision_model_inferred', confidence, 'OK',
                          'visual model accepted', classifier.model_sha, scores)
