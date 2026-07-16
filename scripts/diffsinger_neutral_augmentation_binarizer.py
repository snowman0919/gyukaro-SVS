"""DiffSinger augmentation that expands pitch/rate coverage without control leakage."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIFFSINGER = ROOT / "data/cache/diffsinger"
if str(DIFFSINGER) not in sys.path:
    sys.path.insert(0, str(DIFFSINGER))

from augmentation.spec_stretch import SpectrogramStretchAugmentation
from preprocessing.acoustic_binarizer import AcousticBinarizer


UNVOICED = {
    "ja_ch", "ja_f", "ja_h", "ja_hy", "ja_k", "ja_ky", "ja_p",
    "ja_s", "ja_sh", "ja_t", "ja_ts", "SP", "AP",
    "c_ja", "h_ja", "i̥_ja", "k_ja", "p_ja", "pː_ja", "s_ja", "t_ja",
    "tː_ja", "ts_ja", "tɕ_ja", "ç_ja", "ɕ_ja", "ɨ̥_ja", "ɯ̥_ja", "ɸ_ja", "ʔ_ja",
}


def neutralize_controls(item: dict) -> dict:
    """Label transformed examples as the neutral singer instead of a formant/rate effect."""
    if "key_shift" in item:
        item["key_shift"] = 0.0
    if "speed" in item:
        item["speed"] = 1.0
    return item


def apply_phoneme_voicing(item: dict) -> dict:
    """Keep interpolated F0 only on voiced phones; silence/obstruents receive zero."""
    if item is None:
        return item
    phones = item["ph_text"].split()
    for frame, phone_index in enumerate(item["mel2ph"]):
        if phone_index <= 0 or phones[int(phone_index) - 1] in UNVOICED:
            item["f0"][frame] = 0.0
    return item


class NeutralSpectrogramStretchAugmentation(SpectrogramStretchAugmentation):
    def process_item(self, item: dict, **kwargs) -> dict:
        return apply_phoneme_voicing(neutralize_controls(super().process_item(item, **kwargs)))


class NeutralAugmentationBinarizer(AcousticBinarizer):
    def process_item(self, item_name, meta_data, binarization_args):
        return apply_phoneme_voicing(super().process_item(item_name, meta_data, binarization_args))

    def arrange_data_augmentation(self, data_iterator):
        # AcousticBinarizer imports this class lazily inside the method.
        import augmentation.spec_stretch as module

        original = module.SpectrogramStretchAugmentation
        module.SpectrogramStretchAugmentation = NeutralSpectrogramStretchAugmentation
        try:
            return super().arrange_data_augmentation(data_iterator)
        finally:
            module.SpectrogramStretchAugmentation = original
