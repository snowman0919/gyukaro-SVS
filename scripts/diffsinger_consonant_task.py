"""DiffSinger task that prevents short Japanese consonants being averaged away."""
from __future__ import annotations

import torch

from gyu_singer.diffsinger_weighting import weighted_frame_l1
from training.acoustic_task import AcousticTask, ShallowDiffusionOutput
from utils.hparams import hparams


class ConsonantWeightedAcousticTask(AcousticTask):
    def __init__(self):
        super().__init__()
        self.low_weight_ids = {
            self.phoneme_dictionary.encode_one(phone)
            for phone in ("AP", "SP", "ja_a", "ja_e", "ja_i", "ja_o", "ja_u", "ja_I", "ja_O", "ja_U")
        }

    def run_model(self, sample, infer=False):
        if infer:
            return super().run_model(sample, infer=True)
        txt_tokens = sample["tokens"]
        target = sample["mel"]
        mel2ph = sample["mel2ph"]
        variances = {name: sample[name] for name in self.required_variances}
        output: ShallowDiffusionOutput = self.model(
            txt_tokens, mel2ph=mel2ph, f0=sample["f0"], **variances,
            key_shift=sample.get("key_shift"), speed=sample.get("speed"),
            spk_embed_id=sample.get("spk_ids") if hparams["use_spk_id"] else None,
            languages=sample.get("languages") if hparams["use_lang_id"] else None,
            gt_mel=target, infer=False,
        )
        assert output.aux_out is not None and output.diff_out is None
        phone_positions = (mel2ph.long() - 1).clamp(min=0, max=txt_tokens.shape[1] - 1)
        frame_token_ids = torch.gather(txt_tokens, 1, phone_positions)
        loss = weighted_frame_l1(
            output.aux_out,
            self.model.aux_decoder.norm_spec(target),
            frame_token_ids,
            self.low_weight_ids,
            consonant_weight=float(hparams.get("consonant_loss_weight", 5.0)),
        )
        return {"aux_mel_loss": self.lambda_aux_mel_loss * loss}
