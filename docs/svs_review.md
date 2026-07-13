# SVS review and decision

| System | Useful idea | GYU v1 decision |
|---|---|---|
| TCSinger 2 | multilingual content and blurred phoneme/note boundaries | future neural backbone; needs verified lyrics/alignments |
| FM-Singer | latent conditional flow matching | future acoustic generator, not usable with current label uncertainty |
| TechSinger | technique conditioning | reserve expression curves in protocol |
| SoulX-Singer | zero-shot singer conditioning | future reference/timbre encoder candidate |
| YingMusic-Singer Plus | score-aware expressive generation | future pseudo-singing teacher candidate |
| OpenVPI DiffSinger | established score/phoneme training and renderer conventions | compatibility reference; supplied environment scaffold retained |

TCSinger 2 paper: https://arxiv.org/abs/2505.14910. Current v1 rejects code reuse: a randomly initialized flow/diffusion SVS trained on 29 minutes with unverified text would be less honest and less useful than an explicitly limited real-anchor renderer.
