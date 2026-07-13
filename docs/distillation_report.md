# Distillation report

633 retained teacher rows are representation-only. Their trust weights are applied by `teacher_distillation_loss`; they do not create acoustic decoder targets. Actual 160-step run sampled one weighted teacher representation row per real-anchor step. `test_losses_use_pitch_mask_and_teacher_trust` verifies zero-trust rows contribute zero loss.
