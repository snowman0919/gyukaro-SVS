from .objectives import flow_matching_loss, pitch_loss, teacher_distillation_loss

log_pitch_loss = pitch_loss
weighted_distillation_loss = teacher_distillation_loss

__all__ = ["flow_matching_loss", "pitch_loss", "teacher_distillation_loss", "log_pitch_loss", "weighted_distillation_loss"]
