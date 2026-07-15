"""Human-approved post-RC4 release-candidate backend."""
from .v09 import GyuSingerV09Renderer


class GyuSingerRC5Renderer(GyuSingerV09Renderer):
    def model_info(self) -> dict:
        return super().model_info() | {
            "backend": "gyu-singer-rc5",
            "model_version": "1.0.0-rc.5",
            "release_state": "release_candidate",
            "final_v1_tagged": False,
        }
