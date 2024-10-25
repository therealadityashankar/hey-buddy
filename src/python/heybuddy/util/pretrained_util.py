import os
from typing import Optional
from heybuddy.util.file_util import check_download_file_to_dir

LOCAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pretrained"))
os.makedirs(LOCAL_DIR, exist_ok=True)

__all__ = ["PretrainedModelMixin"]

class PretrainedModelMixin:
    """
    Mixin class for models with pretrained weights.
    Allows for simple but robust downloading and caching of pretrained weights.
    """
    pretrained_model_url: str # URL of the pretrained model file
    pretrained_model_sha256_sum: Optional[str] = None # sha256 sum of the file, optional
    pretrained_model_authorization: Optional[str] = None # Authorization header, optional

    @property
    def pretrained_model_path(self) -> str:
        """
        Get the path of the pretrained model file.
        Download the file if it does not exist.
        """
        if not hasattr(self, "_pretrained_model_path"):
            self._pretrained_model_path = check_download_file_to_dir(
                self.pretrained_model_url,
                LOCAL_DIR,
                use_tqdm=True,
                sha256_sum=self.pretrained_model_sha256_sum,
                authorization=self.pretrained_model_authorization
            )
        return self._pretrained_model_path
