#验证 ModelScope token
from modelscope.hub.api import HubApi
import os
api = HubApi()
api.login(os.getenv('MODELSCOPE_TOKEN'))

#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('azhao2050/CosyVoice2-0.5B-finetune-v1')