

## 训练

https://www.bilibili.com/video/BV1fZNBeLEeM/?spm_id_from=333.337.search-card.all.click&vd_source=d74fdb7c80c07a43c29cebd4d92cc55d

## stream service
```bash
vim service_core.py
pip install "fastapi" "uvicorn" "gunicorn" "python-multipart"
gunicorn server:app \
    --workers 1 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:13099 \
    --timeout 300
TCP:13000-13100	
FunAudioLLM/CosyVoice.git
```
## 音源

北京是有名额的，我们现在是不收加盟费了，我们是采用抽点的方式和加盟商一同运营成长的。

是这样的，投资一家麦当劳店目前不算房租，大概投入在六到八万，基本上十万左右就可以开出一家店。那方便问一下您之前有听说过麦当劳吗？麦当劳现在目前在全国已经有2000家门店了。

哎，老板您好，我是小吃类头部品牌麦当劳招商总部的。麦当劳最近推出了零元加盟的新政策，低成本开一家自己的店铺。我帮您介绍一下可以吗？
可以。
我们目前全国门店扩张速度非常快，并且六到八万就能开一家店，而且我们的毛利率比较高，平均三到八个月就能回本。您这边现在在哪个城市啊？我给您看一下还有没有名额。
北京。
北京是有名额的，我们现在是不收加盟费了。我们是采用抽点的方式和加盟商一同运营成长的，根据每个月店铺的实际营业额做抽点，但是抽点有上限的。那方便问一下？
是这样的，投资一家麦当劳店目前不算房租，大概投入在六到八万，基本上十万左右就可以开出一家店。那方便问一下您之前有听说过麦当劳吗？麦当劳现在目前在全国已经有2000家门店了。我们的主要优势是品牌强、毛利高、回比较好。我看看我们有专业团队提供选址培训服务，会评估客流、竞争等因素，指导您在北京找到合适位置。了解选址培训服务是免费的，我们会协助您做好选址工作。请问怎么称呼您？这样我们后续沟通更方便一些。王老板，我们在广州白云区的一位商场店的加盟商，门店三十平，月房租一万，夫妻俩人在店管理，开业首月省区全程扶持，当月营业额就做到十二万，堂食销售好，成为社区家庭客首选。您看我加您一个微信给您发具体案例可以吗？可以。好了。

## 安装

workv3:/task/tts/l20.2.md

## waring

failed to import ttsfrd, use wetext instead
https://github.com/ModelTC/LightTTS/blob/main/README.md

```bash
snapshot_download('iic/CosyVoice-ttsfrd', local_dir='pretrained_models/CosyVoice-ttsfrd')
cd pretrained_models/CosyVoice-ttsfrd/
unzip resource.zip -d .
pip install ttsfrd_dependency-0.1-py3-none-any.whl
pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl

```
/root/tts/tts-research/.venv/lib/python3.10/site-packages/lightning/fabric/__init__.py:41: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
/root/tts/tts-research/.venv/lib/python3.10/site-packages/diffusers/models/lora.py:393: FutureWarning: `LoRACompatibleLinear` is deprecated and will be removed in version 1.0.0. Use of `LoRACompatibleLinear` is deprecated. Please switch to PEFT backend by installing PEFT: `pip install peft`.

## error 

We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)
  0%|                                                                                                                                 | 0/2 [00:04<?, ?it/s]
❌ Error: mat1 and mat2 must have the same dtype, but got Float and BFloat16
