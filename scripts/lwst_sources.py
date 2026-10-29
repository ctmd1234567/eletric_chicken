"""固定版本的作者推理源码；仅下载到被 Git 忽略的本地目录。"""
from pathlib import Path
import urllib.request

from download_models import sha256_of

REVISION = 'af3ed8ac9686a607ec2330275fbb95cf8d1f910d'
SOURCE_ROOT = Path(__file__).resolve().parent.parent / 'samples/lwst_fsd/upstream'
SOURCE_HASHES = {
    'mmdet/models/layers/csp_layer.py': '9838bca1ee7b0791ee39afe6f86f7c1d8020581517af0b7c0d18f11a1d5ade55',
    'mmdet/models/backbones/csp_darknet_tsm.py': '468d666093906cdfad60451db7a1dda10b6aeafd0bbd6d5843d18064fa78521d',
    'mmdet/models/necks/yolox_pafpn_PTA.py': '7c1ee950d5b8abef69284efc652e66e85729d7435b31c0242c07582b9995668d',
    'mmdet/models/dense_heads/yolox_head.py': 'bf7e89f0d7de20b3dc130d16a134b9590d515f78cf2995c9d045568f77c8fd8d',
}


def ensure_sources():
    for name, digest in SOURCE_HASHES.items():
        path = SOURCE_ROOT / name
        if path.exists() and sha256_of(path) == digest:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.py.part')
        url = f'https://raw.githubusercontent.com/WCUSTC/LwST-FSD/{REVISION}/{name}'
        with urllib.request.urlopen(url, timeout=30) as response:
            temporary.write_bytes(response.read())
        if sha256_of(temporary) != digest:
            raise ValueError(f'LwST-FSD 源码校验失败: {name}')
        temporary.replace(path)
    return SOURCE_ROOT
