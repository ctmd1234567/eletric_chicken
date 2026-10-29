"""执行作者原始 TSM/PTA/CSP 和 YOLOX 推理层，提供所需的最小标准算子。

不修改原文件，不导入作者探索代码中的 DCN/XFeat，不安装或伪装 mmdet 包。
只支持公开 FIgLib 五帧 checkpoint 的结构，state_dict 必须严格匹配。
"""
import ast
import math
import typing

import torch
from torch import nn

from lwst_sources import ensure_sources


class BaseModule(nn.Module):
    """推理时仅需 Module 行为；不执行 MMEngine 训练初始化。"""
    def __init__(self, init_cfg=None):
        super().__init__()


class ConvModule(nn.Module):
    """Conv2d → BN → SiLU，与已固定源码的字段名、默认值兼容。"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 conv_cfg=None, norm_cfg=None, act_cfg=None, bias='auto'):
        super().__init__()
        if conv_cfg is not None or not norm_cfg or norm_cfg.get('type') != 'BN':
            raise ValueError('LwST 推理仅支持该 checkpoint 使用的 Conv2d + BN')
        if act_cfg.get('type') not in {'Swish', 'SiLU'}:
            raise ValueError('LwST 推理仅支持 SiLU')
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                              bias=False if bias == 'auto' else bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=norm_cfg.get('eps', 1e-5),
                                 momentum=norm_cfg.get('momentum', .1))
        self.activate = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.activate(self.bn(self.conv(x)))


def load_classes(path, names, namespace, methods=None):
    """从经过哈希校验的源码选择类，移除注册装饰器及无关依赖的导入。"""
    tree = ast.parse(path.read_text(encoding='utf-8'))
    selected = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name not in names:
            continue
        node.decorator_list = []
        if methods is not None:
            node.bases = [ast.Attribute(value=ast.Name(id='nn', ctx=ast.Load()), attr='Module', ctx=ast.Load())]
            node.body = [method for method in node.body if isinstance(method, ast.FunctionDef) and method.name in methods]
        selected.append(node)
    if len(selected) != len(names):
        raise ValueError(f'作者源码缺少所需类: {path}')
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)


def build_network():
    root = ensure_sources()
    namespace = dict(torch=torch, nn=nn, math=math, BaseModule=BaseModule,
                     ConvModule=ConvModule, Tensor=torch.Tensor,
                     ConfigType=dict, OptConfigType=typing.Optional[dict],
                     OptMultiConfig=typing.Optional[dict], Tuple=typing.Tuple, List=typing.List,
                     _BatchNorm=nn.modules.batchnorm._BatchNorm,
                     multi_apply=lambda function, *items: tuple(map(list, zip(*(function(*x) for x in zip(*items))))))
    load_classes(root / 'mmdet/models/layers/csp_layer.py', {'DarknetBottleneck', 'CSPLayer'}, namespace)
    load_classes(root / 'mmdet/models/backbones/csp_darknet_tsm.py',
                 {'Focus', 'SPPBottleneck', 'TemporalShift', 'CSPDarknetTSM'}, namespace)
    load_classes(root / 'mmdet/models/necks/yolox_pafpn_PTA.py',
                 {'IdentityLayer', 'PTA_Layers', 'YOLOXPAFPN_PTA'}, namespace)
    load_classes(root / 'mmdet/models/dense_heads/yolox_head.py', {'YOLOXHead'}, namespace,
                 methods={'_init_layers', '_build_stacked_convs', '_build_predictor', 'forward_single', 'forward'})
    norm = dict(type='BN', momentum=.03, eps=.001)
    act = dict(type='Swish')
    network = nn.Module()
    network.backbone = namespace['CSPDarknetTSM'](deepen_factor=.33, widen_factor=.5,
                                                 num_segments=5, shift_div=8, norm_cfg=norm, act_cfg=act)
    network.neck = namespace['YOLOXPAFPN_PTA'](
        in_channels=[128, 256, 512], out_channels=128, num_csp_blocks=1,
        ptas_cfg=dict(in_channels=[128, 256, 512], out_channels=[128, 256, 512],
                      T=5, partial=8, left_partial=6, group=2), norm_cfg=norm, act_cfg=act)
    head = namespace['YOLOXHead']()
    head.strides = (8, 16, 32)
    head.in_channels = head.feat_channels = 128
    head.stacked_convs = 2
    head.cls_out_channels = 1
    head.use_depthwise = head.dcn_on_last_conv = False
    head.conv_cfg = None
    head.norm_cfg, head.act_cfg, head.conv_bias = norm, act, 'auto'
    head._init_layers()
    network.bbox_head = head
    return network
