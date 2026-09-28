"""Generate editable, responsive SVG summaries for the model documentation."""

import argparse
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1] / 'docs' / 'assets' / 'models'

MODELS = [
    {
        'slug': 'clft', 'title': 'CLFT', 'subtitle': 'Global ViT features reassembled into a progressive fusion decoder',
        'color': '#66d9e8', 'takeaway': 'Fusion follows token reassembly at several decoder scales.',
        'steps': [
            ('Inputs', ('RGB image', 'LiDAR projection'), 'two streams'),
            ('Shared ViT', ('Global-attention', 'token features'), 'encoder'),
            ('Reassemble', ('Hooked tokens →', 'spatial maps'), 'multi-scale'),
            ('Residual fusion', ('RGB + LiDAR +', 'previous scale'), 'fusion point'),
            ('Segmentation', ('Dense class', 'logits'), 'output'),
        ], 'focus': 3,
    },
    {
        'slug': 'clftv2', 'title': 'CLFTv2',
        'subtitle': 'Hierarchical features and a top-down fusion pyramid',
        'color': '#a2e073', 'takeaway': 'Native Swin feature maps replace CLFT’s ViT token reassembly.',
        'steps': [
            ('Inputs', ('RGB image', 'LiDAR projection'), 'two streams'),
            ('Swin V2', ('Shifted-window', 'feature hierarchy'), 'encoder'),
            ('Per-scale maps', ('Project both', 'modality features'), 'multi-scale'),
            ('Residual fusion', ('Merge streams +', 'coarser stage'), 'fusion point'),
            ('Segmentation', ('Upsampled', 'class logits'), 'output'),
        ], 'focus': 3,
    },
    {
        'slug': 'maskformer', 'title': 'MaskFormerFusion',
        'subtitle': 'Feature fusion followed by query-based mask classification',
        'color': '#b99aff', 'takeaway': 'Queries predict class-labelled masks; their scores form the semantic map.',
        'steps': [
            ('Inputs', ('RGB image', 'LiDAR projection'), 'two streams'),
            ('Shared Swin', ('Multi-scale', 'features'), 'encoder'),
            ('Feature fusion', ('Residual sum', 'at each scale'), 'fusion point'),
            ('FPN + queries', ('Pixel features →', 'class + masks'), 'decoder'),
            ('Segmentation', ('Merge query', 'predictions'), 'output'),
        ], 'focus': 3,
    },
    {
        'slug': 'mask2former', 'title': 'Mask2FormerFusion',
        'subtitle': 'Multi-scale pixel features guide masked-attention queries',
        'color': '#f7ba72', 'takeaway': 'Masked attention narrows each query to its predicted foreground.',
        'steps': [
            ('Inputs', ('RGB image', 'LiDAR projection'), 'two streams'),
            ('Shared Swin', ('Multi-scale', 'features'), 'encoder'),
            ('Feature fusion', ('Residual sum', 'at each scale'), 'fusion point'),
            ('Pixel decoder', ('Deformable', 'multi-scale maps'), 'multi-scale'),
            ('Masked queries', ('Class + masks →', 'semantic map'), 'decoder'),
        ], 'focus': 4,
    },
    {
        'slug': 'deeplabv3plus', 'title': 'DeepLabV3+ · late fusion',
        'subtitle': 'Two complete encoder–decoder branches meet just before classification',
        'color': '#ff8ea5', 'takeaway': 'Fusion happens after ASPP and decoding, unlike the transformer variants.',
        'steps': [
            ('Inputs', ('RGB image', 'LiDAR projection'), 'two streams'),
            ('Two ResNets', ('Separate', 'ResNet-101 paths'), 'encoder'),
            ('ASPP + decode', ('Context + fine', 'spatial detail'), 'two branches'),
            ('Late fusion', ('Average branch', 'feature maps'), 'fusion point'),
            ('Segmentation', ('Shared classifier', '→ class logits'), 'output'),
        ], 'focus': 3,
    },
]


def txt(x, y, content, *, size=18, weight=500, fill='#dce7f5', extra=''):
    return (f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}" {extra}>{escape(content)}</text>')


def header(title, subtitle, color, label):
    return '\n'.join([
        '<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="550" viewBox="0 0 1280 550" role="img">',
        f'<title>{escape(title)} architecture overview</title>',
        f'<desc>{escape(subtitle)}</desc>',
        '<defs><filter id="shadow" x="-20%" y="-30%" width="140%" height="160%">'
        '<feDropShadow dx="0" dy="10" stdDeviation="12" flood-color="#010711" flood-opacity=".3"/>'
        '</filter></defs>',
        '<rect width="1280" height="550" rx="26" fill="#0e1b2e"/>',
        f'<rect x="52" y="45" width="118" height="30" rx="15" fill="{color}" fill-opacity=".14"/>',
        txt(70, 66, label, size=14, weight=700, fill=color, extra='letter-spacing="1.2"'),
        txt(52, 118, title, size=36, weight=750, fill='#f5f9ff'),
        txt(52, 150, subtitle, size=18, fill='#afc2d9'),
    ])


def card(index, x, title, details, tag, color, focused):
    border = color if focused else '#38506e'
    fill = '#1a3048' if focused else '#15253b'
    parts = [
        f'<rect x="{x}" y="198" width="216" height="210" rx="20" fill="{fill}" '
        f'stroke="{border}" stroke-width="{2 if focused else 1}" filter="url(#shadow)"/>',
        f'<circle cx="{x+34}" cy="234" r="15" fill="{color}" fill-opacity=".16"/>',
        txt(x+28, 240, str(index+1), size=17, weight=700, fill=color),
        txt(x+20, 284, title, size=21, weight=700, fill='#f4f8ff'),
    ]
    if index == 0:
        parts += [
            f'<rect x="{x+20}" y="305" width="176" height="33" rx="9" fill="#263f59"/>',
            f'<circle cx="{x+35}" cy="321" r="5" fill="#75cfff"/>',
            txt(x+48, 327, details[0], size=16),
            f'<rect x="{x+20}" y="345" width="176" height="33" rx="9" fill="#263f59"/>',
            f'<circle cx="{x+35}" cy="361" r="5" fill="#f8bc76"/>',
            txt(x+48, 367, details[1], size=16),
        ]
    else:
        parts += [txt(x+20, 325, details[0], size=17), txt(x+20, 351, details[1], size=17)]
    parts += [txt(x+20, 390, tag.upper(), size=12, weight=700, fill=color, extra='letter-spacing="1.2"')]
    return '\n'.join(parts)


def model_svg(model):
    parts = [header(model['title'], model['subtitle'], model['color'], 'MODEL FLOW')]
    for i, (title, details, tag) in enumerate(model['steps']):
        x = 52 + i * 240
        parts.append(card(i, x, title, details, tag, model['color'], i == model['focus']))
    for i in range(4):
        x = 52 + i * 240
        parts.append(f'<line x1="{x+217}" y1="304" x2="{x+237}" y2="304" '
                     f'stroke="{model["color"]}" stroke-width="3"/>')
        parts.append(f'<polygon points="{x+231},298 {x+240},304 {x+231},310" '
                     f'fill="{model["color"]}"/>')
    parts += [
        f'<rect x="52" y="445" width="1176" height="60" rx="14" fill="{model["color"]}" fill-opacity=".09"/>',
        txt(74, 481, 'CORE IDEA', size=13, weight=700, fill=model['color'], extra='letter-spacing="1.3"'),
        txt(220, 482, model['takeaway'], size=18, fill='#e4edf8'),
        '</svg>',
    ]
    return '\n'.join(parts) + '\n'


def comparison_svg():
    rows = [
        ('CLFT', 'Global ViT', 'Fusion while decoding', 'Dense logits', '#66d9e8'),
        ('CLFTv2', 'Hierarchical Swin', 'Fusion at each pyramid scale', 'Dense logits', '#a2e073'),
        ('MaskFormer', 'Hierarchical Swin', 'Fusion before FPN', 'Queries → masks', '#b99aff'),
        ('Mask2Former', 'Hierarchical Swin', 'Fusion before pixel decoder', 'Masked queries', '#f7ba72'),
        ('DeepLabV3+', 'Two ResNet-101s', 'Fusion after both decoders', 'Dense logits', '#ff8ea5'),
    ]
    parts = [header('Five ways to combine two views',
                    'Same inputs and task; different feature extraction, fusion point and prediction head.',
                    '#88bfff', 'MODEL COMPARISON')]
    columns = [(64, 'MODEL'), (323, 'ENCODER'), (620, 'WHERE STREAMS MEET'), (1000, 'PREDICTION')]
    for x, label in columns:
        parts.append(txt(x, 196, label, size=13, weight=700, fill='#8da9c7', extra='letter-spacing="1.1"'))
    for i, row in enumerate(rows):
        y = 211 + i * 60
        parts += [f'<rect x="52" y="{y}" width="1176" height="51" rx="12" fill="#182a42"/>',
                  f'<rect x="52" y="{y}" width="5" height="51" rx="2" fill="{row[4]}"/>',
                  txt(65, y+33, row[0], size=18, weight=700, fill=row[4]),
                  txt(323, y+33, row[1], size=16),
                  txt(620, y+33, row[2], size=16),
                  txt(1000, y+33, row[3], size=16)]
    parts += ['</svg>']
    return '\n'.join(parts) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='fail if committed SVGs need regeneration')
    args = parser.parse_args()
    diagrams = {f'{model["slug"]}.svg': model_svg(model) for model in MODELS}
    diagrams['comparison.svg'] = comparison_svg()
    if not args.check:
        ROOT.mkdir(parents=True, exist_ok=True)
    for filename, contents in diagrams.items():
        path = ROOT / filename
        if args.check:
            if not path.exists() or path.read_text(encoding='utf-8') != contents:
                parser.error(f'{path} is stale; run python tools/generate_model_diagrams.py')
        else:
            path.write_text(contents, encoding='utf-8')


if __name__ == '__main__':
    main()
