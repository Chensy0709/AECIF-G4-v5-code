#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent, safe, 5-epoch Element-head-only G4 pilot on the locked 200/34 split.

NOT a formal full-network ablation; exploratory fine-tuning of a preselected G4 checkpoint.
Does not touch locked sources or baseline weights. Validation used for epoch selection,
held-out test set is never accessed. All of G4 except cif.last_layer_e stays frozen in eval.
"""
from __future__ import annotations

import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

DEFAULT_ROOT = Path('/mnt/workspace/AECIF_G4_v5_independent')
DEFAULT_DATA = Path('/mnt/workspace/AECIF_Kaggle_bundle/VOCdevkit')
DEFAULT_BEST = DEFAULT_ROOT / 'runs/G0_A10_det_150_v1/best_model_1c9ce6832025f2a7f5ca54b9d956ce451109f7019e63cd6854f524f54d8927df.pth'
EXPECTED_BEST_SHA = '76d6f522565b1f85cfa27c4298c0f78fefd43088669e79ef6f46e075e20aaf7a'
EXPECTED_G0_E = 0.879966915511981
EXPECTED_G0_C = 0.8634959270833518
HEAD_PATH = 'cif.last_layer_e'
HEAD_PARAMS = 267271


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read_ids(p):
    return [s.strip().split()[0] for s in p.read_text(encoding='utf-8-sig').splitlines() if s.strip()]


def check_gpu_free():
    # Must run BEFORE any cuda context is created: DSW self-PID may be '[Not Found]'.
    for i in range(3):
        cp = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name',
                             '--format=csv,noheader'], text=True, capture_output=True, check=True)
        busy = [s.strip() for s in cp.stdout.splitlines() if s.strip()]
        if busy:
            raise RuntimeError('Other GPU compute processes detected; do not share GPU: ' + '; '.join(busy))
        if i != 2:
            time.sleep(.6)


def configure_repro(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


class ReproTrainDataset(Dataset):
    def __init__(self, dataset, ids, seed):
        self.dataset = dataset
        self.ids = ids
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Isolate both Python and NumPy augmentation RNG; deterministic per image/epoch.
        np_state, py_state = np.random.get_state(), random.getstate()
        digest = hashlib.sha256(f'{self.seed}:{self.epoch}:{self.ids[idx]}'.encode()).digest()
        seed = int.from_bytes(digest[:4], 'little')
        try:
            np.random.seed(seed)
            random.seed(seed)
            return self.dataset[idx]
        finally:
            np.random.set_state(np_state)
            random.setstate(py_state)


def masks_and_images(batch, device):
    images, gt_e, gt_c = batch[:3]
    x = torch.as_tensor(np.ascontiguousarray(images), dtype=torch.float32, device=device)
    if x.ndim != 4 or x.shape[1] != 3:
        raise ValueError('Unexpected input batch shape: '+str(tuple(x.shape)))
    gt_e = torch.as_tensor(np.ascontiguousarray(gt_e), dtype=torch.int64, device=device)
    gt_c = torch.as_tensor(np.ascontiguousarray(gt_c), dtype=torch.int64, device=device)
    return x, gt_e, gt_c


def outs(output):
    if not isinstance(output, (tuple, list)) or len(output) < 2:
        raise RuntimeError('Unexpected dual-task output; expected tuple/list Element, Corrosion')
    return output[0], output[1]


def matching_logits(logits, gt):
    if logits.shape[-2:] != gt.shape[-2:]:
        logits = F.interpolate(logits, size=gt.shape[-2:], mode='bilinear', align_corners=True)
    return logits


def confusion(gt, pred, classes):
    gt, pred = np.asarray(gt).astype(np.int64), np.asarray(pred).astype(np.int64)
    valid = (gt >= 0) & (gt < classes)
    if not np.all((pred[valid] >= 0) & (pred[valid] < classes)):
        raise RuntimeError('Out of range predicted class')
    return np.bincount((gt[valid]*classes+pred[valid]).reshape(-1), minlength=classes*classes).reshape(classes, classes)


def metrics(cm):
    intersection = np.diag(cm).astype(float)
    union = cm.sum(axis=0) + cm.sum(axis=1) - intersection
    ious = np.divide(intersection, union, out=np.full(len(union), np.nan), where=(union>0))
    return {'miou': float(np.nanmean(ious)),
            'ious': [float(v) if np.isfinite(v) else None for v in ious],
            'pixel_errors': int(cm.sum() - intersection.sum()),
            'pixel_accuracy': float(intersection.sum()/cm.sum())}


def evaluate(model, val_ds, device, batch_size):
    model.eval()  # essential: keep all BN running stats frozen
    ce, cc = np.zeros((7,7), dtype=np.int64), np.zeros((2,2), dtype=np.int64)
    corrosion_pred_hash = hashlib.sha256()
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0,
                        pin_memory=False, drop_last=False)
    with torch.inference_mode():
        for batch in loader:
            x, gt_e, gt_c = masks_and_images(batch, device)
            y_e, y_c = outs(model(x))
            y_e = matching_logits(y_e, gt_e)
            y_c = matching_logits(y_c, gt_c)
            pe = y_e.argmax(dim=1).cpu().numpy().astype(np.int64)
            pc = y_c.argmax(dim=1).cpu().numpy().astype(np.int64)
            ge = gt_e.cpu().numpy()
            gc = gt_c.cpu().numpy()
            ce += confusion(ge, pe, 7)
            cc += confusion(gc, pc, 2)
            # Same validation order every epoch: the digest tests pixel-for-pixel invariance.
            corrosion_pred_hash.update(pc.astype(np.uint8).tobytes(order='C'))
    return {'element': metrics(ce), 'corrosion': metrics(cc),
            'corrosion_prediction_sha256': corrosion_pred_hash.hexdigest(),
            'element_cm': ce.tolist(), 'corrosion_cm': cc.tolist(),
            'element_4_to_2': int(ce[4,2]), 'element_6_to_5': int(ce[6,5])}


def atomic_save(state, p):
    temp = p.with_suffix('.tmp')
    try:
        torch.save(state, temp)
        os.replace(temp, p)
    finally:
        if temp.exists(): temp.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    ap.add_argument('--data', type=Path, default=DEFAULT_DATA)
    ap.add_argument('--best', type=Path, default=DEFAULT_BEST)
    ap.add_argument('--check-only', action='store_true', help='CPU-only file/model/head audit')
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--max-minutes', type=float, default=45.0)
    ap.add_argument('--output', type=Path)
    args=ap.parse_args()
    if not 1 <= args.epochs <= 10: raise ValueError('pilot epochs must be 1..10')
    if not 0 < args.lr <= 3e-4: raise ValueError('unsafe learning rate')
    root=args.root.resolve(); data=args.data.resolve(); best=args.best.resolve()
    if str(root) not in sys.path: sys.path.insert(0,str(root))
    if str(root/'vendor') not in sys.path:sys.path.insert(0,str(root/'vendor'))
    if not best.is_file() or sha256_file(best) != EXPECTED_BEST_SHA:
        raise RuntimeError('G4 best artifact missing or SHA256 mismatch; refusing to train')
    if not data.is_dir():raise FileNotFoundError(data)
    cfg=json.loads((root/'configs/protocol.json').read_text(encoding='utf-8'))
    assert cfg['model']['num_classes']==[7,2] and cfg['model']['input_size']==[520,520]
    train_manifest=root/cfg['data']['train_manifest'];val_manifest=root/cfg['data']['validation_manifest']
    for kind,path in [('train',train_manifest),('validation',val_manifest)]:
        if sha256_file(path)!=cfg['data'][f'{kind}_manifest_sha256']:
            raise RuntimeError(kind+' locked manifest hash mismatch')
    train_ids=read_ids(train_manifest);val_ids=read_ids(val_manifest)
    if len(train_ids)!=200 or len(val_ids)!=34 or set(train_ids)&set(val_ids):
        raise RuntimeError('Expected disjoint train=200/validation=34')

    # Read and verify checkpoint only on CPU before GPU preflight.
    from g4_model import build_model
    from baseline_code.utils.dataloader_multi import SegmentationDataset
    configure_repro(int(cfg['training']['seed']))
    model=build_model('G0',num_classes=(7,2),backbone=cfg['model']['backbone'],
                      pretrained=False,init_seed=int(cfg['training']['seed']))
    weights=torch.load(best,map_location='cpu',weights_only=True)
    model.load_state_dict(weights,strict=True)
    del weights
    if not hasattr(model, 'cif') or not hasattr(model.cif, 'last_layer_e'):
        raise RuntimeError('Unexpected model architecture; missing '+HEAD_PATH)
    head=model.cif.last_layer_e
    named_head=[(n,p) for n,p in model.named_parameters() if n.startswith(HEAD_PATH+'.')]
    if sum(p.numel() for _,p in named_head)!=HEAD_PARAMS:
        raise RuntimeError('Element head count changed; expected '+str(HEAD_PARAMS))
    if any(n.startswith('cif.last_layer_d.') for n,p in named_head):
        raise RuntimeError('Element head unexpectedly overlaps corrosion head')
    param_count=sum(p.numel() for p in model.parameters())
    print('CPU_HEAD_AUDIT_OK',json.dumps({'train':len(train_ids),'val':len(val_ids),
          'trainable_params':HEAD_PARAMS,'all_params':param_count,
          'head_names':[n for n,p in named_head], 'G0_best_sha256':EXPECTED_BEST_SHA,
          'train_manifest_sha256':sha256_file(train_manifest),
          'val_manifest_sha256':sha256_file(val_manifest)},ensure_ascii=False),flush=True)
    if args.check_only:
        print('HEAD_ONLY_CHECK_COMPLETE: no CUDA initialized and no files changed.',flush=True)
        return

    out=args.output.resolve() if args.output else (root/'diagnostics/runs'/('G0_element_weighted_ce_pilot_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())))
    dest_base=(root/'diagnostics/runs').resolve()
    if not out.is_relative_to(dest_base):raise ValueError('output must be inside diagnostics/runs')
    if out.exists():raise FileExistsError('Refuse to overwrite '+str(out))
    # Prevent occupying active GPU or mistaken selection of a different device.
    check_gpu_free()
    if not torch.cuda.is_available() or 'A10' not in torch.cuda.get_device_name(0).upper():
        raise RuntimeError('Expected one free NVIDIA A10')
    device=torch.device('cuda:0')
    model.to(device)
    for param in model.parameters():param.requires_grad_(False)
    for _,param in named_head:param.requires_grad_(True)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==HEAD_PARAMS
    model.eval()  # training only head with fixed BN stats and deterministic, stable features
    if model.cif.last_layer_e.training:
        raise RuntimeError('BN in Element head is in train mode: must be eval')
    optimizer=torch.optim.AdamW([p for _,p in named_head], lr=args.lr,weight_decay=1e-4)
    # Class-balanced weights: normalized inverse square-root frequency
    class_pixel_counts=torch.tensor(
        [49449430, 60139672, 23100509, 20071331,
         15349864, 217597934, 147906525],
        dtype=torch.float64, device=device)
    class_frequency=class_pixel_counts/class_pixel_counts.sum()
    element_class_weights=torch.rsqrt(class_frequency)
    element_class_weights=(element_class_weights/element_class_weights.mean()).float()

    if element_class_weights.numel()!=7 or not bool(torch.isfinite(element_class_weights).all()):
        raise RuntimeError('Invalid Element class weights')

    print('G0_CLASS_BALANCED_CE',
          json.dumps({
              'pixel_counts':[int(x) for x in class_pixel_counts.cpu().tolist()],
              'weights':[round(float(x),8) for x in element_class_weights.cpu().tolist()]
          }),flush=True)
    base_train=SegmentationDataset(train_ids,cfg['model']['input_size'],cfg['model']['num_classes'],True,str(data))
    train_ds=ReproTrainDataset(base_train,train_ids,int(cfg['training']['seed'])+500001)
    val_ds=SegmentationDataset(val_ids,cfg['model']['input_size'],cfg['model']['num_classes'],False,str(data))
    print('Evaluating original G0 validation protocol (without forced deterministic kernels)...',flush=True)
    start=time.monotonic()
    # The locked historical validation/branch-probe scripts did not enable
    # torch.use_deterministic_algorithms or cudnn.deterministic. Compare their
    # numerical protocol first, WITHOUT training and WITHOUT relaxing tolerance.
    saved_det_alg=torch.are_deterministic_algorithms_enabled()
    saved_cudnn_det=torch.backends.cudnn.deterministic
    try:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic=False
        original_protocol_baseline=evaluate(model,val_ds,device,batch_size=1)
    finally:
        torch.backends.cudnn.deterministic=saved_cudnn_det
        torch.use_deterministic_algorithms(saved_det_alg)
    print('HISTORICAL_PROTOCOL_CHECK',
          f'E={original_protocol_baseline["element"]["miou"]:.9f}',
          f'C={original_protocol_baseline["corrosion"]["miou"]:.9f}',flush=True)
    if (abs(original_protocol_baseline['element']['miou']-EXPECTED_G0_E)>1e-6 or
        abs(original_protocol_baseline['corrosion']['miou']-EXPECTED_G0_C)>1e-6):
        raise RuntimeError('Original-protocol metrics still differ. Stop before training; '
            'please share HISTORICAL_PROTOCOL_CHECK and traceback: '
            +str(original_protocol_baseline['element']['miou'])+' / '
            +str(original_protocol_baseline['corrosion']['miou']))
    print('HISTORICAL_G0_REPRODUCED_WITHIN_1E-6',flush=True)
    # Fine-tuning and all later validation use the stricter deterministic
    # setting consistently. Small FP32/cuDNN-algorithm changes are logged,
    # not mistaken for weight improvements.
    baseline=evaluate(model,val_ds,device,batch_size=1)
    print('DETERMINISTIC_EPOCH0',
          f'E={baseline["element"]["miou"]:.9f}',
          f'C={baseline["corrosion"]["miou"]:.9f}',flush=True)
    if (abs(baseline['element']['miou']-EXPECTED_G0_E)>2e-5 or
        abs(baseline['corrosion']['miou']-EXPECTED_G0_C)>2e-5):
        raise RuntimeError('Deterministic-kernel difference is unexpectedly large; do not train.')
    out.mkdir(parents=True,exist_ok=False)
    (out/'epoch0_baseline.json').write_text(json.dumps({
        'historical_protocol':original_protocol_baseline,
        'deterministic_fine_tune_protocol':baseline,
        'historical_expected':{'element_miou':EXPECTED_G0_E,'corrosion_miou':EXPECTED_G0_C},
        'diagnostic':'Different cuDNN determinism algorithms; no model weights updated prior to epoch 1',
    },ensure_ascii=False,indent=2)+'\n')
    records=[]
    best_miou=baseline['element']['miou']
    best_epoch=0
    best_head_path=out/'best_element_head_only.pth'
    g=torch.Generator(device='cpu')
    for epoch in range(1,args.epochs+1):
        if (time.monotonic()-start)/60 >= args.max_minutes:
            print('PILOT_TIME_BUDGET_REACHED: stopping safely before next epoch',flush=True)
            break
        train_ds.epoch=epoch
        g.manual_seed(int(cfg['training']['seed'])+929*epoch)
        loader=DataLoader(train_ds,batch_size=4,shuffle=True,generator=g,num_workers=0,
                          drop_last=True,pin_memory=False)
        model.eval() # NEVER call model.train() or BN stats will shift
        running=0.0; valid_items=0
        epoch_start=time.monotonic()
        for batch in loader:
            x,gt_e,_=masks_and_images(batch,device)
            optimizer.zero_grad(set_to_none=True)
            y_e,_=outs(model(x))
            y_e=matching_logits(y_e,gt_e)
            valid=(gt_e>=0)&(gt_e<7)
            if not torch.any(valid):raise RuntimeError('Empty Element valid pixels')
            label=torch.where(valid,gt_e,torch.full_like(gt_e,7))
            log_probs=F.log_softmax(y_e.float(),dim=1)
            safe_label=label.masked_fill(label==7,0)
            nll=-log_probs.gather(1,safe_label.unsqueeze(1)).squeeze(1)
            valid_pixels=(label!=7)
            pixel_weights=element_class_weights[safe_label]*valid_pixels.to(log_probs.dtype)
            denominator=pixel_weights.sum()
            if not bool(denominator>0):
                raise RuntimeError('No valid Element pixels')
            loss=(nll*pixel_weights).sum()/denominator
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite train CE loss')
            loss.backward()
            # Explicitly ensure no non-Element parameters received gradients.
            if any(p.grad is not None for n,p in model.named_parameters() if not n.startswith(HEAD_PATH+'.')):
                raise RuntimeError('A frozen parameter received gradients; refusing to continue')
            torch.nn.utils.clip_grad_norm_([p for _,p in named_head],max_norm=1.0)
            optimizer.step()
            bs=x.shape[0]; running+=float(loss.item())*bs;valid_items+=bs
        val=evaluate(model,val_ds,device,batch_size=1)
        if val['corrosion_prediction_sha256']!=baseline['corrosion_prediction_sha256'] or val['corrosion_cm']!=baseline['corrosion_cm']:
            raise RuntimeError('Corrosion predictions changed despite frozen branch: stop immediately')
        changed=val['element']['miou']-baseline['element']['miou']
        record={'epoch':epoch,'train_ce':running/valid_items,'element_miou':val['element']['miou'],
                'corrosion_miou':val['corrosion']['miou'],'element_delta_pp_vs_G4':100*changed,
                'element_pixel_errors':val['element']['pixel_errors'],
                'element_class2_iou':val['element']['ious'][2],
                'element_class4_iou':val['element']['ious'][4],
                'element_4_to_2':val['element_4_to_2'],'element_6_to_5':val['element_6_to_5'],
                'seconds':round(time.monotonic()-epoch_start,3)}
        records.append(record)
        (out/f'epoch_{epoch:02d}_metrics.json').write_text(json.dumps({'record':record,'details':val},ensure_ascii=False,indent=2)+'\n')
        with (out/'training_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
        if val['element']['miou']>best_miou:
            best_miou=val['element']['miou'];best_epoch=epoch
            atomic_save({n.removeprefix(HEAD_PATH+'.'):p.detach().cpu().clone() for n,p in named_head},best_head_path)
        print(f'EPOCH {epoch}/{args.epochs} | E={val["element"]["miou"]:.6%} '
              f'({changed*100:+.4f} pp vs G4) | C={val["corrosion"]["miou"]:.6%} '
              f'| class2={val["element"]["ious"][2]:.6%} class4={val["element"]["ious"][4]:.6%} '
              f'| trainCE={record["train_ce"]:.6f} | {record["seconds"]:.1f}s',flush=True)
    if sha256_file(best)!=EXPECTED_BEST_SHA:
        raise RuntimeError('Original G4 checkpoint was modified: serious integrity failure')
    status={'status':'HEAD_ONLY_PILOT_COMPLETE','scope':'EXPLORATORY_HEAD_ONLY_5E',
            'warning':'NOT a fair independently initialized G0/G4 ablation; do not claim formal improvement based on validation-selected checkpoint alone.',
            'baseline_sha256':EXPECTED_BEST_SHA,'train_ids':len(train_ids),'val_ids':len(val_ids),
            'trainable_parameter_count':HEAD_PARAMS,'other_parameters_frozen':True,
            'batch_norm_running_stats_frozen':True,'precision':'FP32', 'evaluation_protocol':'historical baseline check then deterministic-kernel pilot','epochs_requested':args.epochs,
            'epochs_completed':len(records),'learning_rate':args.lr,
            'baseline_element_miou':baseline['element']['miou'],
            'baseline_corrosion_miou':baseline['corrosion']['miou'],
            'historical_protocol_element_miou':original_protocol_baseline['element']['miou'],
            'historical_protocol_corrosion_miou':original_protocol_baseline['corrosion']['miou'],
            'corrosion_prediction_sha256':baseline['corrosion_prediction_sha256'],
            'best_epoch':best_epoch,'best_element_miou':best_miou,
            'best_checkpoint':'best_element_head_only.pth' if best_epoch>0 else None,
            'best_element_delta_pp_vs_G0':(best_miou-0.879966915511981)*100,
            'notes':'Only Element head was optimized from frozen G4 best checkpoint. No external test images touched. Per-epoch corrosion prediction digest was checked.'}
    (out/'summary.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
    pzip=out.parent/(out.name+'_REPORT.zip')
    with zipfile.ZipFile(pzip,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in out.iterdir():
            if p.is_file():z.write(p,p.name)
    print('HEAD_ONLY_PILOT_COMPLETE',flush=True)
    print('BEST_ELEMENT=',f'{best_miou:.6%}','BEST_EPOCH=',best_epoch,flush=True)
    print('UNCHANGED_CORROSION=',f'{baseline["corrosion"]["miou"]:.6%}',flush=True)
    print('FINAL_REPORT_ZIP=',pzip,flush=True)

if __name__=='__main__':main()
