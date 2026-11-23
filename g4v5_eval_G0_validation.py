#!/usr/bin/env python3
"""Read-only G0 baseline evaluation with exactly the same metric definitions as G4-v5."""
import argparse, csv, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import cv2

ROOT = Path('/mnt/workspace/AECIF_G4_v5_independent')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'vendor'))
from g4_model import build_model
from baseline_code.utils.dataloader_multi import SegmentationDataset
from checkpointing import state_dict_sha256


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def boundary(mask, valid):
    mask = (mask.astype(np.uint8) & valid.astype(np.uint8))
    kernel = np.ones((3,3), np.uint8)
    return ((cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel) > 0) & valid)


def boundary_counts(pred, gt, valid, tolerance):
    # Foreground (corrosion class 1) boundaries, 2-pixel tolerance at 520x520.
    pb = boundary(pred == 1, valid)
    gb = boundary(gt == 1, valid)
    npb, ngb = int(pb.sum()), int(gb.sum())
    if not npb and not ngb:
        return 0,0,0,0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*tolerance+1,2*tolerance+1))
    g_near = cv2.dilate(gb.astype(np.uint8), kernel) > 0
    p_near = cv2.dilate(pb.astype(np.uint8), kernel) > 0
    return int((pb & g_near).sum()), npb, int((gb & p_near).sum()), ngb


def cm_for(pred, gt, classes):
    valid = (gt >= 0) & (gt < classes)
    if not np.all((pred[valid] >= 0) & (pred[valid] < classes)):
        raise RuntimeError('Prediction outside valid class range')
    return np.bincount((gt[valid] * classes + pred[valid]).ravel(),minlength=classes*classes).reshape(classes,classes),valid


def iou_summary(cm):
    tp = np.diag(cm).astype(float)
    union = cm.sum(axis=0) + cm.sum(axis=1) - tp
    ious = np.divide(tp, union, out=np.full(len(tp), np.nan), where=union > 0)
    return ious, float(np.nanmean(ious)) if np.any(union > 0) else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', default=str(ROOT/'runs/G0_A10_det_150_v1'))
    p.add_argument('--output-dir', default='/mnt/workspace/g4v5_validation_eval/G0_20261120_v1')
    p.add_argument('--data-root', default='/mnt/workspace/AECIF_Kaggle_bundle/VOCdevkit')
    p.add_argument('--boundary-tolerance',type=int,default=2)
    a = p.parse_args()
    run, out, data_root = Path(a.run_dir).resolve(), Path(a.output_dir).resolve(), Path(a.data_root).resolve()
    if out.exists():
        raise SystemExit('Refusing to overwrite existing evaluation output directory: '+str(out))
    if not out.is_relative_to(Path('/mnt/workspace/g4v5_validation_eval')):
        raise SystemExit('Evaluation output must be beneath /mnt/workspace/g4v5_validation_eval')
    cfg = json.loads((ROOT/'configs/protocol.json').read_text())
    assert cfg['model']['num_classes'] == [7,2]
    assert cfg['model']['input_size'] == [520,520]
    manifest = ROOT / cfg['data']['validation_manifest']
    if sha256(manifest) != cfg['data']['validation_manifest_sha256']:
        raise RuntimeError('Validation manifest SHA mismatch')
    ids = [line.strip().split()[0] for line in manifest.read_text().splitlines() if line.strip()]
    if len(ids) != 34 or len(set(ids)) != 34:
        raise RuntimeError('Expected 34 unique validation IDs')
    best_files = list(run.glob('best_model_*.pth'))
    if len(best_files) != 1:
        raise RuntimeError(f'Expected exactly one best_model file, got {len(best_files)}')
    best_path = best_files[0]
    state = torch.load(best_path,map_location='cpu',weights_only=True)
    actual_state_sha = state_dict_sha256(state)
    if best_path.name != f'best_model_{actual_state_sha}.pth':
        raise RuntimeError('Best model filename does not match tensor SHA256')
    # Verify the selected best epoch from saved checkpoint metadata, without loading optimizer onto GPU.
    cp = torch.load(run/'checkpoint_latest.pth',map_location='cpu',weights_only=False)
    best_metadata = cp.get('best_state', None)
    if best_metadata is None:
        print('WARNING: best_state key not found in checkpoint; inspect checkpoint schema before publication',flush=True)
    else:
        print('Checkpoint best_state:',best_metadata,flush=True)
        if best_metadata.get('state_sha256') != actual_state_sha:
            raise RuntimeError('Best state SHA mismatch between checkpoint and artifact')
    del cp
    ds = SegmentationDataset(ids, cfg['model']['input_size'], cfg['model']['num_classes'], False, str(data_root))
    model = build_model('G0',num_classes=tuple(cfg['model']['num_classes']),backbone=cfg['model']['backbone'],pretrained=False,init_seed=cfg['training']['seed'])
    model.load_state_dict(state,strict=True)
    del state
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    ce = np.zeros((7,7),np.int64); cc = np.zeros((2,2),np.int64)
    b_tot = np.zeros(4,np.int64)
    records = []
    with torch.inference_mode():
        for idx, image_id in enumerate(ids):
            img, gt_e, gt_c, _, _ = ds[idx]
            inp = torch.from_numpy(np.ascontiguousarray(img)).float().unsqueeze(0).to(device)
            outputs = model(inp)
            if not isinstance(outputs,(list,tuple)) or len(outputs)<2:
                raise RuntimeError('Unexpected model output structure')
            preds=[]
            for output,n in zip(outputs[:2],(7,2)):
                if output.ndim != 4 or output.shape[0] != 1 or output.shape[1] != n:
                    raise RuntimeError(f'Unexpected logits shape: {tuple(output.shape)}')
                if tuple(output.shape[-2:]) != tuple(gt_e.shape):
                    output = F.interpolate(output,size=tuple(gt_e.shape),mode='bilinear',align_corners=True)
                preds.append(output.argmax(dim=1)[0].cpu().numpy().astype(np.int64))
            pred_e,pred_c=preds
            e,v_e=cm_for(pred_e,np.asarray(gt_e,dtype=np.int64),7)
            c,v_c=cm_for(pred_c,np.asarray(gt_c,dtype=np.int64),2)
            ce+=e;cc+=c
            be=boundary_counts(pred_c,np.asarray(gt_c,dtype=np.int64),v_c,a.boundary_tolerance)
            b_tot+=np.asarray(be,dtype=np.int64)
            ei,em=iou_summary(e); ci,cm=iou_summary(c)
            # Empty/empty per-image corrosion IoU is undefined, not arbitrarily 1 or 0.
            corrosion_union=int(c[:,1].sum()+c[1,:].sum()-c[1,1])
            rec={'image_id':image_id,'element_miou_present_classes':em,'corrosion_miou_present_classes':cm,
                 'corrosion_foreground_iou':float(ci[1]) if np.isfinite(ci[1]) else None,
                 'corrosion_foreground_union':corrosion_union,'element_valid_pixels':int(v_e.sum()),
                 'corrosion_valid_pixels':int(v_c.sum()),'boundary_matched_pred':be[0],
                 'boundary_pred_pixels':be[1],'boundary_matched_gt':be[2],'boundary_gt_pixels':be[3]}
            records.append(rec)
            print(f'[{idx+1:02d}/34] {image_id}: corr_fg_iou={rec["corrosion_foreground_iou"]}',flush=True)
    e_ious,e_miou=iou_summary(ce);c_ious,c_miou=iou_summary(cc)
    mp,npb,mg,ngb=map(int,b_tot)
    precision=mp/npb if npb else (1.0 if not ngb else 0.0)
    recall=mg/ngb if ngb else (1.0 if not npb else 0.0)
    f1=2*precision*recall/(precision+recall) if precision+recall else 0.0
    summary={'status':'EVALUATION_COMPLETE','group':'G0','split':'validation_34','evaluation_pixels':'520x520_letterbox',
             'model_path':str(best_path),'model_file_sha256':sha256(best_path),'model_state_sha256':actual_state_sha,
             'manifest_sha256':sha256(manifest),'num_images':len(ids),'element_miou_global':e_miou,
             'corrosion_miou_global':c_miou,'element_per_class_iou':[float(x) if np.isfinite(x) else None for x in e_ious],
             'corrosion_per_class_iou':[float(x) if np.isfinite(x) else None for x in c_ious],
             'corrosion_foreground_iou_global':float(c_ious[1]) if np.isfinite(c_ious[1]) else None,
             'miou_absent_class_policy':'exclude classes with zero union; global pixel-confusion aggregation',
             'ignored_label_policy':'ground-truth class >= num_classes ignored',
             'boundary_f1_corrosion_fg':f1,'boundary_precision':precision,'boundary_recall':recall,
             'boundary_tolerance_pixels':a.boundary_tolerance,
             'boundary_definition':'cv2 MORPH_GRADIENT 3x3, ellipse dilation matching on 520x520 letterboxed masks',
             'boundary_counts':{'matched_pred':mp,'pred_boundary_pixels':npb,'matched_gt':mg,'gt_boundary_pixels':ngb},
             'note':'Uses exactly the same inference, global IoU, and 2-pixel boundary metric implementation as the G4-v5 validation evaluator; protocol equivalence of training must be verified separately.'}
    out.mkdir(parents=True)
    (out/'G0_validation_metrics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    with (out/'G0_per_image_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    with (out/'G0_confusion_matrices.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['task','gt_class','pred_class','pixels'])
        for task,m in [('element',ce),('corrosion',cc)]:
            for i in range(len(m)):
                for j in range(len(m)):
                    w.writerow([task,i,j,int(m[i,j])])
    print('SUMMARY:',json.dumps({k:summary[k] for k in ('element_miou_global','corrosion_miou_global','corrosion_foreground_iou_global','boundary_f1_corrosion_fg')},indent=2))
    print('OUTPUT:',out)


if __name__=='__main__':main()
