#!/usr/bin/env python3
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import sys, json, csv, time, hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT=Path("/mnt/workspace/AECIF_G4_v5_independent")
DATA=Path("/mnt/workspace/AECIF_Kaggle_bundle/VOCdevkit")
HELPER=Path("/mnt/workspace/g0_element_ce_dice_pilot.py")
RUN=ROOT/"diagnostics/runs/G0_element_ce_dice_pilot_20261122T064546Z"
BASE=ROOT/"runs/G0_A10_det_150_v1/best_model_1c9ce6832025f2a7f5ca54b9d956ce451109f7019e63cd6854f524f54d8927df.pth"
HEAD=RUN/"best_element_head_only.pth"
EXPECTED_SHA="76d6f522565b1f85cfa27c4298c0f78fefd43088669e79ef6f46e075e20aaf7a"
B=10000
SEED=20261122

def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for z in iter(lambda:f.read(8*1024*1024),b""): h.update(z)
    return h.hexdigest()

def met(cm):
    d=np.diag(cm).astype(float)
    u=cm.sum(0)+cm.sum(1)-d
    i=np.divide(d,u,out=np.full(7,np.nan),where=u>0)
    return float(np.nanmean(i)),i

def main():
    for p in [HELPER,BASE,HEAD,RUN/"epoch0_baseline.json",RUN/"epoch_05_metrics.json"]:
        if not p.is_file(): raise FileNotFoundError(p)
    if sha(BASE)!=EXPECTED_SHA: raise RuntimeError("G0 baseline SHA mismatch")
    sys.path[:0]=[str(ROOT),str(ROOT/"vendor"),str(HELPER.parent)]
    import g0_element_ce_dice_pilot as exp
    from g4_model import build_model
    from baseline_code.utils.dataloader_multi import SegmentationDataset
    exp.check_gpu_free()  # refuse to share the A10 with another compute process
    if not torch.cuda.is_available() or "A10" not in torch.cuda.get_device_name(0).upper():
        raise RuntimeError("Expected an available NVIDIA A10")
    cfg=json.loads((ROOT/"configs/protocol.json").read_text())
    tm=ROOT/cfg["data"]["train_manifest"]; vm=ROOT/cfg["data"]["validation_manifest"]
    if sha(tm)!=cfg["data"]["train_manifest_sha256"] or sha(vm)!=cfg["data"]["validation_manifest_sha256"]:
        raise RuntimeError("Locked manifest SHA mismatch")
    ids=exp.read_ids(vm)
    if len(ids)!=34: raise RuntimeError(f"Expected 34 validation images, got {len(ids)}")
    exp.configure_repro(int(cfg["training"]["seed"]))
    model=build_model("G0",num_classes=(7,2),backbone=cfg["model"]["backbone"],
                      pretrained=False,init_seed=int(cfg["training"]["seed"]))
    w=torch.load(BASE,map_location="cpu",weights_only=True)
    model.load_state_dict(w,strict=True); del w
    model.cuda().eval()
    ds=SegmentationDataset(ids,cfg["model"]["input_size"],cfg["model"]["num_classes"],False,str(DATA))
    loader=DataLoader(ds,batch_size=1,shuffle=False,num_workers=0,pin_memory=False,drop_last=False)

    def infer():
        out=[]
        model.eval()
        with torch.inference_mode():
            for image_id,batch in zip(ids,loader):
                x,ge,_=exp.masks_and_images(batch,torch.device("cuda:0"))
                ye,_=exp.outs(model(x)); ye=exp.matching_logits(ye,ge)
                pred=ye.argmax(1).cpu().numpy().astype(np.int64)
                out.append(exp.confusion(ge.cpu().numpy(),pred,7))
        if len(out)!=len(ids): raise RuntimeError("Validation count mismatch")
        return np.stack(out)

    print("BASELINE_INFERENCE",flush=True)
    cm0=infer()
    hs=torch.load(HEAD,map_location="cpu",weights_only=True)
    incompat = model.cif.last_layer_e.load_state_dict(hs, strict=False)

    expected_missing = {"1.running_mean", "1.running_var"}

    if set(incompat.missing_keys) != expected_missing or incompat.unexpected_keys:
        raise RuntimeError(
            f"Unexpected head-only checkpoint keys: "
            f"missing={incompat.missing_keys}, "
            f"unexpected={incompat.unexpected_keys}"
        )

    print(
        "HEAD_ONLY_LOAD_OK: trained parameters loaded; "
        "baseline BN buffers retained.",
        flush=True
    )
    del hs
    print("E2_INFERENCE",flush=True)
    cm1=infer()
    saved0=json.loads((RUN/"epoch0_baseline.json").read_text())["deterministic_fine_tune_protocol"]["element_cm"]
    saved1=json.loads((RUN/"epoch_05_metrics.json").read_text())["details"]["element_cm"]
    if not np.array_equal(cm0.sum(0),np.asarray(saved0)): raise RuntimeError("Fresh baseline does not match epoch0")
    if not np.array_equal(cm1.sum(0),np.asarray(saved1)): raise RuntimeError("Fresh E2 does not match epoch5")
    a,ia=met(cm0.sum(0)); b,ib=met(cm1.sum(0))
    rng=np.random.default_rng(SEED)
    db=np.empty(B); dclass=np.full((B,7),np.nan)
    n=len(ids)
    for k in range(B):
        ix=rng.integers(0,n,size=n)
        m0,i0=met(cm0[ix].sum(0)); m1,i1=met(cm1[ix].sum(0))
        db[k]=(m1-m0)*100
        dclass[k]=(i1-i0)*100
    ci=np.percentile(db,[2.5,97.5]).tolist()
    cci=[]
    for j in range(7):
        v=dclass[:,j]; v=v[np.isfinite(v)]
        cci.append(np.percentile(v,[2.5,97.5]).tolist() if len(v) else [None,None])
    rows=[]; image_d=[]
    for name,x,y in zip(ids,cm0,cm1):
        mx,ix=met(x); my,iy=met(y); delta=(my-mx)*100; image_d.append(delta)
        r={"image_id":name,"baseline_miou":mx,"e2_miou":my,"delta_pp":delta}
        for j in range(7):
            r[f"base_c{j}"]=None if not np.isfinite(ix[j]) else float(ix[j])
            r[f"e2_c{j}"]=None if not np.isfinite(iy[j]) else float(iy[j])
        rows.append(r)
    out=ROOT/"diagnostics/runs"/("G0_element_ce_dice_paired_bootstrap_"+time.strftime("%Y%m%dT%H%M%SZ",time.gmtime()))
    out.mkdir(parents=True,exist_ok=False)
    with (out/"per_image_metrics.csv").open("w",newline="",encoding="utf-8-sig") as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    np.savez_compressed(out/"per_image_confusion_matrices.npz",image_ids=np.asarray(ids),baseline=cm0,e2=cm1)
    summary={
      "status":"PAIRED_IMAGE_BOOTSTRAP_AUDIT_COMPLETE","training_performed":False,
      "validation_images":n,"bootstrap_samples":B,"bootstrap_seed":SEED,
      "baseline_sha256":sha(BASE),"e2_head_sha256":sha(HEAD),
      "fresh_inference_matches_saved_epoch0_and_epoch5":True,
      "baseline_miou":a,"e2_miou":b,"observed_delta_pp":(b-a)*100,
      "paired_image_bootstrap_95ci_pp":ci,
      "images_improved":int(np.sum(np.array(image_d)>0)),
      "images_declined":int(np.sum(np.array(image_d)<0)),
      "images_unchanged":int(np.sum(np.array(image_d)==0)),
      "mean_per_image_delta_pp":float(np.mean(image_d)),
      "baseline_class_iou":ia.tolist(),"e2_class_iou":ib.tolist(),
      "class_delta_pp":((ib-ia)*100).tolist(),"class_bootstrap_95ci_pp":cci,
      "baseline_pixel_errors":int(cm0.sum()-np.trace(cm0.sum(0))),
      "e2_pixel_errors":int(cm1.sum()-np.trace(cm1.sum(0))),
      "note":"Paired bootstrap resamples images, not pixels; no training or checkpoint writes."
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    if sha(BASE)!=EXPECTED_SHA: raise RuntimeError("Baseline checkpoint changed")
    print("PAIRED_IMAGE_BOOTSTRAP_AUDIT_COMPLETE",flush=True)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    print("OUTPUT_DIR:",out,flush=True)

if __name__=="__main__": main()
