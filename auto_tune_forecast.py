# -*- coding: utf-8 -*-
"""
auto_tune_forecast.py — MAÁTria Energia · Auto-Tuning v2
=========================================================
Redesenhado após diagnóstico v1: R² negativo = 31 features x 220 amostras.

Estratégia v2 (progressiva):
  FASE 0 — Diagnóstico do dataset (distribuição, correlações)
  FASE 1 — Baselines simples (média, Ridge, Lasso) → piso de qualidade
  FASE 2 — Feature selection agressiva (MI + correlação → top 5-8)
  FASE 3 — GBM raso (depth≤3) com features selecionadas
  FASE 4 — Target winsorized (remoção de outliers extremos)
  FASE 5 — Refinamento ao redor do melhor encontrado
"""
from __future__ import annotations
import argparse, json, os, time, warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")

PMO_XLSX = Path(os.getenv("PMO_XLSX","data/ons/PMOs/validacao_pmo.xlsx"))
MODEL_DIR = Path(os.getenv("MODEL_DIR","data/models"))
DATA_DIR = Path(os.getenv("DATA_DIR","data"))
META_DIR = MODEL_DIR / "meta"
META_DIR.mkdir(parents=True, exist_ok=True)
SUBSISTEMAS = ["seco","s","ne","n"]
SUB_LABEL = {"seco":"SE/CO","s":"Sul","ne":"Nordeste","n":"Norte"}
QUANTILES = [0.10, 0.50, 0.90]

DEFAULT_TARGETS = {"r2_test_min":0.05,"mae_pld_max":50.0,"overfit_gap_max":0.25,
    "coverage_p10_range":(0.03,0.22),"coverage_p90_range":(0.78,0.97)}

def _score(m):
    r2=m.get("r2_test",-1); r2cv=m.get("r2_cv_mean",-1) or -1
    mae=m.get("mae_pld",999); cp10=m.get("coverage_p10",0.5); cp90=m.get("coverage_p90",0.5)
    s_r2 = min(100, (r2+0.5)/1.0*100) if r2>=0 else max(0, 50+r2*50)
    s_mae = max(0, min(100, (1-mae/100)*100))
    s_cov = max(0, 100-(abs(cp10-0.10)+abs(cp90-0.90))*200)
    gap = abs(r2-(r2cv if r2cv else r2))
    s_gap = max(0, min(100, (1-gap/0.5)*100))
    return round(0.35*s_r2 + 0.30*s_mae + 0.20*s_cov + 0.15*s_gap, 2)

def _targets_met(m, targets=DEFAULT_TARGETS):
    msgs,ok = [],True
    r2=m.get("r2_test",-1); mae=m.get("mae_pld",999)
    if r2<targets["r2_test_min"]: msgs.append(f"R²={r2:.3f} < {targets['r2_test_min']}"); ok=False
    else: msgs.append(f"✅ R²={r2:.3f}")
    if mae>targets["mae_pld_max"]: msgs.append(f"MAE={mae:.1f} > {targets['mae_pld_max']}"); ok=False
    else: msgs.append(f"✅ MAE={mae:.1f}")
    return ok, msgs

# ── DIAGNÓSTICO ──────────────────────────────────────────────────────────────

def diagnose_dataset(df, sub="seco"):
    ecol = f"erro_pld_{sub}"
    if ecol not in df.columns:
        c,p = f"cmo_med_{sub}", f"pld_real_{sub}"
        if p in df.columns and c in df.columns: df = df.copy(); df[ecol]=df[p]-df[c]
    y = df[ecol].dropna()
    if y.empty: return {}
    d = {"n":len(y),"mean":round(float(y.mean()),2),"std":round(float(y.std()),2),
         "median":round(float(y.median()),2),"skew":round(float(y.skew()),2),
         "autocorr_lag1":round(float(y.autocorr(1)),3) if len(y)>10 else None}
    z=(y-y.mean())/y.std(); d["outliers_3s"]=int((z.abs()>3).sum())
    num = df.select_dtypes(include=[np.number])
    if ecol in num.columns:
        corrs=num.corrwith(df[ecol]).drop(ecol,errors="ignore").dropna().abs().sort_values(ascending=False)
        d["top_corr"]={k:round(float(v),3) for k,v in corrs.head(15).items()}
        d["n_corr_gt_0.1"]=int((corrs>0.1).sum())
        d["n_corr_gt_0.2"]=int((corrs>0.2).sum())
    d["max_features"]=max(3,len(y)//30)
    # Compare target variants
    variants = {}
    for tgt_name, tgt_col in [
        ("erro_pld (mediana PLD)", f"erro_pld_{sub}"),
        ("erro_pld_mean (média PLD)", f"erro_pld_mean_{sub}"),
        ("erro_cmo (CMO real vs prev)", f"erro_cmo_{sub}"),
    ]:
        if tgt_col in df.columns:
            s = df[tgt_col].dropna()
            if not s.empty:
                variants[tgt_name] = {"std": round(float(s.std()),1),
                    "mean": round(float(s.mean()),1), "autocorr1": round(float(s.autocorr(1)),3) if len(s)>10 else None}
    d["target_variants"] = variants
    return d

# ── PREPARE X,y ──────────────────────────────────────────────────────────────

def _prepare(df, feat_cols, sub, h, target_col=None):
    ecol = target_col or f"erro_pld_{sub}"
    if ecol not in df.columns:
        c,p = f"cmo_med_{sub}", f"pld_real_{sub}_p50"  # MEDIANA, não média
        if p not in df.columns: p = f"pld_real_{sub}"  # fallback
        if p in df.columns and c in df.columns: df=df.copy(); df[ecol]=df[p]-df[c]
        else: return None
    avail=[c for c in feat_cols if c in df.columns]
    if not avail: return None
    y=df[ecol].shift(-h); mask=y.notna()
    X=df[avail][mask].copy().ffill().fillna(0); y=y[mask]
    if len(X)<60: return None
    n=int(len(X)*0.80)
    cmo_col=f"cmo_med_{sub}"
    cmo_te=df[cmo_col].shift(-h).iloc[n:n+len(X)-n].values if cmo_col in df.columns else None
    return X.iloc[:n],X.iloc[n:],y.iloc[:n],y.iloc[n:],avail,cmo_te

def _eval(yt,yp,yp10,yp90,cmo_te=None):
    from sklearn.metrics import r2_score,mean_absolute_error
    m={"r2_test":round(float(r2_score(yt,yp)),4),"mae_erro":round(float(mean_absolute_error(yt,yp)),2)}
    if yp10 is not None: m["coverage_p10"]=round(float(np.mean(yt<=yp10)),3)
    if yp90 is not None: m["coverage_p90"]=round(float(np.mean(yt<=yp90)),3)
    if cmo_te is not None and len(cmo_te)==len(yp):
        pp=cmo_te+yp; pr=cmo_te+yt
        m["mae_pld"]=round(float(np.mean(np.abs(pp-pr))),2)
        dn=np.where(pr==0,np.nan,pr)
        m["mape_pld"]=round(float(np.nanmean(np.abs((pp-pr)/dn))*100),1)
    return m

# ── BASELINES ────────────────────────────────────────────────────────────────

def run_baselines(df, feat_cols, sub="seco", h=4):
    from sklearn.linear_model import Ridge,Lasso,ElasticNet
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import r2_score
    r=_prepare(df,feat_cols,sub,h)
    if r is None: return []
    Xtr,Xte,ytr,yte,cols,cmo_te=r
    results=[]
    # Média
    mp=np.full(len(yte),ytr.mean())
    m=_eval(yte.values,mp,None,None,cmo_te); m["r2_cv_mean"]=0.0; m["model"]="mean"
    m["n_features"]=0; m["score"]=_score(m); results.append(m)

    # AR(1) puro: predição = α × lag1 (o baseline que DEVE ser batido)
    ar_col = f"ar_erro_pld_{sub}_lag1"
    if ar_col in df.columns:
        r_ar = _prepare(df, [ar_col], sub, h)
        if r_ar:
            Xtr_ar,Xte_ar,ytr_ar,yte_ar,_,cmo_te_ar = r_ar
            from sklearn.linear_model import LinearRegression
            lr_ar = LinearRegression()
            lr_ar.fit(Xtr_ar.values, ytr_ar.values)
            yp_ar = lr_ar.predict(Xte_ar.values)
            m_ar = _eval(yte_ar.values, yp_ar, None, None, cmo_te_ar)
            m_ar["r2_cv_mean"] = None; m_ar["model"] = "AR(1)_puro"
            m_ar["n_features"] = 1; m_ar["score"] = _score(m_ar)
            m_ar["coef"] = round(float(lr_ar.coef_[0]), 4)
            results.append(m_ar)
            print(f"    AR(1) puro:  R²={m_ar['r2_test']:+.4f} MAE_PLD={m_ar.get('mae_pld','?')} "
                  f"coef={m_ar['coef']}")

    # AR(1)+MA4: predição = α × lag1 + β × ma4
    ar_ma_cols = [c for c in [f"ar_erro_pld_{sub}_lag1", f"ar_erro_pld_{sub}_ma4"] if c in df.columns]
    if len(ar_ma_cols) == 2:
        r_ar2 = _prepare(df, ar_ma_cols, sub, h)
        if r_ar2:
            Xtr2,Xte2,ytr2,yte2,_,cmo_te2 = r_ar2
            lr2 = LinearRegression()
            lr2.fit(Xtr2.values, ytr2.values)
            yp2 = lr2.predict(Xte2.values)
            m2 = _eval(yte2.values, yp2, None, None, cmo_te2)
            m2["r2_cv_mean"] = None; m2["model"] = "AR(1)+MA4"
            m2["n_features"] = 2; m2["score"] = _score(m2)
            results.append(m2)
            print(f"    AR(1)+MA4:   R²={m2['r2_test']:+.4f} MAE_PLD={m2.get('mae_pld','?')}")
    tscv=TimeSeriesSplit(n_splits=min(5,max(2,len(Xtr)//30)))
    for name,cls,params in [
        ("Ridge_1",Ridge,{"alpha":1.0}),("Ridge_10",Ridge,{"alpha":10.0}),
        ("Ridge_100",Ridge,{"alpha":100.0}),("Ridge_500",Ridge,{"alpha":500.0}),
        ("Lasso_1",Lasso,{"alpha":1.0,"max_iter":5000}),
        ("Lasso_5",Lasso,{"alpha":5.0,"max_iter":5000}),
        ("Lasso_20",Lasso,{"alpha":20.0,"max_iter":5000}),
        ("ElasticNet",ElasticNet,{"alpha":5.0,"l1_ratio":0.5,"max_iter":5000}),
        ("ElasticNet_l1",ElasticNet,{"alpha":5.0,"l1_ratio":0.9,"max_iter":5000}),
    ]:
        pipe=Pipeline([("s",StandardScaler()),("m",cls(**params))])
        pipe.fit(Xtr.values,ytr.values); yp=pipe.predict(Xte.values)
        cvs=[]
        for ti,vi in tscv.split(Xtr):
            pc=Pipeline([("s",StandardScaler()),("m",cls(**params))])
            pc.fit(Xtr.values[ti],ytr.values[ti])
            cvs.append(r2_score(ytr.values[vi],pc.predict(Xtr.values[vi])))
        m=_eval(yte.values,yp,None,None,cmo_te)
        m["r2_cv_mean"]=round(float(np.mean(cvs)),4) if cvs else None
        m["r2_cv_std"]=round(float(np.std(cvs)),4) if cvs else None
        m["model"]=name; m["n_features"]=len(cols); m["score"]=_score(m)
        if hasattr(pipe["m"],"coef_"):
            nz={k:round(abs(v),4) for k,v in zip(cols,pipe["m"].coef_.tolist()) if abs(v)>1e-6}
            m["nonzero_features"]=len(nz)
            m["top_features"]=dict(sorted(nz.items(),key=lambda x:-x[1])[:10])
        results.append(m)
    return results

# ── FEATURE SELECTION ────────────────────────────────────────────────────────

def select_features(df, sub="seco", h=4, max_f=8):
    from sklearn.feature_selection import mutual_info_regression
    ecol=f"erro_pld_{sub}"
    if ecol not in df.columns:
        c,p=f"cmo_med_{sub}",f"pld_real_{sub}"
        if p in df.columns and c in df.columns: df=df.copy(); df[ecol]=df[p]-df[c]
    y=df[ecol].shift(-h).dropna()
    X=df.select_dtypes(include=[np.number]).drop(
        columns=[c for c in df.columns
                 if (c.startswith("erro_pld") or c.startswith("erro_cmo") or c.startswith("pld_real"))
                 and not c.startswith("ar_")],
        errors="ignore")
    # Também excluir AR de OUTROS subsistemas (evitar multicolinearidade r=0.99)
    other_subs = [s for s in ["seco","s","ne","n"] if s != sub]
    X = X.drop(columns=[c for c in X.columns
                         if c.startswith("ar_") and any(f"_{os}_" in c or c.endswith(f"_{os}") for os in other_subs)],
               errors="ignore")
    common=y.index.intersection(X.index); X=X.loc[common].ffill().fillna(0); y=y.loc[common]
    if len(X)<60: return list(X.columns[:max_f])
    # FORÇAR apenas as 2 AR features mais informativas (lag1 + ma4)
    # NÃO forçar todas 6 — são correlacionadas entre si e o GBM overfita.
    # lag1: melhor preditor individual (corr=0.32 com h=4w)
    # ma4: suaviza ruído, captura tendência
    ar_priority = [f"ar_erro_pld_{sub}_lag1", f"ar_erro_pld_{sub}_ma4"]
    forced = [c for c in ar_priority if c in X.columns]
    remaining_slots = max_f - len(forced)
    if remaining_slots <= 0:
        return forced[:max_f]
    # Selecionar features complementares (excluindo as forçadas)
    X_rest = X.drop(columns=forced, errors="ignore")
    if X_rest.empty:
        return forced
    corrs=X_rest.corrwith(y).abs().dropna().sort_values(ascending=False)
    corr_pass=corrs[corrs>0.04].index.tolist()
    if len(corr_pass)<3: corr_pass=corrs.head(remaining_slots*2).index.tolist()
    Xf=X_rest[corr_pass] if corr_pass else X_rest
    try:
        mi=mutual_info_regression(Xf.values,y.values,random_state=42,n_neighbors=5)
        mi_pass=pd.Series(mi,index=Xf.columns).sort_values(ascending=False).head(remaining_slots*2).index.tolist()
    except: mi_pass=corr_pass[:remaining_slots*2]
    cm=X_rest[mi_pass].corr().abs() if mi_pass else pd.DataFrame()
    extra=[]
    for col in mi_pass:
        too_corr=any(cm.loc[col,s]>0.85 for s in extra if s in cm.columns and col in cm.index)
        if not too_corr: extra.append(col)
        if len(extra)>=remaining_slots: break
    return forced + extra

# ── GBM LEVE ─────────────────────────────────────────────────────────────────

def train_light_gbm(df, feat_cols, config, sub="seco", h=4):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import r2_score
    r=_prepare(df,feat_cols,sub,h)
    if r is None: return None
    Xtr,Xte,ytr,yte,cols,cmo_te=r
    hp=config.get("hyperparams",{})
    tscv=TimeSeriesSplit(n_splits=min(5,max(2,len(Xtr)//30)))
    models={}; cvs=[]
    for q in QUANTILES:
        gb=HistGradientBoostingRegressor(loss="quantile",quantile=q,
            max_iter=hp.get("max_iter",200),max_depth=hp.get("max_depth",2),
            learning_rate=hp.get("learning_rate",0.05),
            min_samples_leaf=hp.get("min_samples_leaf",15),
            l2_regularization=hp.get("l2_regularization",1.0),
            max_features=hp.get("max_features",0.8),random_state=42)
        gb.fit(Xtr.values,ytr.values)
        if q==0.50:
            for ti,vi in tscv.split(Xtr):
                gc=HistGradientBoostingRegressor(loss="quantile",quantile=0.50,
                    max_iter=min(150,hp.get("max_iter",200)),max_depth=hp.get("max_depth",2),
                    learning_rate=hp.get("learning_rate",0.05),
                    min_samples_leaf=hp.get("min_samples_leaf",15),
                    l2_regularization=hp.get("l2_regularization",1.0),
                    max_features=hp.get("max_features",0.8),random_state=42)
                gc.fit(Xtr.values[ti],ytr.values[ti])
                cvs.append(r2_score(ytr.values[vi],gc.predict(Xtr.values[vi])))
        models[q]=gb
    yp50=models[0.50].predict(Xte.values)
    yp10=models[0.10].predict(Xte.values)
    yp90=models[0.90].predict(Xte.values)
    m=_eval(yte.values,yp50,yp10,yp90,cmo_te)
    m["r2_cv_mean"]=round(float(np.mean(cvs)),4) if cvs else None
    m["r2_cv_std"]=round(float(np.std(cvs)),4) if cvs else None
    m["coverage_p10"]=round(float(np.mean(yte.values<=yp10)),3)
    m["coverage_p90"]=round(float(np.mean(yte.values<=yp90)),3)
    m["n_features"]=len(cols); m["score"]=_score(m)
    fi={}
    if hasattr(models[0.50],"feature_importances_"):
        fi=dict(sorted(zip(cols,models[0.50].feature_importances_.tolist()),key=lambda x:-x[1]))
    return {"models":models,"feature_cols":cols,"metrics":m,"feature_importance":fi,
            "config":config,"n_train":len(Xtr),"n_test":len(Xte),"subsistema":sub,
            "horizon_weeks":h,"target":"erro_pld","trained_at":datetime.now().isoformat()}

def prepare_winsorized(df, sub="seco"):
    df=df.copy()
    p,c=f"pld_real_{sub}",f"cmo_med_{sub}"
    if p not in df.columns or c not in df.columns: return df
    e=df[p]-df[c]; p5,p95=e.quantile(0.05),e.quantile(0.95)
    df[f"erro_pld_{sub}"]=e.clip(p5,p95)
    return df

def _gbm_configs():
    cfgs=[]
    # ── Grupo A: STUMP depth=1 (quase linear — mínimo overfitting) ──────
    for lr in [0.02,0.03,0.05,0.08,0.12]:
        cfgs.append({"name":f"stump_lr{lr}","hyperparams":{"max_iter":150,"max_depth":1,
            "learning_rate":lr,"min_samples_leaf":20,"l2_regularization":2.0,"max_features":1.0}})
    for l2 in [0.5,1.0,2.0,5.0,10.0]:
        cfgs.append({"name":f"stump_l2{l2}","hyperparams":{"max_iter":200,"max_depth":1,
            "learning_rate":0.05,"min_samples_leaf":20,"l2_regularization":l2,"max_features":1.0}})
    # ── Grupo B: shallow depth=2 (leve não-linearidade) ─────────────────
    for lr in [0.03,0.05,0.08]:
        cfgs.append({"name":f"shallow_lr{lr}","hyperparams":{"max_iter":100,"max_depth":2,
            "learning_rate":lr,"min_samples_leaf":20,"l2_regularization":2.0,"max_features":0.8}})
    for l2 in [1.0,2.0,4.0]:
        cfgs.append({"name":f"d2_l2{l2}","hyperparams":{"max_iter":200,"max_depth":2,
            "learning_rate":0.05,"min_samples_leaf":15,"l2_regularization":l2,"max_features":0.8}})
    # ── Grupo C: slow shrinkage (depth=1, muitas iterações pequenas) ────
    cfgs.append({"name":"slow_stump","hyperparams":{"max_iter":500,"max_depth":1,"learning_rate":0.01,
        "min_samples_leaf":25,"l2_regularization":5.0,"max_features":1.0}})
    cfgs.append({"name":"slow_d2","hyperparams":{"max_iter":400,"max_depth":2,"learning_rate":0.01,
        "min_samples_leaf":20,"l2_regularization":3.0,"max_features":0.8}})
    return cfgs

# ── MAIN ─────────────────────────────────────────────────────────────────────

def run_optimization(max_iter=80, focus_sub="seco", focus_h=4, verbose=True):
    from meta_forecast_engine import (load_pmo_features,load_actuals,
        load_maatria_weekly_features,build_enhanced_dataset,save_meta_models,
        ENHANCED_FEATURE_COLS,BASE_FEATURE_COLS)
    targets=DEFAULT_TARGETS
    print("\n"+"="*70)
    print("  MAÁTria Energia · Auto-Tune v2 (progressivo)")
    print("="*70)
    print(f"  Metas: R²≥{targets['r2_test_min']} | MAE≤R${targets['mae_pld_max']} | max {max_iter} iter")
    print(f"  Foco: {SUB_LABEL[focus_sub]} h={focus_h}w")
    print("="*70)
    t0=time.time()
    print("\n[CARGA] Carregando dados...")
    pmo=load_pmo_features(PMO_XLSX); act=load_actuals(DATA_DIR)
    maat=load_maatria_weekly_features(DATA_DIR)
    df=build_enhanced_dataset(pmo,act,maat)
    if df.empty: print("ERRO: dataset vazio"); return {}
    print(f"    {len(df)} semanas × {df.shape[1]} colunas ({time.time()-t0:.1f}s)")
    history=[]; best_score=-999; best_bundle=None; best_name=""
    def _rec(name,m,bun=None):
        nonlocal best_score,best_bundle,best_name
        history.append({"name":name,"score":m["score"],"metrics":m})
        imp=m["score"]>best_score
        if imp: best_score=m["score"]; best_bundle=bun; best_name=name
        return imp

    # FASE 0
    print(f"\n{'='*70}\n  FASE 0 — Diagnóstico\n{'='*70}")
    diag=diagnose_dataset(df,focus_sub)
    print(f"    Target: média={diag.get('mean')}, std={diag.get('std')}, mediana={diag.get('median')}")
    print(f"    Skew={diag.get('skew')}, Autocorr lag1={diag.get('autocorr_lag1')}")
    print(f"    Outliers 3σ: {diag.get('outliers_3s')} | Corr>0.1: {diag.get('n_corr_gt_0.1',0)} | Corr>0.2: {diag.get('n_corr_gt_0.2',0)}")
    tv=diag.get("target_variants",{})
    if tv:
        print(f"\n    Comparação de targets:")
        for tname, tvals in tv.items():
            print(f"      {tname:<35} std={tvals['std']:>6} mean={tvals['mean']:>7} autocorr1={tvals.get('autocorr1','?')}")
    tc=diag.get("top_corr",{})
    if tc:
        print(f"    Top 5 correlações:")
        for k,v in list(tc.items())[:5]: print(f"      {k:<40} r={v:.3f}")
    mf=diag.get("max_features",8)
    print(f"    → Max features: {mf}")

    # FASE 1
    print(f"\n{'='*70}\n  FASE 1 — Baselines\n{'='*70}")
    all_feats=[c for c in ENHANCED_FEATURE_COLS if c in df.columns]
    bl=run_baselines(df,all_feats,focus_sub,focus_h)
    for b in bl:
        imp=_rec(f"bl_{b['model']}",b)
        mk=" ★" if imp else ""
        print(f"    {b['model']:<25} R²={b['r2_test']:+.4f} MAE_PLD={str(b.get('mae_pld','?')):>6} "
              f"R²cv={str(b.get('r2_cv_mean','?')):>7} score={b['score']:5.1f}{mk}")
    bb=max(bl,key=lambda x:x["score"]) if bl else {"score":0,"model":"?"}
    print(f"\n    → Melhor baseline: {bb.get('model')} score={bb['score']:.1f}")

    # Testar targets alternativos com melhor baseline
    print(f"\n    Testando targets alternativos (CMO erro, PLD médio):")
    for tgt_label, tgt_col in [("erro_cmo", f"erro_cmo_{focus_sub}"),
                                ("erro_pld_mean", f"erro_pld_mean_{focus_sub}")]:
        if tgt_col not in df.columns: continue
        bl_alt = run_baselines(df, all_feats, focus_sub, focus_h)
        # Override: usar target alternativo
        r_alt = _prepare(df, all_feats, focus_sub, focus_h, target_col=tgt_col)
        if r_alt:
            from sklearn.linear_model import Ridge; from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline; from sklearn.metrics import r2_score
            Xtr,Xte,ytr,yte,cols,cmo_te = r_alt
            pipe=Pipeline([("s",StandardScaler()),("m",Ridge(alpha=10.0))])
            pipe.fit(Xtr.values,ytr.values); yp=pipe.predict(Xte.values)
            r2=round(float(r2_score(yte.values,yp)),4)
            mae=round(float(np.mean(np.abs(yp-yte.values))),2)
            print(f"      {tgt_label:<20} Ridge R²={r2:+.4f} MAE_erro={mae:.1f}")

    # FASE 2
    print(f"\n{'='*70}\n  FASE 2 — Feature selection (top {mf})\n{'='*70}")
    sel=select_features(df,focus_sub,focus_h,mf)
    # Calcular correlações REAIS com target shifted (não do diagnóstico unshifted)
    ecol=f"erro_pld_{focus_sub}"
    y_shifted = df[ecol].shift(-focus_h).dropna() if ecol in df.columns else pd.Series(dtype=float)
    shifted_corrs = {}
    if not y_shifted.empty:
        for fn in sel:
            if fn in df.columns:
                common = y_shifted.index.intersection(df[fn].dropna().index)
                if len(common) > 30:
                    shifted_corrs[fn] = round(float(df[fn].loc[common].corr(y_shifted.loc[common])), 3)
    forced_count = sum(1 for f in sel if f.startswith("ar_"))
    print(f"    Selecionadas ({len(sel)}): {forced_count} forçadas (AR) + {len(sel)-forced_count} data-driven")
    for i,fn in enumerate(sel):
        c = shifted_corrs.get(fn, 0)
        tag = " [AR forçada]" if fn.startswith("ar_") else ""
        print(f"      {i+1}. {fn:<40} corr_shifted={c:+.3f}{tag}")
    print(f"\n    Re-testando baselines com features selecionadas:")
    bl2=run_baselines(df,sel,focus_sub,focus_h)
    for b in bl2:
        imp=_rec(f"sel_{b['model']}",b)
        mk=" ★" if imp else ""
        print(f"    {b['model']:<25} R²={b['r2_test']:+.4f} MAE={str(b.get('mae_pld','?')):>6} "
              f"score={b['score']:5.1f}{mk}")
    # Winsorized baselines
    dfw=prepare_winsorized(df,focus_sub)
    print(f"\n    Target winsorized:")
    bl3=run_baselines(dfw,sel,focus_sub,focus_h)
    for b in bl3:
        imp=_rec(f"wins_{b['model']}",b)
        mk=" ★" if imp else ""
        print(f"    {b['model']:<25} R²={b['r2_test']:+.4f} MAE={str(b.get('mae_pld','?')):>6} "
              f"score={b['score']:5.1f}{mk}")

    # FASE 3
    print(f"\n{'='*70}\n  FASE 3 — GBM leve (depth≤3)\n{'='*70}")
    gcfgs=_gbm_configs(); ic=len(history)
    for cfg in gcfgs:
        if ic>=max_iter: break; ic+=1
        bun=train_light_gbm(df,sel,cfg,focus_sub,focus_h)
        if bun is None: continue
        m=bun["metrics"]; imp=_rec(f"gbm_{cfg['name']}",m,bun)
        mk=" ★" if imp else ""
        print(f"    [{ic:3d}] {cfg['name']:<25} R²={m['r2_test']:+.4f} MAE={str(m.get('mae_pld','?')):>6} "
              f"R²cv={str(m.get('r2_cv_mean','?')):>7} P10={m.get('coverage_p10','?')} "
              f"P90={m.get('coverage_p90','?')} score={m['score']:5.1f}{mk}")
    # GBM + winsorized
    print(f"\n    GBM + target winsorized:")
    for cfg in gcfgs[:8]:
        if ic>=max_iter: break; ic+=1
        bun=train_light_gbm(dfw,sel,cfg,focus_sub,focus_h)
        if bun is None: continue
        m=bun["metrics"]; imp=_rec(f"wgbm_{cfg['name']}",m,bun)
        mk=" ★" if imp else ""
        print(f"    [{ic:3d}] w_{cfg['name']:<22} R²={m['r2_test']:+.4f} MAE={str(m.get('mae_pld','?')):>6} "
              f"score={m['score']:5.1f}{mk}")

    # FASE 4 — refinamento
    if best_bundle and isinstance(best_bundle,dict) and "config" in best_bundle:
        print(f"\n{'='*70}\n  FASE 4 — Refinamento de '{best_name}'\n{'='*70}")
        bhp=best_bundle["config"]["hyperparams"]
        for pn,delta in [("lr×0.7",{"learning_rate":bhp["learning_rate"]*0.7}),
            ("lr×1.3",{"learning_rate":bhp["learning_rate"]*1.3}),
            ("l2×0.5",{"l2_regularization":bhp["l2_regularization"]*0.5}),
            ("l2×2",{"l2_regularization":bhp["l2_regularization"]*2}),
            ("leaf-5",{"min_samples_leaf":max(5,bhp["min_samples_leaf"]-5)}),
            ("leaf+5",{"min_samples_leaf":bhp["min_samples_leaf"]+5}),
            ("iter+100",{"max_iter":bhp["max_iter"]+100}),("mf0.6",{"max_features":0.6})]:
            if ic>=max_iter: break; ic+=1
            hp={**bhp,**delta}; cfg={"name":f"ref_{pn}","hyperparams":hp}
            udf=dfw if "wins_" in best_name or "wgbm_" in best_name else df
            bun=train_light_gbm(udf,sel,cfg,focus_sub,focus_h)
            if bun is None: continue
            m=bun["metrics"]; imp=_rec(f"ref_{pn}",m,bun)
            mk=" ★" if imp else ""
            print(f"    [{ic:3d}] {pn:<25} R²={m['r2_test']:+.4f} MAE={str(m.get('mae_pld','?')):>6} "
                  f"score={m['score']:5.1f}{mk}")

    # RESULTADO
    tt=time.time()-t0
    print(f"\n{'='*70}\n  RESULTADO — {ic} iterações em {tt:.0f}s\n{'='*70}")
    if best_bundle and isinstance(best_bundle,dict) and "models" in best_bundle:
        bm=best_bundle["metrics"]
        print(f"  Melhor:       {best_name}")
        print(f"  Score:        {best_score:.1f}/100")
        print(f"  R² teste:     {bm.get('r2_test','?')}")
        print(f"  R² CV:        {bm.get('r2_cv_mean','?')}")
        print(f"  MAE erro:     R${bm.get('mae_erro','?')}")
        print(f"  MAE PLD:      R${bm.get('mae_pld','?')}")
        print(f"  Coverage:     P10={bm.get('coverage_p10','?')} P90={bm.get('coverage_p90','?')}")
        print(f"  N features:   {bm.get('n_features','?')}")
        fi=best_bundle.get("feature_importance",{})
        if fi:
            print(f"\n  Top features:")
            for i,(ft,im) in enumerate(list(fi.items())[:10]):
                print(f"    {i+1:2d}. {ft:<40} {im:.4f} {'█'*max(1,int(im*100))}")
        print(f"\n  Salvando modelos finais...")
        cfg=best_bundle.get("config",_gbm_configs()[0])
        udf=dfw if "wins_" in best_name or "wgbm_" in best_name else df
        final={}
        for sub in SUBSISTEMAS:
            final[sub]={}
            for hw in [4,8,12,26]:
                s=select_features(udf,sub,hw,mf)
                b=train_light_gbm(udf,s,cfg,sub,hw)
                if b:
                    final[sub][hw]=b; m2=b["metrics"]
                    print(f"    {SUB_LABEL[sub]:>10} h={hw:2d}w: R²={m2['r2_test']:+.4f} "
                          f"MAE={str(m2.get('mae_pld','?')):>6} score={m2['score']:5.1f}")
        save_meta_models(final,META_DIR)
        with open(META_DIR/"best_config.json","w") as f:
            json.dump({"name":best_name,"score":best_score,"metrics":bm,
                "config":cfg,"selected_features":sel,"diagnosis":diag,
                "iterations":ic,"total_time":round(tt,1),
                "timestamp":datetime.now().isoformat()},f,indent=2,default=str)
    else:
        print(f"  Melhor geral: {best_name} score={best_score:.1f} (sem modelo GBM)")
    with open(META_DIR/"tuning_log.json","w") as f:
        json.dump(history,f,indent=2,default=str)
    met,msgs=_targets_met((best_bundle or {}).get("metrics",{"r2_test":-1,"mae_pld":999}))
    print(f"\n  {'✅ METAS' if met else '⚠️  Pendentes'}:")
    for mg in msgs: print(f"    {mg}")
    print(f"\n{'='*70}\n")
    return {"best":best_name,"score":best_score,"iter":ic,"time":round(tt,1),"met":met,"diag":diag}

def show_report():
    p=META_DIR/"best_config.json"
    if not p.exists(): print("Execute: python auto_tune_forecast.py"); return
    with open(p) as f: d=json.load(f)
    print(f"\n{'='*70}\n  Relatório Auto-Tune v2\n{'='*70}")
    for k in ["name","score","iterations","total_time"]: print(f"  {k}: {d.get(k)}")
    m=d.get("metrics",{})
    for k in ["r2_test","r2_cv_mean","mae_erro","mae_pld","coverage_p10","coverage_p90"]:
        print(f"  {k}: {m.get(k)}")
    sf=d.get("selected_features",[])
    if sf: print(f"\n  Features ({len(sf)}):"); [print(f"    - {f}") for f in sf]
    print(f"{'='*70}\n")

if __name__=="__main__":
    ap=argparse.ArgumentParser(description="MAÁTria · Auto-Tune v2")
    ap.add_argument("--max-iter",type=int,default=80)
    ap.add_argument("--focus-sub",default="seco",choices=SUBSISTEMAS)
    ap.add_argument("--focus-h",type=int,default=4)
    ap.add_argument("--report",action="store_true")
    ap.add_argument("--xlsx",default=str(PMO_XLSX))
    ap.add_argument("--data",default=str(DATA_DIR))
    args=ap.parse_args()
    PMO_XLSX=Path(args.xlsx); DATA_DIR=Path(args.data)
    if args.report: show_report()
    else: run_optimization(args.max_iter,args.focus_sub,args.focus_h)
