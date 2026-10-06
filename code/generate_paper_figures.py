"""Generate the complete RouteCast manuscript figure package.

All quantitative panels are sourced from recorded JSON/CSV artifacts. Legacy
Qwen-only development results are confined to Extended Data figures.
"""
from __future__ import annotations
import csv, json, math, os, sys
from collections import defaultdict
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_ROOT.parent
PKG = PROJECT_ROOT / "python_packages" / "figure"
sys.path.insert(0, str(PKG))
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = PROJECT_ROOT / "record"
LEGACY_REC = ROOT / "legacy_qwen"
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family":"sans-serif", "font.sans-serif":["Arial","DejaVu Sans"],
    "font.size":7, "axes.titlesize":8, "axes.labelsize":7,
    "xtick.labelsize":6.5, "ytick.labelsize":6.5, "legend.fontsize":6.5,
    "axes.spines.top":False, "axes.spines.right":False,
    "axes.linewidth":0.7, "lines.linewidth":1.5,
    "svg.fonttype":"none", "pdf.fonttype":42,
    "savefig.facecolor":"white", "figure.facecolor":"white",
})

COL = {"Paper Heatmap":"#A7B0B8", "Request-aware Heatmap":"#7D8C99",
       "GRU-only":"#7FA9D1", "RouteCast V3":"#D55E5E",
       "qwen3":"#3B78B5", "deepseek_r1":"#D98242", "llama4_maverick":"#4F9D69",
       "LRU":"#9AA1A7", "fixed_native":"#6C8EBF", "adaptive_mass":"#D6A34A",
       "adaptive_cache":"#C94C4C"}
DISPLAY={"qwen3":"Qwen3", "deepseek_r1":"DeepSeek-R1", "llama4_maverick":"Llama4 Maverick"}
MODELS=list(DISPLAY)

def load_json(p):
    with Path(p).open(encoding="utf-8-sig") as f:return json.load(f)
def load_csv(p):
    with Path(p).open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def panel(ax,label): ax.text(-.10,1.14,label,transform=ax.transAxes,fontweight="bold",fontsize=9,va="top")
def save(fig,name):
    fig.subplots_adjust(left=.08,right=.98,bottom=.14,top=.91,wspace=.35,hspace=.42)
    # Keep explicit calls so publication preflight can verify every export.
    fig.savefig(OUT/f"{name}.svg", bbox_inches="tight")
    fig.savefig(OUT/f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT/f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT/f"{name}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig); print(f"saved={name}",flush=True)
def nice(ax):
    ax.grid(axis="y",color="#E7E9EB",lw=.6,zorder=0); ax.set_axisbelow(True)

def fig1_framework():
    fig,ax=plt.subplots(figsize=(7.2,3.25)); ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis("off")
    stages=[(.02,.31,.15,.38,"Route traces","Recent token routes\nRequest context"),
            (.21,.21,.22,.58,"Multi-source forecasting","Cross-token history\nTransition prior\nPopularity prior"),
            (.48,.27,.17,.46,"Context gate","Sample-dependent\nbranch weighting"),
            (.70,.27,.13,.46,"Calibration","Temperature-scaled\nexpert probabilities"),
            (.87,.17,.11,.66,"Selective scheduling","Mass / dynamic K\nConfidence reject\nCache admission")]
    colors=["#E8EEF4","#DCEAF7","#F4E6D1","#E7E1F2","#F5DADA"]
    for i,(x,y,w,h,title,body) in enumerate(stages):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.012,rounding_size=.018",fc=colors[i],ec="#555",lw=.8))
        ax.text(x+w/2,y+h*.70,title,ha="center",va="center",fontweight="bold",fontsize=7.5)
        ax.text(x+w/2,y+h*.40,body,ha="center",va="center",fontsize=6.5,linespacing=1.45)
        if i<len(stages)-1:
            nx=stages[i+1][0]; ax.add_patch(FancyArrowPatch((x+w+.008,.5),(nx-.008,.5),arrowstyle="-|>",mutation_scale=9,lw=1,color="#555"))
    ax.text(.5,.93,"RouteCast: calibrated cross-token forecasting for selective expert prefetch",ha="center",fontweight="bold",fontsize=10)
    ax.text(.5,.07,"Forecast expert utility first; issue prefetches only when probability mass and cache value justify the transfer.",ha="center",fontsize=7,color="#444")
    save(fig,"Fig1_routecast_framework")

def fig2_main():
    d=load_json(ROOT/"paper/results/main_limit1000/main_forecasting_limit1000.json")["rows"]
    boot=load_json(ROOT/"paper/results/main_limit1000/bootstrap_routecast_vs_gru_limit1000.json")["models"]
    fig,axs=plt.subplots(1,4,figsize=(7.2,3.00),gridspec_kw={"width_ratios":[1,1,1,1.15]})
    methods=["Paper Heatmap","Request-aware Heatmap","GRU-only","RouteCast V3"]
    for j,m in enumerate(MODELS):
        ax=axs[j]; rows={r["method"]:r for r in d if r["model"]==m}; vals=[rows[x]["recall"] for x in methods]
        ax.bar(range(4),vals,color=[COL[x] for x in methods],width=.72,zorder=3)
        ax.set_xticks(range(4),["Paper","Req.-aware","GRU","RouteCast"],rotation=35,ha="right",rotation_mode="anchor")
        ax.set_ylim(0,max(vals)*1.23); ax.set_ylabel(f"Recall@{rows['RouteCast V3']['native_k']}") if j==0 else None
        ax.set_title(DISPLAY[m]); nice(ax); panel(ax,chr(97+j))
        ax.text(3,vals[3]+.02,f"{vals[3]:.3f}",ha="center",fontweight="bold",color=COL["RouteCast V3"])
    ax=axs[3]; ys=np.arange(3); delta=[]; lo=[]; hi=[]
    for m in MODELS:
        z=boot[m]["metrics"]["recall"];delta.append(z["delta"]);lo.append(z["delta"]-z["ci95"][0]);hi.append(z["ci95"][1]-z["delta"])
    ax.errorbar(delta,ys,xerr=[lo,hi],fmt="o",color=COL["RouteCast V3"],ecolor="#555",capsize=2.5,lw=1.1)
    ax.axvline(0,color="#777",lw=.8);ax.set_yticks(ys,["Qwen3","DeepSeek","Llama4"]);ax.tick_params(axis="y",labelsize=5.5,pad=2);ax.invert_yaxis();ax.set_xlabel("Recall gain vs GRU-only\n(95% paired bootstrap CI)");ax.grid(axis="x",color="#E7E9EB");panel(ax,"d")
    save(fig,"Fig2_main_prediction")

def fig3_budget():
    d=load_json(ROOT/"budget_curve_limit100_v2.json")["results"]
    fig,axs=plt.subplots(1,3,figsize=(7.2,2.90))
    for i,m in enumerate(MODELS):
        ax=axs[i]; rows=d[m]; k=[x["prefetch_k"] for x in rows]; rec=[x["recall"] for x in rows]; waste=[x["wasted_prefetch_ratio"] for x in rows]
        ax.plot(k,rec,"o-",color=COL[m],label="Recall");ax.plot(k,waste,"s--",color="#7E8790",label="Waste")
        ax.axvline(rows[0]["native_topk"],color="#222",ls=":",lw=.9,label="Native K")
        ax.set_title(DISPLAY[m]);ax.set_xlabel("Prefetched experts (K)");ax.set_ylim(0,1);nice(ax);panel(ax,chr(97+i))
        if i==0:ax.set_ylabel("Rate")
        if i==2:ax.legend(loc="lower right")
    save(fig,"Fig3_budget_quality_tradeoff")

def fig4_ablation():
    rows=load_json(ROOT/"ablations/cross_token_limit100/cross_token_ablation_limit100.json")["rows"]
    variants=["no_history","no_two_hop","no_prefill"]
    mat=np.array([[next(r["delta_vs_full"] for r in rows if r["model"]==m and r["variant"]==v) for v in variants] for m in MODELS])
    fig,ax=plt.subplots(figsize=(5.8,2.55)); im=ax.imshow(mat,cmap="RdBu_r",vmin=-max(abs(mat.min()),abs(mat.max())),vmax=max(abs(mat.min()),abs(mat.max())),aspect="auto")
    ax.set_xticks(range(3),["Remove history","Remove two-hop","Remove prefill"]);ax.set_yticks(range(3),[DISPLAY[m] for m in MODELS])
    for i in range(3):
        for j in range(3):ax.text(j,i,f"{mat[i,j]:+.3f}",ha="center",va="center",color="white" if abs(mat[i,j])>.012 else "#222",fontweight="bold")
    cb=fig.colorbar(im,ax=ax,pad=.025);cb.set_label("Recall change vs full model")
    ax.set_title("Component effects vary across MoE routing regimes");panel(ax,"a")
    save(fig,"Fig4_cross_model_ablation")

def fig5_calibration_mass():
    d=load_json(ROOT/"calibration/routecast_v3_limit100_calibration.json")
    cache=load_json(ROOT/"adaptive_cache/routecast_v3_limit100_adaptive_cache.json")["models"]
    fig,axs=plt.subplots(2,3,figsize=(7.2,4.75))
    for i,m in enumerate(MODELS):
        ax=axs[0,i]; z=d["models"][m]
        for key,lab,ls in [("test_before","Before","--"),("test_after","After","-")]:
            rel=z[key]["reliability"]; x=[r["confidence"] for r in rel if r["count"]>0];y=[r["accuracy"] for r in rel if r["count"]>0];ax.plot(x,y,"o"+ls,ms=2.5,label=lab,color="#888" if key=="test_before" else COL[m])
        ax.plot([0,1],[0,1],":",color="#333",lw=.8);ax.set_xlim(0,1);ax.set_ylim(0,1);ax.set_title(f"{DISPLAY[m]}  T={z['temperature']:.2f}");ax.set_xlabel("Confidence");panel(ax,chr(97+i));nice(ax)
        if i==0:ax.set_ylabel("Empirical accuracy")
        if i==2:ax.legend(loc="upper left")
        ax=axs[1,i]; curve=cache[m]["val_curve"]; mass=sorted(float(x) for x in curve); rec=[curve[str(x)]["recall"] for x in mass]; k=[curve[str(x)]["avg_k"] for x in mass]
        ax.plot(k,rec,"o-",color=COL[m],ms=3); chosen=cache[m]["selected_mass"]; q=curve[str(chosen)]; ax.scatter([q["avg_k"]],[q["recall"]],s=38,color="#C94C4C",zorder=4,label=f"selected m={chosen:g}")
        ax.set_xlabel("Average prefetched experts");ax.set_ylabel("Validation recall" if i==0 else "");ax.legend(loc="lower right");nice(ax);panel(ax,chr(100+i))
    save(fig,"Fig5_calibration_and_mass")

def fig6_cache():
    rows=load_csv(ROOT/"adaptive_cache/routecast_v3_limit100_adaptive_cache.csv"); policies=["LRU","fixed_native","adaptive_mass","adaptive_cache"]
    fig,axs=plt.subplots(2,3,figsize=(7.2,4.55),sharex="col")
    for j,m in enumerate(MODELS):
        rr=[r for r in rows if r["model"]==m]
        for p in policies:
            x=sorted(int(r["capacity"]) for r in rr if r["policy"]==p); hit=[float(next(r["hit_rate"] for r in rr if r["policy"]==p and int(r["capacity"])==q)) for q in x]
            tr=[float(next(r["total_transfers"] for r in rr if r["policy"]==p and int(r["capacity"])==q)) for q in x]
            axs[0,j].plot(x,hit,"o-",ms=3,color=COL[p],label=p.replace("_"," "))
            axs[1,j].plot(x,np.array(tr)/1e6,"o-",ms=3,color=COL[p])
        axs[0,j].set_title(DISPLAY[m]);axs[1,j].set_xlabel("Cache capacity (× native working set)");axs[1,j].set_xticks([1,2,4]);nice(axs[0,j]);nice(axs[1,j]);panel(axs[0,j],chr(97+j));panel(axs[1,j],chr(100+j))
    axs[0,0].set_ylabel("Cache hit rate");axs[1,0].set_ylabel("Total transfers (million)");axs[0,2].legend(loc="lower right",fontsize=5.5)
    save(fig,"Fig6_cache_aware_replay")

def ext1_dataset():
    m=load_json(PROJECT_ROOT/"data/routecast_cache/cross_token_limit1000_h16_packed/manifest.json"); counts=defaultdict(lambda:defaultdict(int)); req=defaultdict(lambda:defaultdict(int))
    for x in m["items"]: counts[x["model"]][x["split"]]+=x["samples"];req[x["model"]][x["split"]]+=1
    fig,axs=plt.subplots(1,2,figsize=(6.4,2.55)); splits=["train","val","test"]
    for i,s in enumerate(splits):
        axs[0].bar(np.arange(3)+(i-1)*.23,[counts[x][s]/1e6 for x in MODELS],width=.23,label=s,color=["#91AEC4","#D7B377","#B67A7A"][i])
        axs[1].bar(np.arange(3)+(i-1)*.23,[req[x][s] for x in MODELS],width=.23,label=s,color=["#91AEC4","#D7B377","#B67A7A"][i])
    for i,ax in enumerate(axs):ax.set_xticks(range(3),[DISPLAY[x] for x in MODELS],rotation=15,ha="right",rotation_mode="anchor");nice(ax);panel(ax,chr(97+i))
    axs[0].set_ylabel("Token-layer samples (million)");axs[1].set_ylabel("Requests");axs[1].legend()
    save(fig,"ExtFig1_dataset_composition")

def ext2_history():
    rows=load_csv(LEGACY_REC/"history_sweep_limit100_summary.csv");x=[int(r["history"]) for r in rows];y=[float(r["test_recall@8"]) for r in rows]
    fig,ax=plt.subplots(figsize=(4.6,2.6));ax.plot(x,y,"o-",color="#3B78B5",ms=3);best=np.argmax(y);ax.scatter([x[best]],[y[best]],s=45,color="#C94C4C",zorder=4);ax.annotate(f"best H={x[best]}",(x[best],y[best]),xytext=(8,8),textcoords="offset points");ax.set_xlabel("History length");ax.set_ylabel("Recall@8");ax.set_title("Development-stage Qwen-only history sweep (limit=100)");nice(ax);panel(ax,"a");save(fig,"ExtFig2_history_sensitivity_legacy")

def ext3_scale():
    small=load_json(ROOT/"ablations/cross_token_limit100/cross_token_ablation_limit100.json")["rows"]; large=load_json(ROOT/"paper/results/main_limit1000/main_forecasting_limit1000.json")["rows"]
    y100=[next(r["recall"] for r in small if r["model"]==m and r["variant"]=="full") for m in MODELS];y1000=[next(r["recall"] for r in large if r["model"]==m and r["method"]=="RouteCast V3") for m in MODELS]
    fig,ax=plt.subplots(figsize=(4.8,2.7));x=np.arange(3);ax.bar(x-.18,y100,.36,label="Limit=100",color="#9DBAD2");ax.bar(x+.18,y1000,.36,label="Limit=1000",color="#C94C4C");ax.set_xticks(x,[DISPLAY[m] for m in MODELS]);ax.set_ylabel("Native-K recall");ax.legend();nice(ax);panel(ax,"a");save(fig,"ExtFig3_data_scale_generalization")

def ext4_seeds():
    d=load_json(ROOT/"seed_repro/cross_token_limit100/cross_token_seed_summary.json");fig,ax=plt.subplots(figsize=(4.8,2.7))
    vals=[];err=[]
    for m in MODELS:
        z=d["models"][m];vals.append(z["mean_recall"]);err.append(z["sample_std"])
    ax.errorbar(range(3),vals,yerr=err,fmt="o",capsize=4,color="#C94C4C");ax.set_xticks(range(3),[DISPLAY[m] for m in MODELS]);ax.set_ylabel("Native-K recall");ax.set_title("Three-seed reproducibility (limit=100)");nice(ax);panel(ax,"a");save(fig,"ExtFig4_seed_reproducibility")

def ext5_layer():
    rows=load_csv(LEGACY_REC/"layerwise_gated_limit1000.csv");x=[int(r["layer"]) for r in rows];y=[float(r["recall@8"]) for r in rows]
    fig,ax=plt.subplots(figsize=(6.2,2.55));ax.plot(x,y,color="#3B78B5");lo=int(np.argmin(y));hi=int(np.argmax(y));ax.scatter([x[lo],x[hi]],[y[lo],y[hi]],color=["#C94C4C","#4F9D69"],zorder=3);ax.annotate(f"worst L{x[lo]}",(x[lo],y[lo]),xytext=(5,-13),textcoords="offset points");ax.annotate(f"best L{x[hi]}",(x[hi],y[hi]),xytext=(5,7),textcoords="offset points");ax.set_xlabel("MoE layer");ax.set_ylabel("Recall@8");ax.set_title("Development-stage Qwen-only layer heterogeneity");nice(ax);panel(ax,"a");save(fig,"ExtFig5_layerwise_heterogeneity_legacy")

def ext6_calibration_metrics():
    d=load_json(ROOT/"calibration/routecast_v3_limit100_calibration.json")["models"];metrics=["bce","brier","ece"]
    fig,axs=plt.subplots(1,3,figsize=(6.7,2.35))
    for j,metric in enumerate(metrics):
        before=[d[m]["test_before"][metric] for m in MODELS];after=[d[m]["test_after"][metric] for m in MODELS];x=np.arange(3)
        axs[j].bar(x-.18,before,.36,label="Before",color="#A7B0B8");axs[j].bar(x+.18,after,.36,label="After",color="#C94C4C");axs[j].set_xticks(x,["Qwen3","DeepSeek","Llama4"],rotation=18,ha="right",rotation_mode="anchor");axs[j].set_title(metric.upper());nice(axs[j]);panel(axs[j],chr(97+j))
    axs[2].legend();save(fig,"ExtFig6_calibration_metrics")

def ext7_overhead():
    d=load_json(LEGACY_REC/"predictor_overhead_limit1000.json");labs=["Parameters\n(thousand)","Model size\n(0.01 MB)","Per sample\n(µs)","Peak memory\n(100 MB)"];vals=[d["parameters"]/1e3,d["model_size_mb"]/.01,d["per_sample_us"],d["peak_gpu_memory_mb"]/100]
    fig,ax=plt.subplots(figsize=(5.3,2.65));bars=ax.bar(range(4),vals,color=["#6C8EBF","#8DB596","#D7B377","#B67A7A"]);ax.set_xticks(range(4),labs);ax.set_ylabel("Scaled value");ax.set_title("Legacy Qwen-only gate overhead benchmark");nice(ax);panel(ax,"a")
    for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v+.15,f"{v:.2f}",ha="center",fontsize=6)
    save(fig,"ExtFig7_predictor_overhead_legacy")

def ext8_cache_matrix():
    rows=load_csv(ROOT/"adaptive_cache/routecast_v3_limit100_adaptive_cache.csv");pol=["LRU","fixed_native","adaptive_mass","adaptive_cache"]; caps=[1,2,4]
    fig,axs=plt.subplots(3,3,figsize=(7.2,6.1))
    metrics=[("hit_rate","Hit rate",1),("total_transfers","Transfers (M)",1e6),("prefetch_precision","Prefetch precision",1)]
    for i,m in enumerate(MODELS):
        rr=[r for r in rows if r["model"]==m]
        for j,(key,title,scale) in enumerate(metrics):
            mat=np.array([[float(next(r[key] for r in rr if r["policy"]==p and int(r["capacity"])==c))/scale for c in caps] for p in pol]);im=axs[i,j].imshow(mat,aspect="auto",cmap="Blues")
            axs[i,j].set_xticks(range(3),caps);axs[i,j].set_yticks(range(4),[x.replace("_"," ") for x in pol] if j==0 else []);axs[i,j].set_title(title if i==0 else "");axs[i,j].set_xlabel("Cache capacity (×)" if i==2 else "");
            if j==0:axs[i,j].set_ylabel(DISPLAY[m]);
            for a in range(4):
                for b in range(3):axs[i,j].text(b,a,f"{mat[a,b]:.2f}",ha="center",va="center",fontsize=5,color="white" if mat[a,b]>.65*mat.max() else "#222")
            panel(axs[i,j],chr(97+i*3+j))
    save(fig,"ExtFig8_full_cache_replay_matrix")

def write_manifest():
    items=[
      ("Fig1_routecast_framework","Method schematic","No quantitative data"),
      ("Fig2_main_prediction","Main native-K comparison and paired CI","Current limit=1000"),
      ("Fig3_budget_quality_tradeoff","Recall-waste budget curves","Cross-model limit=100 development"),
      ("Fig4_cross_model_ablation","Component ablation","Cross-model limit=100 development"),
      ("Fig5_calibration_and_mass","Reliability and mass selection","Cross-model limit=100 development"),
      ("Fig6_cache_aware_replay","Trace-driven cache replay","Cross-model limit=100 development"),
      ("ExtFig1_dataset_composition","Dataset composition","Current limit=1000 manifest"),
      ("ExtFig2_history_sensitivity_legacy","History sensitivity","Legacy Qwen-only development"),
      ("ExtFig3_data_scale_generalization","Scale comparison","limit=100 vs stable limit=1000"),
      ("ExtFig4_seed_reproducibility","Seed robustness","Cross-model limit=100"),
      ("ExtFig5_layerwise_heterogeneity_legacy","Layer heterogeneity","Legacy Qwen-only"),
      ("ExtFig6_calibration_metrics","Calibration metrics","Cross-model limit=100"),
      ("ExtFig7_predictor_overhead_legacy","Predictor overhead","Legacy Qwen-only; not V3"),
      ("ExtFig8_full_cache_replay_matrix","Full cache matrix","Cross-model limit=100")]
    md=["# RouteCast figure package","","> Main and development evidence are explicitly separated. Legacy panels must not be described as RouteCast V3 cross-model results.","","| Figure | Claim role | Evidence status |","|---|---|---|"]+[f"| {a} | {b} | {c} |" for a,b,c in items]
    (OUT/"FIGURE_INDEX.md").write_text("\n".join(md)+"\n",encoding="utf-8")
    (OUT/"figure_manifest.json").write_text(json.dumps([{"figure":a,"role":b,"evidence_status":c} for a,b,c in items],indent=2),encoding="utf-8")

def main():
    for fn in [fig1_framework,fig2_main,fig3_budget,fig4_ablation,fig5_calibration_mass,fig6_cache,ext1_dataset,ext2_history,ext3_scale,ext4_seeds,ext5_layer,ext6_calibration_metrics,ext7_overhead,ext8_cache_matrix]:fn()
    write_manifest();print(f"complete={OUT}")
if __name__=="__main__":main()
