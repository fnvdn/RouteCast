"""Generate atomic, single-claim panels for IEEE two-column placement."""
from pathlib import Path
import sys,json,csv
CODE_ROOT=Path(__file__).resolve().parent
PROJECT_ROOT=CODE_ROOT.parent
sys.path.insert(0,str(PROJECT_ROOT/"python_packages"/"figure"))
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT=PROJECT_ROOT/"record"; OUT=ROOT/"paper/figures/split";OUT.mkdir(parents=True,exist_ok=True)
mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],"font.size":7,
"axes.titlesize":8,"axes.labelsize":7,"xtick.labelsize":6.5,"ytick.labelsize":6.5,"legend.fontsize":6,
"axes.spines.top":False,"axes.spines.right":False,"svg.fonttype":"none","pdf.fonttype":42})
M=["qwen3","deepseek_r1","llama4_maverick"];D={"qwen3":"Qwen3","deepseek_r1":"DeepSeek-R1","llama4_maverick":"Llama4 Maverick"}
C={"Paper Heatmap":"#A7B0B8","Request-aware Heatmap":"#7D8C99","GRU-only":"#91B7D8","TCN-only":"#5F88B2","RouteCast V3":"#D55E5E","qwen3":"#3B78B5","deepseek_r1":"#D98242","llama4_maverick":"#4F9D69","LRU":"#9AA1A7","fixed_native":"#6C8EBF","adaptive_mass":"#D6A34A","adaptive_cache":"#C94C4C"}
def js(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def cs(p):
 with Path(p).open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def nice(ax,axis="y"):ax.grid(axis=axis,color="#E7E9EB",lw=.6);ax.set_axisbelow(True)
def save(fig,n):
 fig.tight_layout(pad=.65)
 fig.savefig(OUT/f"{n}.svg",bbox_inches="tight");fig.savefig(OUT/f"{n}.pdf",bbox_inches="tight")
 fig.savefig(OUT/f"{n}.png",dpi=300,bbox_inches="tight");fig.savefig(OUT/f"{n}.tiff",dpi=600,bbox_inches="tight");plt.close(fig);print(n,flush=True)

def main_comparison():
 rows=js(ROOT/"paper/results/main_limit1000/main_forecasting_limit1000.json")["rows"]; methods=["Paper Heatmap","Request-aware Heatmap","GRU-only","TCN-only","RouteCast V3"]
 fig,ax=plt.subplots(figsize=(7.2,2.55));x=np.arange(3);w=.15
 for i,me in enumerate(methods):
  vals=[next(r["recall"] for r in rows if r["model"]==m and r["method"]==me) for m in M];display="RouteCast" if me=="RouteCast V3" else me;ax.bar(x+(i-2)*w,vals,w,label=display,color=C[me],zorder=3)
 ax.set_xticks(x,["Qwen3\nRecall@8","DeepSeek-R1\nRecall@8","Llama4\nRecall@1"]);ax.set_ylabel("Native-budget recall");ax.set_ylim(0,.64);ax.legend(ncol=3,loc="upper right");nice(ax);save(fig,"Fig2a_main_grouped_comparison")
 boots={"GRU-only":js(ROOT/"paper/results/main_limit1000/bootstrap_routecast_vs_gru_limit1000.json")["models"],"TCN-only":js(ROOT/"paper/results/main_limit1000/bootstrap_routecast_vs_tcn_limit1000.json")["models"]}
 fig,ax=plt.subplots(figsize=(3.25,2.45));ys=np.arange(3);offset={"GRU-only":-.11,"TCN-only":.11}
 for label,boot in boots.items():
  v=[];lo=[];hi=[]
  for m in M:
   z=boot[m]["metrics"]["recall"];v.append(z["delta"]);lo.append(z["delta"]-z["ci95"][0]);hi.append(z["ci95"][1]-z["delta"])
  ax.errorbar(v,ys+offset[label],xerr=[lo,hi],fmt="o",label=label,color=C[label],ecolor=C[label],capsize=2.5,ms=4)
 ax.axvline(0,color="#777",ls=":");ax.set_yticks(ys,[D[m] for m in M]);ax.invert_yaxis();ax.set_xlabel("RouteCast Recall gain\n(95% paired request bootstrap CI)");ax.legend(loc="lower center",bbox_to_anchor=(.5,1.01),ncol=2);nice(ax,"x");save(fig,"Fig2b_paired_bootstrap")

def budgets():
 dat=js(ROOT/"budget_curve_limit100_v2.json")["results"]
 for m in M:
  r=dat[m];fig,ax=plt.subplots(figsize=(3.25,2.35));k=[x["prefetch_k"] for x in r]
  ax.plot(k,[x["recall"] for x in r],"o-",label="Recall",color=C[m]);ax.plot(k,[x["wasted_prefetch_ratio"] for x in r],"s--",label="Waste",color="#7E8790")
  ax.axvline(r[0]["native_topk"],color="#222",ls=":",label="Native K");ax.set_ylim(0,1);ax.set_xlabel("Prefetched experts (K)");ax.set_ylabel("Rate");ax.set_title(D[m]);ax.legend(loc="best");nice(ax);save(fig,f"Fig3_budget_{m}")

def calibration_mass():
 cal=js(ROOT/"calibration/routecast_v3_limit100_calibration.json")["models"];cache=js(ROOT/"adaptive_cache/routecast_v3_limit100_adaptive_cache.json")["models"]
 for m in M:
  z=cal[m];fig,ax=plt.subplots(figsize=(3.25,2.35))
  for key,lab,ls,col in [("test_before","Before","--","#888"),("test_after","After","-",C[m])]:
   rr=[x for x in z[key]["reliability"] if x["count"]>0];ax.plot([x["confidence"] for x in rr],[x["accuracy"] for x in rr],"o"+ls,ms=2.6,label=lab,color=col)
  ax.plot([0,1],[0,1],":",color="#333");ax.set(xlim=(0,1),ylim=(0,1),xlabel="Confidence",ylabel="Empirical accuracy",title=f"{D[m]}  T={z['temperature']:.2f}");ax.legend();nice(ax);save(fig,f"Fig5_reliability_{m}")
  curve=cache[m]["val_curve"];mass=sorted(float(x) for x in curve);avg=[curve[str(x)]["avg_k"] for x in mass];rec=[curve[str(x)]["recall"] for x in mass];chosen=cache[m]["selected_mass"];q=curve[str(chosen)]
  fig,ax=plt.subplots(figsize=(3.25,2.35));ax.plot(avg,rec,"o-",ms=3,color=C[m]);ax.scatter(q["avg_k"],q["recall"],s=40,color="#C94C4C",label=f"Selected m={chosen:g}",zorder=3);ax.set(xlabel="Average candidate width",ylabel="Validation recall",title=D[m]);ax.legend();nice(ax);save(fig,f"Fig5_mass_{m}")

def cache_pareto():
 rows=cs(ROOT/"adaptive_cache/routecast_v3_limit100_adaptive_cache.csv");pol=["LRU","fixed_native","adaptive_mass","adaptive_cache"]
 for m in M:
  fig,ax=plt.subplots(figsize=(3.25,2.45));rr=[r for r in rows if r["model"]==m]
  for p in pol:
   z=sorted([r for r in rr if r["policy"]==p],key=lambda x:int(x["capacity"]));x=[float(r["total_transfers"])/1e6 for r in z];y=[float(r["hit_rate"]) for r in z]
   ax.plot(x,y,"o-",ms=3,color=C[p],label=p.replace("_"," "))
   for xx,yy,r in zip(x,y,z):ax.annotate(r["capacity"]+"x",(xx,yy),xytext=(3,2),textcoords="offset points",fontsize=5.5)
  ax.set(xlabel="Total transfers (million)",ylabel="Cache hit rate",title=D[m]);ax.legend(fontsize=5.3);nice(ax);save(fig,f"Fig6_cache_pareto_{m}")

if __name__=="__main__":main_comparison();budgets();calibration_mass();cache_pareto()
