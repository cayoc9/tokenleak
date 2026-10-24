"""Standalone scientific figure for the measured primary partition."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    summary=json.loads((args.run/'summary.json').read_text())
    names=['gpt2','gpt2_input_blinded','constant']
    metrics=[summary['conditions'][n]['new_session_new_context'] for n in names]
    means=np.array([m['top1']*100 for m in metrics])
    lower=np.array([m['cluster_bootstrap_ci95'][0]*100 for m in metrics])
    upper=np.array([m['cluster_bootstrap_ci95'][1]*100 for m in metrics])
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    axes[0].bar(np.arange(3),means,color=['#286f8e','#6c9f88','#b5b9bc'],
                yerr=np.vstack([means-lower,upper-means]),capsize=5)
    axes[0].axhline(12.5,color='#a53a3a',ls='--',label='Uniform / frequency prior: 12.5%')
    axes[0].set_xticks(range(3),['GPT-2','Input blinded','No inference'])
    axes[0].set_ylabel('Input-token candidate Top-1 (%)')
    axes[0].set_ylim(0,max(upper)+8)
    axes[0].legend(fontsize=8)
    for i,m in enumerate(metrics):
        axes[0].text(i,upper[i]+1,f'p={m["conditional_permutation_p_one_sided"]:.3f}',ha='center',fontsize=9)
    axes[0].set_title('Held-out session and contexts; n=128 per condition')
    rows=[json.loads(line) for line in (args.run/'predictions.jsonl').read_text().splitlines()]
    rows=[r for r in rows if r['condition']=='gpt2' and r['partition']=='new_session_new_context']
    classes=sorted({r['true_class'] for r in rows})
    values=[[r['features'][1] for r in rows if r['true_class']==c] for c in classes]
    axes[1].boxplot(values,labels=classes,showfliers=True)
    axes[1].set_ylabel('Request send to last marker (ms)')
    axes[1].set_xlabel('True input candidate (16 requests each)')
    axes[1].set_title('Observed response times overlap across candidates')
    for axis in axes:
        axis.grid(axis='y',alpha=.2)
    fig.suptitle('Measured GPT-2 small CPU pilot — no classification advantage demonstrated\n'
                 'Loopback laboratory markers; 95% intervals across eight repetition blocks; no physical cache probe',fontsize=11)
    for extension in ['png','pdf']:
        fig.savefig(args.output/f'observation_primary.{extension}',dpi=160)
    plt.close(fig)


if __name__=='__main__':
    main()
