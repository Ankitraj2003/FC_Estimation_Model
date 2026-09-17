"""R-SAX-style symbolic cycle similarity.

Bean et al. (2018) used R-SAX as a supervised cycle-labeling method with expert-labeled
training cycles. This project has no independent expert labels, so this implementation
DOES NOT replace the hydrological validation rules. Instead it computes a leave-one-event-out
SAX similarity feature using only already rule-valid cycles. This is a scientifically
conservative use of SAX until independent manual labels are available.
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import norm
ROOT=Path(__file__).resolve().parents[1]; R=ROOT/'Results'

def sax_word(y, word_size=8, alphabet_size=5):
    y=np.asarray(y,float); y=y[np.isfinite(y)]
    if len(y)<word_size: return None
    edges=np.linspace(0,len(y),word_size+1).astype(int)
    means=np.array([np.mean(y[edges[i]:edges[i+1]]) for i in range(word_size)])
    z=(means-means.mean())/(means.std() if means.std()>1e-12 else 1.0)
    bp=norm.ppf(np.arange(1,alphabet_size)/alphabet_size)
    return ''.join(chr(97+int(np.sum(v>bp))) for v in z)

def similarity(a,b):
    if a is None or b is None: return np.nan
    # Normalized Hamming similarity, a simple direct analogue of symbolic sequence comparison.
    n=min(len(a),len(b)); return float(np.mean([a[i]==b[i] for i in range(n)]))

def run():
    df=pd.read_csv(R/'cleaned_vmc_data.csv'); df.Timestamp=pd.to_datetime(df.Timestamp,format='mixed')
    cy=pd.read_csv(R/'valid_recession_cycles.csv'); valid=cy[cy.Is_Valid.astype(bool)].copy()
    if valid.empty:
        pd.DataFrame(columns=['Event_ID','Column','SAX_Word','LeaveOneEventOut_SAX_Similarity']).to_csv(R/'r_sax_cycle_similarity.csv',index=False); return
    words=[]
    for _,r in valid.iterrows():
        a=int(r.Peak_Idx); sched=pd.to_datetime(r.Scheduled_Irrigation_Time); c=r.Column
        seg=df[(df.Timestamp>=pd.to_datetime(r.Peak_Time))&(df.Timestamp<=sched)][c+'_smoothed'].to_numpy()
        words.append({'Event_ID':int(r.Event_ID),'Column':c,'SAX_Word':sax_word(seg)})
    wd=pd.DataFrame(words)
    sims=[]
    for i,r in wd.iterrows():
        ref=wd[(wd.Event_ID!=r.Event_ID)&(wd.Column==r.Column)]['SAX_Word'].dropna().tolist()
        sims.append(np.nanmax([similarity(r.SAX_Word,w) for w in ref]) if ref else np.nan)
    wd['LeaveOneEventOut_SAX_Similarity']=sims
    wd.to_csv(R/'r_sax_cycle_similarity.csv',index=False)
    print('R-SAX diagnostic generated; it is not used as an independent validity label without expert training labels.')
if __name__=='__main__': run()
