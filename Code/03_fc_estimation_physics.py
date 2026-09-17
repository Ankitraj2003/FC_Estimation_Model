from pathlib import Path
import re, numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]; R=ROOT/'Results'

def parse_depth(c):
    m=re.search(r'_(\d+)_m3',c); return int(m.group(1)) if m else int(re.search(r'(\d+)',c).group(1))

def rfp_fc(y, tol=0.0005):
    """R-FP-style plateau: first sustained low-change segment after the peak."""
    d=np.gradient(y)
    peak=int(np.argmax(y))
    for i in range(peak+2,len(y)-2):
        if np.nanmedian(np.abs(d[i:i+3]))<tol:
            return float(np.nanmedian(y[i:i+3])),i
    return float(y[-1]),len(y)-1

def run_physics_fc_estimation():
    df=pd.read_csv(R/'cleaned_vmc_data.csv'); df.Timestamp=pd.to_datetime(df.Timestamp,format='mixed')
    cy=pd.read_csv(R/'valid_recession_cycles.csv'); cy=cy[cy.Is_Valid.astype(bool)].copy()
    rows=[]
    for _,c in cy.iterrows():
        col=c.Column; sched=pd.to_datetime(c.Scheduled_Irrigation_Time); peak=pd.to_datetime(c.Peak_Time)
        rec_post=df[(df.Timestamp>=peak+pd.Timedelta(hours=2))&(df.Timestamp<=sched)].copy()
        fc_r=np.nan; swdp_r_r2=np.nan
        if len(rec_post)>=3:
            y_r=rec_post[col+'_smoothed'].to_numpy(float)
            t_r=(rec_post.Elapsed_hours-rec_post.Elapsed_hours.iloc[0]).to_numpy(float)+2.0
            pos=t_r>0
            if pos.sum()>=3:
                coef=np.polyfit(1/t_r[pos],y_r[pos],1); pred=np.polyval(coef,1/t_r[pos]); fc_r=float(coef[1])
                ss_res=float(np.sum((y_r[pos]-pred)**2)); ss_tot=float(np.sum((y_r[pos]-np.mean(y_r[pos]))**2)); swdp_r_r2=1-ss_res/ss_tot if ss_tot>0 else np.nan
        rec=df[(df.Timestamp>=pd.to_datetime(c.Peak_Time))&(df.Timestamp<=sched)]
        yr=rec[col+'_smoothed'].to_numpy(float)
        fc_fp,_=rfp_fc(yr) if len(yr)>=5 else (np.nan,None)
        fc_direct=float(c.FC_Pre_Irrigation_VMC)
        # SWDP-K: numerical knee sensitivity estimate, not described as curvature.
        end=len(yr)-1; xx=np.arange(end+1); denom=np.hypot(end,yr[-1]-yr[0]); dist=np.abs((yr[-1]-yr[0])*xx-end*(yr-yr[0]))/denom if denom else np.zeros_like(xx); fc_k=float(yr[int(np.argmax(dist))])
        vals=[fc_direct,fc_fp,fc_r]; valid_vals=[v for v in vals if np.isfinite(v)]
        spread=max(valid_vals)-min(valid_vals) if valid_vals else np.nan
        # Consensus target: robust median of independent recession-based estimators when they agree.
        # The 0.03 V/V agreement bound follows the scale used by Bean et al. when comparing
        # automated FC estimates with expert values; it is a validation/consensus rule, not ground truth.
        if valid_vals and spread<=0.03:
            fc_cons=float(np.median(valid_vals)); target_method='median(Direct,R-FP,SWDP-R)'
        else:
            fc_cons=fc_direct; target_method='expert-style direct point'
        rows.append({'Event_ID':int(c.Event_ID),'Column':col,'Depth_cm':int(c.Depth_cm),'Peak_VMC':float(c.Peak_VMC),
                     'FC_ExpertStyle_Direct':fc_direct,'FC_R_FP':fc_fp,'FC_SWDP_R':fc_r,'SWDP_R_R2':swdp_r_r2,
                     'FC_SWDP_K':fc_k,'Method_Spread':spread,'FC_Reference':fc_cons,'Target_Method':target_method})
    out=pd.DataFrame(rows)
    out['FC_Primary']=out['FC_Reference']
    out['FC_Robust_Median']=out[['FC_ExpertStyle_Direct','FC_R_FP','FC_SWDP_R']].median(axis=1,skipna=True)
    out.to_csv(R/'physics_fc_event_estimates.csv',index=False)
    summary=out.groupby('Depth_cm').agg(Valid_Cycles=('Event_ID','count'),FC_Reference_Mean=('FC_Reference','mean'),FC_Reference_SD=('FC_Reference','std'),FC_Direct_Mean=('FC_ExpertStyle_Direct','mean'),FC_R_FP_Mean=('FC_R_FP','mean'),FC_SWDP_R_Mean=('FC_SWDP_R','mean'),SWDP_R_R2_Median=('SWDP_R_R2','median')).reset_index()
    summary['CV_pct']=100*summary.FC_Reference_SD/summary.FC_Reference_Mean
    summary.to_csv(R/'physics_fc_summary_by_depth.csv',index=False)
    print(summary.to_string(index=False)); return out,summary
if __name__=='__main__': run_physics_fc_estimation()
