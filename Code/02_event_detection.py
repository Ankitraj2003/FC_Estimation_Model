from pathlib import Path
import re, numpy as np, pandas as pd
from scipy.signal import find_peaks
ROOT=Path(__file__).resolve().parents[1]; R=ROOT/'Results'
WETTING_THRESHOLD=0.006  # Bean et al. (2018)
NO_INPUT_START_HOUR=22
DEFAULT_IRR_HOUR=7

def parse_depth(c):
    m=re.search(r'_(\d+)_m3',c); return int(m.group(1)) if m else int(re.search(r'(\d+)',c).group(1))

def detect_events(df):
    # A wetting event must be supported by an observed precipitation/irrigation input.
    # Do not invent events from VMC rise alone. Very small measured inputs are retained;
    # the 0.006 V/V threshold below decides whether they produced a defensible cycle.
    mask=df.Total_Water_mm>0
    idx=np.flatnonzero(mask.to_numpy()); groups=[]
    if len(idx):
        s=p=idx[0]
        for i in idx[1:]:
            if i-p<=3: p=i
            else: groups.append((s,p)); s=p=i
        groups.append((s,p))
    rows=[]
    for eid,(s,e) in enumerate(groups,1):
        g=df.iloc[s:e+1]
        rows.append({'Event_ID':eid,'Start_Idx':s,'End_Idx':e,'Start_Time':g.Timestamp.iloc[0],'End_Time':g.Timestamp.iloc[-1],
                     'Precipitation_mm':g.Precipitation_mm.sum(),'Irrigation_mm':g.Irrigation_mm.sum(),'Total_Water_mm':g.Total_Water_mm.sum()})
    return pd.DataFrame(rows)

def next_scheduled_time(peak_time, hour):
    base=peak_time.normalize()+pd.Timedelta(days=1,hours=hour)
    if peak_time < base: return base
    return base+pd.Timedelta(days=1)

def run_event_detection():
    df=pd.read_csv(R/'cleaned_vmc_data.csv'); df['Timestamp']=pd.to_datetime(df['Timestamp'],format='mixed')
    sm=sorted([c for c in df.columns if c.startswith('SM_') and not any(c.endswith(x) for x in ['_smoothed','_d1','_d2'])],key=parse_depth)
    events=detect_events(df)
    if events.empty: raise RuntimeError('No measured water-input events found.')
    irr=df.loc[df.Irrigation_mm>0,'Timestamp'].dt.hour
    scheduled=int(irr.mode().iloc[0]) if len(irr) else DEFAULT_IRR_HOUR
    events['Scheduled_Irrigation_Hour']=scheduled; events.to_csv(R/'detected_events_summary.csv',index=False)
    cycles=[]
    for _,ev in events.iterrows():
        s,e=int(ev.Start_Idx),int(ev.End_Idx)
        # Search for the first local maximum with the Bean et al. wetting threshold.
        search_end=min(e+24,len(df)-1)
        for c in sm:
            x=df[c+'_smoothed'].to_numpy(); baseline=float(np.nanmedian(x[max(0,s-6):s+1]))
            peaks,_=find_peaks(x[s:search_end+1],prominence=0.001,distance=4)
            cand=[s+int(i) for i in peaks if x[s+int(i)]-baseline>=WETTING_THRESHOLD]
            if not cand:
                p=s+int(np.nanargmax(x[s:search_end+1])); cand=[p] if x[p]-baseline>=WETTING_THRESHOLD else []
            if not cand: continue
            peak_idx=cand[0]; peak=float(x[peak_idx]); peak_time=df.Timestamp.iloc[peak_idx]; rise=peak-baseline
            sched=next_scheduled_time(peak_time,scheduled)
            # FC is evaluated immediately before the scheduled irrigation time, after one clean overnight drainage.
            overnight_start=peak_time.normalize()+pd.Timedelta(hours=NO_INPUT_START_HOUR)
            overnight_end=peak_time.normalize()+pd.Timedelta(days=1,hours=scheduled)
            if peak_time >= overnight_end:
                overnight_start += pd.Timedelta(days=1)
                overnight_end += pd.Timedelta(days=1)
            night=df[(df.Timestamp>=overnight_start)&(df.Timestamp<=overnight_end)].copy()
            # Any input between 22:00 and scheduled irrigation invalidates the cycle, exactly as Bean expert labeling.
            night_input=float(night.Total_Water_mm.sum()) if len(night) else np.nan
            # Use a response window through the scheduled irrigation; no need to demand an arbitrary 12-h recession.
            rec=df[(df.Timestamp>=peak_time)&(df.Timestamp<=sched)].copy()
            rec_hours=float(rec.Elapsed_hours.iloc[-1]-rec.Elapsed_hours.iloc[0]) if len(rec)>1 else 0
            reason='Valid'; valid=True; slope=np.nan
            if rise < WETTING_THRESHOLD: valid=False; reason='Wetting rise < 0.006 m3/m3'
            elif len(night)<4: valid=False; reason='Insufficient overnight observations before scheduled irrigation'
            elif night_input>0: valid=False; reason='Additional rainfall/irrigation between 22:00 and scheduled irrigation'
            else:
                y=night[c+'_smoothed'].to_numpy(); tt=night.Elapsed_hours.to_numpy()
                slope=np.polyfit(tt-tt[0],y,1)[0] if len(y)>=3 else np.nan
                # Expert criterion: drying must be a steady decline overnight. Allow a small sensor-scale tolerance.
                dy=np.diff(y)
                if np.nanmedian(dy)>=0 or slope>=0: valid=False; reason='Overnight drying was not a steady decline'
                elif np.nanmax(dy)>0.002: valid=False; reason='Overnight series contains a substantial rebound'
            fc_at_sched=float(night.loc[night.Timestamp<=sched,c+'_smoothed'].iloc[-1]) if len(night) else np.nan
            cycles.append({'Event_ID':int(ev.Event_ID),'Column':c,'Depth_cm':parse_depth(c),'Event_Start_Time':ev.Start_Time,'Peak_Time':peak_time,'Peak_Idx':peak_idx,
                           'Scheduled_Irrigation_Time':sched,'Recession_End_Idx':int(night.index[-1]) if len(night) else int(peak_idx),
                           'Pre_Event_VMC':baseline,'Peak_VMC':peak,'VMC_Rise':rise,'Total_Water_mm':ev.Total_Water_mm,
                           'Recession_Hours':rec_hours,'Overnight_Input_mm':night_input,'Overnight_Slope':slope,'FC_Pre_Irrigation_VMC':fc_at_sched,
                           'Is_Valid':valid,'Rejection_Reason':reason})
    out=pd.DataFrame(cycles); out.to_csv(R/'valid_recession_cycles.csv',index=False)
    print(f'Measured water events: {len(events)}; depth cycles: {len(out)}; valid: {int(out.Is_Valid.sum())}')
    print(out.Rejection_Reason.value_counts().to_string())
    return events,out
if __name__=='__main__': run_event_detection()
