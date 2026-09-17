from pathlib import Path
import re
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'Data'; RESULTS = ROOT / 'Results'; RESULTS.mkdir(exist_ok=True)
INPUT = DATA / 'FINAL_VMC_Irrigation_Precipitation_Merged.xlsx'

def parse_depth(col):
    m = re.search(r'_(\d+)_m3', col)
    return int(m.group(1)) if m else int(re.search(r'(\d+)', col).group(1))

def run_preprocessing():
    df = pd.read_excel(INPUT)
    df.columns = [str(c).replace("'", '').strip() for c in df.columns]
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values('Timestamp').drop_duplicates('Timestamp', keep='first').reset_index(drop=True)
    sm_cols = sorted([c for c in df.columns if c.startswith('SM_')], key=parse_depth)
    qc=[]
    df = df.set_index('Timestamp')
    for c in sm_cols:
        bad = ~df[c].between(0,1) | df[c].isna()
        n=int(bad.sum())
        df.loc[bad,c]=np.nan
        df[c]=df[c].interpolate(method='time', limit_direction='both')
        qc.append({'Column':c,'Depth_cm':parse_depth(c),'Invalid_or_missing_Count':n,'Min_VMC':df[c].min(),'Max_VMC':df[c].max(),'Mean_VMC':df[c].mean(),'Std_VMC':df[c].std()})
    df = df.reset_index()
    for c in ['Irrigation_mm','Precipitation_mm']:
        df[c]=pd.to_numeric(df[c],errors='coerce').fillna(0).clip(lower=0)
    df['Total_Water_mm']=df['Irrigation_mm']+df['Precipitation_mm']
    dt_sec = df['Timestamp'].diff().dt.total_seconds().to_numpy(copy=True)
    dt_sec[0] = np.nanmedian(dt_sec[1:])
    df['Gap_hours'] = dt_sec / 3600.0
    df['Elapsed_hours'] = (df['Timestamp'] - df['Timestamp'].iloc[0]).dt.total_seconds() / 3600.0
    for c in sm_cols:
        x=df[c].to_numpy(float)
        sm=savgol_filter(x,7,2) if len(x)>=7 else x
        d1=np.gradient(sm,df['Elapsed_hours'])
        d2=np.gradient(d1,df['Elapsed_hours'])
        df[f'{c}_smoothed']=sm; df[f'{c}_d1']=d1; df[f'{c}_d2']=d2
    df.to_csv(RESULTS/'cleaned_vmc_data.csv',index=False)
    pd.DataFrame(qc).to_csv(RESULTS/'data_quality_report.csv',index=False)
    print(f'Preprocessed {len(df)} rows, {len(sm_cols)} sensors; gaps >1.5 h: {(df.Gap_hours>1.5).sum()}')
    return df
if __name__=='__main__': run_preprocessing()
