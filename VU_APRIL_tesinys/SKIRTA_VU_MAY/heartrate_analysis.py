import neurokit2 as nk
import pandas as pd
import numpy as np
import math
from atrial_fibrillation import GetAtrialFibrillationEpisodes

def get_bradycardia_episodes(df, bradycardia_min_rate = 60, bradycardia_min_duration = 15):
  # 'where' will change hr_mean value to NaN for non-compliant lines in df
  br = df.where(df['hr_mean'] < bradycardia_min_rate)

  episodes_all=[]
  episode = {}
  in_data = False
  last_v = None

  # iterate over all df lines and find non - NaN chunks, mark their start and end timestamps
  for v in br.itertuples():
      if not in_data:
          if not math.isnan(v.hr_mean):
              episode = {}
              episode['start_ts'] = v.Index
              episode['start'] = int(v.sample_no)
              in_data = True
              last_v = None
      else:
          if math.isnan(v.hr_mean):
            if last_v == None:
              in_data = False
              #last_v is None in case where after start of episode NaN as the next sample, meaning episode is a single sample
              #we don't accept episodes with length of a single sample
            else:
              episode['end_ts'] = last_v.Index
              episode['end'] = int(last_v.sample_no)
              episodes_all.append(episode)
              in_data = False
          last_v = v

  episodes = []

  # calculate duration and mean_hr values for every episode, select episodes of bradycardia_min_duration length
  for e in episodes_all:
      duration_s = (e['end_ts'] - e['start_ts']).seconds
      if duration_s >= bradycardia_min_duration:
          e['duration_s'] = duration_s
          # e['mean_hr'] = int(br['hr_mean'][e['start_ts']:e['end_ts']].mean().round(0))  # buvo,, pakeista į kj
          e['mean_hr'] = int(round(br['hr_mean'][e['start_ts']:e['end_ts']].mean()))   # kj
          # del e['end_ts']
          del e['end_ts']
          del e['start_ts']
          episodes.append(e)
  return episodes

def get_pause_episodes(df, sampling_rate=200, pause_min_duration_s=2):
    pause_min_duration = pause_min_duration_s * sampling_rate

    p = df[df['diff'] > pause_min_duration]
    episodes=[]
    episode = {}
    in_data = False
    i = 0
    last_v = None
    for v in p.itertuples():
      episode = {}
      episode['onset_sampleno'] = int(v.sample_no)
      episode['length'] = v.diff
      episodes.append(episode)
    return episodes

def heart_rate_is_sane(hr):
  return hr > 20 or hr < 300


def AnalyseHeartrate(ecg_signal_df, sampling_rate = 200, avg_time_s=30, bradycardia_min_rate=60, bradycardia_min_duration=15, pause_min_duration_s=2):
  try:
    _, rpeaks = nk.ecg_peaks(ecg_signal_df['orig'], sampling_rate=sampling_rate, correct_artifacts=False)

    df = pd.DataFrame(rpeaks['ECG_R_Peaks'], columns=['sample_no'])
    df['diff'] = np.insert(np.diff(df['sample_no']), 0, 0, axis=0)

    df['ts'] = pd.to_timedelta(df['sample_no'] / sampling_rate, unit='seconds')
    df.set_index('ts', inplace=True)
    # df.drop(columns=[0], inplace=True)

    df['hr_bpm'] = 60 * sampling_rate/df['diff']
    df['hr_mean'] = df['hr_bpm'].rolling(F'{avg_time_s}s').mean()
    df.drop(columns=['hr_bpm'], inplace=True)

    hrate = nk.ecg_rate(rpeaks['ECG_R_Peaks'], sampling_rate=200)

    if (np.ndarray == type(hrate) and len(hrate) > 2) or float == type(hrate) and not math.isnan(hrate) :
      #shift heart rate one position to left in order to better correspond to r-peaks
      hrate = hrate[1:]
      hrate = np.append(hrate, hrate[hrate.size - 1])
    else:
      hrate = []
      
    if df['hr_mean'].size > 2:
      # hr_min = int(df['hr_mean'].min().round(0))# buvo, pakeista į kj
      hr_min = int(round(df['hr_mean'].min(), 0))  # kj 
      # hr_max = int(df['hr_mean'].max().round(0)) # buvo, pakeista į kj 
      hr_max = int(round(df['hr_mean'].max(), 0))  # kj
      # hr_avg = int(df['hr_mean'].mean().round(0))# buvo, pakeista į kj
      hr_avg = int(round(df['hr_mean'].mean(), 0))  # kj

      if not heart_rate_is_sane(hr_min):
        hr_min = -1
      if not heart_rate_is_sane(hr_max):
        hr_max = -1
      if not heart_rate_is_sane(hr_avg):
        hr_avg = -1
    else:
      hr_min = -1
      hr_max = -1
      hr_avg = -1

    ret = {
    'rpeaks':rpeaks['ECG_R_Peaks'].tolist(),
    'heartrate':{
        'min':hr_min,
        'max':hr_max,
        'avg':hr_avg,
        'beats':len(rpeaks['ECG_R_Peaks']),
        'rate':hrate.astype("uint32").tolist()
        }
    }
    ret['bradycardia']={
        'episodes' : get_bradycardia_episodes(df, bradycardia_min_rate, bradycardia_min_duration)
    }
    ret['pause'] = {
        'episodes' : get_pause_episodes(df, sampling_rate, pause_min_duration_s)
    }

    ret['afib'] = {
      'episodes' : GetAtrialFibrillationEpisodes(rpeaks['ECG_R_Peaks'].tolist())
    }
  except Exception as error:
    ret = {
    'rpeaks':[],
    'heartrate':{
        'min':-1,
        'max':-1,
        'avg':-1,
        'beats':0,
        'rate':[]
        }
    }
    ret['bradycardia']={
        'episodes' : []
    }
    ret['pause'] = {
        'episodes' : []
    }
    ret['afib'] = {
      'episodes' : []
    }

  return ret

def DelineateQRS(ecg_signal_df, rpeaks):
  try:
    _, delineation = nk.ecg_delineate(ecg_signal_df['orig'], rpeaks, method='peaks', sampling_rate=200, show=False)
    delineation['ECG_P_Peaks'] = np.array(delineation['ECG_P_Peaks']).astype('uint32').tolist()
    delineation['ECG_Q_Peaks'] = np.array(delineation['ECG_Q_Peaks']).astype('uint32').tolist()
    delineation['ECG_S_Peaks'] = np.array(delineation['ECG_S_Peaks']).astype('uint32').tolist()
    delineation['ECG_T_Peaks'] = np.array(delineation['ECG_T_Peaks']).astype('uint32').tolist()  
  except Exception as e:
    delineation = {}
    delineation['ECG_P_Peaks'] = []
    delineation['ECG_Q_Peaks'] = []
    delineation['ECG_S_Peaks'] = []
    delineation['ECG_T_Peaks'] = []  
  return delineation