import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import numpy as np
import pandas as pd
import json

from quality_analysis import EstimateQuality
from heartrate_analysis import AnalyseHeartrate, DelineateQRS
from zive_cnn_fda_vu_v1 import classify_cnn_fda_vu_v1, zive_read_file_1ch, zive_read_file_3ch


import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument('fileName', metavar="FILE", type=str, help="Path to file containing ECG recording")
parser.add_argument('--channelCount', metavar='COUNT', type=int, choices=(1,3), required=True, help='Number of channels in the file')
parser.add_argument('--recordingId', metavar='REC_ID', type=str, required=True, help='Recording Id')
args = parser.parse_args()

results = {
  'status': {
      'success': True
  },
  'recording_id': args.recordingId
}

hr_analysis_success = True
try:  
  if args.channelCount == 3:
    ecg_signal_df = pd.DataFrame(zive_read_file_3ch(args.fileName), columns=['orig'])
  elif args.channelCount == 1:
    ecg_signal_df = pd.DataFrame(zive_read_file_1ch(args.fileName), columns=['orig'])

  quality = EstimateQuality(ecg_signal_df, method="variance")
  results['quality'] = quality
  
  analysis_results = AnalyseHeartrate(ecg_signal_df)

  results['rpeaks'] = analysis_results['rpeaks']
  results['rate'] = analysis_results['heartrate']['rate']
  results['heartrate'] = analysis_results['heartrate']
  results['bradycardia'] = analysis_results['bradycardia']
  results['pause'] = analysis_results['pause']
  results['afib'] = analysis_results['afib']

  delineation = DelineateQRS(ecg_signal_df, analysis_results['rpeaks'])
  
  results['qpeaks'] = delineation['ECG_Q_Peaks']
  results['ppeaks'] = delineation['ECG_P_Peaks']
  results['speaks'] = delineation['ECG_S_Peaks']
  results['tpeaks'] = delineation['ECG_T_Peaks']

  classification = classify_cnn_fda_vu_v1(zive_read_file_1ch(args.fileName), atr_sample=analysis_results['rpeaks'], 
                                                  model_dir='model_cnn_fda_vu_v1', prediction_labels=['N', 'S', 'V', 'U'])
 
  results['automatic_classification'] = classification


except Exception as error:
  results['status']['success'] = False
  results['status']['error'] = str(error)
else:
  pass
  # print('ECG Analysis failed')

print(json.dumps(results))
