# Skriptas testiniam ECG sekų skaitymui ir požymių skaičiavimui.
# Pritaikytas tiek MIT2ZIVE, tiek Zive duomenims

# Perdarytas iš ReadSeqEKG_2022_10_05_v2.py, tam kad požymių skaičiavimo modulis
# dirbtų su vienu pūpsniu. 
# 
# Originalų skriptą atsiuntė Povilas 2022 10 05, skriptas perdarytas.

import pandas as pd
import numpy as np
from pathlib import Path
import json
import math
# import random

import scipy.signal
# import skfda
from skfda import FDataGrid
# from skfda.preprocessing.smoothing import BasisSmoother
# from skfda.representation.basis import BSpline, Fourier

def read_RR_arr_from_signal(atr_sample, idx, nl_steps, nr_steps):
# Nuskaito ir pateikia EKG seką apie R dantelį seq: reikšmiu kiekis wl_side - iš kairės pusės, 
# reikšmiu kiekis wr_side - iš dešinės pusės, R dantelio vietą EKG įraše sample,
# ir atitinkamo pūpsnio klasės numerį label: 0, 1, 2.
# Taip pat pateikia seką RRl_arr iš nl_steps RR reikšmių tarp iš eilės einančių R dantelių į kairę nuo einamo R dantelio 
# ir seką RRr_arr nr_steps RR reikšmių tarp iš eilės einančių R dantelių į dešinę nuo einamo R dantelio dantelio.
# Seka iš kairės RRl_arr prasideda nuo tolimiausio nuo R dantelio atskaitymo ir jai pasibaigus,
# toliau ją pratesia RRl_arr. 

# **************************** Tikrinimai ******************************************************************

    # Tikriname, ar skaičiuodami RR neišeisime už atr_sample ribų
    if (idx + nr_steps) >= len(atr_sample):
        txt = f"Klaida 1! idx, nl_steps: {idx}, {nr_steps} Skaičiuojant RR viršijama pūpsnių atributo masyvo riba." 
        raise Exception(txt)  
        # Reikia mažinti nr_steps arba koreguoti viršutinę idx ribą
    
    if ((idx - nl_steps) < 0):
        txt = f"Klaida 2! idx, nl_steps: {idx}, {nl_steps} Skaičiuojant RR išeinama už pūpsnių atributo masyvo ribų."
        raise Exception(txt)  
        # Reikia mažinti nl_steps arba didinti apatinę idx ribą 
    
# **************************** Tikrinimų pabaiga ******************************************************************

    # Suformuojame RR sekas kairėje ir dešinėje idx atžvilgiu
    if (nl_steps != 0):
        RRl_arr = np.zeros(shape=(nl_steps), dtype=int)
        for i in range(nl_steps):
            RRl_arr[nl_steps-i-1] = atr_sample[idx-i] - atr_sample[idx-i-1]
    else:    
        RRl_arr = None

    if (nr_steps != 0):
        RRr_arr = np.zeros(shape=(nr_steps), dtype=int)
        for i in range(nr_steps):
            RRr_arr[i] = atr_sample[idx+i+1] - atr_sample[idx+i]
    else:
        RRr_arr = None        
    
    return RRl_arr, RRr_arr

def read_seq_from_signal(signal, atr_sample, idx, wl_side, wr_side):
# Nuskaito ir pateikia EKG seką apie R dantelį seq: reikšmiu kiekis wl_side - iš kairės pusės, 
# reikšmiu kiekis wr_side - iš dešinės pusės, R dantelio vietą EKG įraše sample,

    signal_length = signal.shape[0]
    (seq_start, seq_end)  = get_seq_start_end(signal_length, atr_sample[idx], wl_side, wr_side)

    # Tikriname, ar sekos langas neišeina už įrašo ribų
    if (seq_start == None or seq_end == None):
        raise Exception(f"Klaida! {idx}: Sekos lango rėžiai už EKG įrašo ribų.") 
        # Reikia mažinti wl_side ar wr_side, arba koreguoti idx ribas 
    else:    
        seq = signal[seq_start:seq_end]

    return seq


def get_seq_start_end(signal_length, i_sample, window_left_side, window_right_side):
    # Nustatome išskiriamos EKG sekos pradžią ir pabaigą
    seq_start = i_sample - window_left_side
    seq_end = i_sample + window_right_side
    if (seq_start < 0 or seq_end > signal_length):
        return (None, None)
    else:
        return (seq_start, seq_end)

def read_rec(rec_dir, SubjCode):
    file_path = Path(rec_dir, str(SubjCode) + '.npy')
    signal = np.load(file_path, mmap_mode='r')
    # print(f"SubjCode: {SubjCode}  signal.shape: {signal.shape}")
    return signal

def read_df_rpeaks(rec_dir, SubjCode):
    # Pritaikyta nuskaityti json informaciją tiek mit2zive, tiek zive atvejams
    file_path = Path(rec_dir, str(SubjCode) + '.json')

    if (SubjCode > 1000): # zive atvejis
        with open(file_path,'r', encoding='UTF-8', errors = 'ignore') as f:
            data = json.loads(f.read())
        df = pd.json_normalize(data, record_path =['rpeaks'])
    else: # mit2zive atvejis
        df = pd.read_json(file_path, orient = 'records')
    return df

def split_SubjCode(SubjCode):
    """
    Atnaujintas variantas, po to, kaip padaryti pakeitimai failų varduose 2022 03 26
    
    zive atveju: SubjCode = int(str(userNr) + str(registrationNr)), kur userNr >= 1000,
    pvz. SubjCode = 10001
    mit2zive atveju: SubjCode = userNr,  kur userNr < 1000,
    pvz. SubjCode = 101
    https://www.adamsmith.haus/python/answers/how-to-get-the-part-of-a-string-before-a-specific-character-in-python
    Parameters
    ------------
        SubjCode: int
    Return
    -----------
        userNr: int
        recordingNr: int
    """   
    if (SubjCode < 1000):
        userNr = SubjCode
        recordingNr = 0   
        return userNr, recordingNr
    else:        
        str_code = str(SubjCode) 
        chars = list(str_code)
        str1 =""
        userNr = int(str1.join(chars[:4]))
        str2 =""
        recordingNr = int(str2.join(chars[4:]))
        return userNr, recordingNr

def get_spike_width(orig, derivate, reample_points, positions):
    ret = pd.DataFrame(columns=["P_val", "Q_val", "R_val", "S_val", "T_val", "P_pos", "Q_pos", "R_pos", "S_pos", "T_pos", "QRS", "PR","ST","QT"])
    R = positions[0]
    asign = np.sign(derivate)
    signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
    Q = None
    for down in range(positions[0] - 1, 0, -1):
        if signchange[down] == 1:
            Q = down
            break
    S = None
    times_changed = 0
    for up in range(positions[0], reample_points, 1):
        if (signchange[up] == 1):
            if (times_changed == 1):
                S = up
                break
            else:
                times_changed += 1
    if (Q != None) & (S != None):
        QRS = math.fabs(S-Q)
        P=positions[1]
        T=positions[2]
        PR = math.fabs(Q-P)
        ST = math.fabs(T-S)
        QT = math.fabs(T-Q)
        row = {"P_val": orig[P], "Q_val":orig[Q], "R_val": orig[R], "S_val": orig[S], "T_val":orig[T],
                           "P_pos":P * 1./ reample_points, "Q_pos":Q * 1./ reample_points,
                           "R_pos":R * 1./ reample_points, "S_pos":S * 1./ reample_points,
                           "T_pos": T * 1./ reample_points,
                           "QRS":QRS * 1./ reample_points, "PR":PR * 1./ reample_points,
                           "ST":ST * 1./ reample_points, "QT":QT * 1./ reample_points}
        ret = pd.DataFrame(row, index=[0])
        return ret
    else:
        return pd.DataFrame()

def get_beat_features_fda(signal, atr_sample, idx):
    resapmling_points = 200
    fraction_to_drop_l = 0.7
    fraction_to_drop_r = 0.7
    samples = np.linspace(0, 1, resapmling_points)
    nl_RR =1
    nr_RR = 1
    keys_RR = []
    for tmp_l in range(nl_RR):
        keys_RR.append('RR_l_' + str(tmp_l))
    for tmp_l in range(nl_RR - 1):
        keys_RR.append('RR_l_' + str(tmp_l) + '/' + 'RR_l_' + str(tmp_l + 1))
    for tmp_r in range(nr_RR):
        keys_RR.append('RR_r_' + str(tmp_r))
    for tmp_r in range(nr_RR):
        keys_RR.append('RR_r_' + str(tmp_r) + '/' + 'RR_r_' + str(tmp_r + 1))

    train_set_stats = pd.DataFrame()
    train_set_points = pd.DataFrame()
    Rythm_Data = pd.DataFrame()
    omit_idx = pd.DataFrame()

    RRl_arr, RRr_arr = read_RR_arr_from_signal(atr_sample, idx, nl_steps=1, nr_steps=1)
    wl_side = math.floor(RRl_arr[0] * fraction_to_drop_l)  # pakeista all_beats_attr.loc[idx][5] į RRl, kj
    wr_side = math.floor(RRr_arr[0] * fraction_to_drop_r)  # pakeista all_beats_attr.loc[idx][6] į RRr, kj
    
    seq_1d = read_seq_from_signal(signal, atr_sample, idx, wl_side, wr_side)

    RPT = []
    dictRR = {}
    for tmp_l in range(nl_RR):
        dictRR[keys_RR[tmp_l]] = RRl_arr[tmp_l]
    for tmp_l in range(nl_RR - 1):
        dictRR[keys_RR[tmp_l + nl_RR]] = RRl_arr[tmp_l] / RRl_arr[tmp_l + 1]
        #keys_RR.append('RR_l_' + str(tmp_l) + '/' + 'RR_l_' + str(tmp_l + 1))
    for tmp_r in range(nr_RR):
        dictRR[keys_RR[tmp_r + 2 * nl_RR - 1]] = RRr_arr[tmp_r]
    for tmp_r in range(nr_RR - 1):
        dictRR[keys_RR[tmp_r + 3 * nl_RR - 1]] = RRr_arr[tmp_r] / RRr_arr[tmp_r + 1]
    dictRR['RR_r/RR_l'] = RRl_arr[tmp_r] / RRr_arr[tmp_l]
    
    fd = FDataGrid(data_matrix=seq_1d)

    fdd = fd.derivative()
    fd = fd.evaluate(samples)
    fd = fd.reshape(resapmling_points)

    fdd = fdd.evaluate(samples)
    fdd = fdd.reshape(resapmling_points)

    RPT.append(fd.argmax()) #0-th  element is R

    indexes_lower = scipy.signal.argrelextrema(fdd[math.floor(RPT[0]*0.5):RPT[0] - math.floor(RPT[0]*0.05)], comparator=np.greater, order=3)
    indexes_lower = tuple([math.floor(RPT[0]*0.5) + 1 + x for x in indexes_lower])
    values_lower = fd[indexes_lower]
    ind_low = np.argpartition(values_lower, -1)[-1:]
    ind_low[::-1].sort()

    indexes_upper = scipy.signal.argrelextrema(fdd[RPT[0] + 2:math.floor(RPT[0]*1.8)], comparator=np.greater, order=3) + RPT[0] + 2
    values_upper = fd[indexes_upper[0]]
    ind_up = np.argpartition(values_upper, -1)[-1:]
    tmp1 = np.sort(ind_up)

    if (len(ind_low)>=1) & (len(ind_up)>=1):

# I. Rythm_Data: 'P_val', 'Q_val', 'R_val', 'S_val', 'T_val', 'P_pos', 'Q_pos', 'R_pos',
#  'S_pos', 'T_pos', 'QRS', 'PR', 'ST', 'QT' - viso 14 požymių

        RPT.append(indexes_lower[0][ind_low[0]])  # 1-st is P
        RPT.append(indexes_upper[0][tmp1[0]]) # 2nd is T
    
        Rythm_Data = get_spike_width(fd, fdd, resapmling_points, RPT)
        if not Rythm_Data.empty:

# II. train_set_stats": 'idx', 'seq_size', 'RR_l_0', 'RR_r_0', 'RR_r/RR_l', 'wl_side', 'wr_side',
# 'signal_mean', 'signal_std' - viso 9 požymiai

            dict_full ={'idx': idx, 'seq_size': seq_1d.shape[0]}
            dict_full = dict(dict_full, **dictRR)
            dict_full.update({'wl_side': wl_side, 'wr_side': wr_side, "signal_mean": np.mean(fd), "signal_std": np.std(fd)})
            # print(f'dict: {dict_full}')
            train_set_stats = pd.DataFrame(dict_full, index=[0])

# III. train_set_points: '0', '1', '2', ... , '197', '198', '199' - viso 200 požymių
            
            train_set_points = pd.Series(fd).to_frame().T
        else:
            omit_idx = pd.DataFrame({'idx': idx}, index=[0])
    else:
        omit_idx = pd.DataFrame({'idx': idx}, index=[0])

    train_set_stats = pd.concat([train_set_stats, Rythm_Data, train_set_points], axis=1)
    return train_set_stats, omit_idx


def get_beat_features_fda_set(signal, df_rpeaks, idx_lst):
# Apskaičiuojami užduotų EKG signalo pūpsnių (per idx_lst) požymiai ir iš jų
# suformuojamas požymių dataframe masyvas, pridedant 'label'

    # all_beats pritaikytas tiek MIT2ZIVE, tiek ZIVE duomenims
    all_beats = {'N':0,'R':0, 'L':0, 'e':0, 'j':0, 'A':1,'a':1, 'J':1, 'S':1, 'V':2, 'E':2, 'F':3, 'U':3, 'Q':3}

    beat_features_set = pd.DataFrame()
    omit_idx_set = pd.DataFrame()

    atr_sample = df_rpeaks['sampleIndex'].to_numpy()
    atr_symbol = df_rpeaks['annotationValue'].to_numpy()
    # Jei pasitaiko symbol 'U' arba 'F', pūpsniui suteikiame klasę 3, kurią vėliau apvalysime  
    test_labels = np.array([all_beats[symbol] for symbol in atr_symbol])
    
    # print("\nGet beat features set from signal:")
    for idx in idx_lst:
        beat_features, omit_idx = get_beat_features_fda(signal, atr_sample, idx)
        # beat_features_set = beat_features_set.append(beat_features, ignore_index=True)
        if omit_idx.empty:
            beat_features['label'] = test_labels[idx]
            beat_features_set = pd.concat([beat_features_set, beat_features])
        else:
            omit_idx_set = pd.concat([omit_idx_set, omit_idx])
    
    
    # Konvertuojame int pozymius į float64
    beat_features_set['RR_l_0'] = beat_features_set['RR_l_0'].astype(float)
    beat_features_set['RR_r_0'] = beat_features_set['RR_r_0'].astype(float)

    return beat_features_set, omit_idx_set


pd.set_option("display.max_rows", 6000, "display.max_columns", 200)
pd.set_option('display.width', 1000)

import warnings
# warnings.filterwarnings("ignore")

 # Išvedamų požymių sąrašas 
selected_features = ['seq_size','RR_l_0', 'RR_r_0', 'RR_r/RR_l', 'wl_side','wr_side',
                'signal_mean', 'signal_std', 'P_val', 'Q_val', 'R_val', 'S_val', 'T_val',
                'P_pos', 'Q_pos', 'R_pos', 'S_pos', 'T_pos', 'QRS', 'PR', 'ST', 'QT', '0', '1', '2',
                '3', '4', '5', '199']

db_path = Path.cwd()
print(db_path)

SubjCodes = [100, 10021]
print(SubjCodes)

for SubjCode in SubjCodes:
    sign_raw = read_rec(db_path, SubjCode)
    signal_length = sign_raw.shape[0]
    signal = sign_raw

    # Surandame ir išvedame įrašo atributus
    userNr, recNr = split_SubjCode(SubjCode)
    # print(f"\nSubjCode: {SubjCode} userNr: {userNr:>2} file_name: {file_name:>2} signal_length: {signal_length}")
    print(f"\nSubjCode: {SubjCode} userNr: {userNr:>2}  signal_length: {signal_length}")

    df_rpeaks = read_df_rpeaks(db_path, SubjCode)

    # Užduodame pūpsnius, kuriems skaičiuosime požymius 
    idx_lst = [2, 3, 4, 5]
    print(idx_lst)

    train_set_data, omitted = get_beat_features_fda_set(signal, df_rpeaks, idx_lst)
    data_frame = train_set_data.set_index('idx')
    data_frame.columns = data_frame.columns.astype(str)

    # paruošiame atrinktų požymių masyvą spausdinimui
    data_frame_init = data_frame[selected_features]
    print("\ntrain_set_data:")
    print(data_frame_init.head())
    print("\nomitted:")
    print(omitted.head())

print("\nPabaiga")


