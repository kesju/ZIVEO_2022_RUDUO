# Perdarytas iš ReadSeqEKG_2022_10_05_v2.py, tam kad perdaryti pūpsnių klasifikaciją su su micro moduliu. 
# 
# Skriptas užduoto ilgio sekų skaitymui iš MIT2ZIVE EKG įrašų ir požymių skaičiavimui.
# Skriptą atsiuntė Povilas 2022 10 05, tam kad išsiaiškinti, kodėl nesutampa pavasario ir vasaros
# požymių ilgiai. Pasirodė, kad požymių skaičius pasikeitė, nes 'vasaros' požymiuose dingo požymis
# Rl/Rr.
# Atsiųstas skriptas dar dirba su senesniu duomenų variantu:
# Nuoroda į aplanką su EKG įrašais (.npa) ir anotacijomis (.json) - dar nepataisyti įrašai npa į npy
# db_path = Path(db_path, 'records')  # all_beats_attr šiame aplanke dar su apskaičiuotais RR1 ir RR2

# Šis skriptas perdarytas darbui su naujesniais duomenimis:  all_beats_attr jau be apskaičiuotų RR1 ir RR2
# ir naudoja npy failus. apply_FDA perpavadintas į apply_FDA_vasara


import pandas as pd
import numpy as np
from pathlib import Path
import sys, json
import math
import random

import scipy.signal
import skfda
from skfda import FDataGrid
from skfda.preprocessing.smoothing import BasisSmoother
from skfda.representation.basis import BSpline, Fourier
from matplotlib import pyplot as plt
from zive_cnn_fda_vu_v1 import get_seq_start_end
from zive_cnn_fda_vu_v1 import read_seq_from_signal
# from zive_cnn_fda_vu_v1 import read_RR_arr_from_signal
from zive_util_vu import get_filename, split_SubjCode 
from zive_util_vu import zive_read_df_rpeaks

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

def create_SubjCode(userNr, recordingNr):
    # SubjCode = userNr + recordingNr
    # pvz. SubjCode = 10002
    if (userNr < 1000):  # duomenys mit2zive
        return userNr
    else:  # duomenys zive
        str_code = str(userNr) + str(recordingNr)
        SubjCode = int(str_code)
        return SubjCode


def get_SubjCode(idx, all_beats_attr):
    row = all_beats_attr.loc[idx]
    SubjCode = create_SubjCode(row['userNr'], row['recordingNr'])
    return SubjCode

def read_rec(rec_dir, SubjCode):
    file_path = Path(rec_dir, str(SubjCode) + '.npa')
    signal = np.load(file_path, mmap_mode='r')
    # print(f"SubjCode: {SubjCode}  signal.shape: {signal.shape}")
    return signal

def read_rec_attrib(rec_dir, SubjCode):
    # Pritaikyta nuskaityti json informaciją tiek mit2zive, tiek zive atvejams
    file_path = Path(rec_dir, str(SubjCode) + '.json')

    if (SubjCode > 1000): # zive atvejis
        with open(file_path,'r', encoding='UTF-8', errors = 'ignore') as f:
            data = json.loads(f.read())
        df = pd.json_normalize(data, record_path =['rpeaks'])
    else: # mit2zive atvejis
        df = pd.read_json(file_path, orient = 'records')

    atr_sample = df['sampleIndex'].to_numpy()
    atr_symbol = df['annotationValue'].to_numpy()
    return atr_sample, atr_symbol

def get_spike_width(orig, derivate, reample_points, positions):
    ret = pd.DataFrame(columns=["P_val", "Q_val", "R_val", "S_val", "T_val", "P_pos", "Q_pos", "R_pos", "S_pos", "T_pos", "QRS", "PR","ST","QT"])
    R = positions[0]
    asign = np.sign(derivate)
    signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
    #plt.figure()
    #plt.plot(signchange)
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
        ret = ret. append({"P_val": orig[P], "Q_val":orig[Q], "R_val": orig[R], "S_val": orig[S], "T_val":orig[T],
                           "P_pos":P * 1./ reample_points, "Q_pos":Q * 1./ reample_points,
                           "R_pos":R * 1./ reample_points, "S_pos":S * 1./ reample_points,
                           "T_pos": T * 1./ reample_points,
                           "QRS":QRS * 1./ reample_points, "PR":PR * 1./ reample_points,
                           "ST":ST * 1./ reample_points, "QT":QT * 1./ reample_points}, ignore_index=True)
        return ret
    else:
        return pd.DataFrame()


# def apply_FDA_vasara(train_set_idx, all_beats_attr):
def apply_FDA_vasara(signal, idx_lst, atr_sample):
    # randomlist = random.sample(range(0, len(all_beats_attr)), 3)
    #basis = BSpline(n_basis=40, domain_range=(0,1), order=3)
    # basis = Fourier(n_basis=70, domain_range=(0,1))
    #smoother = BasisSmoother(basis = basis, return_basis=True, method='svd')
    # count = len(train_set_idx)
    count = len(idx_lst)
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
    #train_set_stats = pd.DataFrame(columns=['idx', 'seq_size', 'label', 'RRl', 'RRr', "RRl/RRr", 'wl_side', 'wr_side',
    #                                       "signal_mean", "signal_std"])
    #train_set_stats = pd.DataFrame(columns=['idx', 'seq_size', 'label', keys_RR, 'wl_side', 'wr_side',
    #                                       "signal_mean", "signal_std"])

    train_set_points = pd.DataFrame()
    Rythm_Data = pd.DataFrame()
    omit_idx = pd.DataFrame()

    # for idx in train_set_idx.index:
    for idx in idx_lst:
        #idx =7883
         # ************************************* pakeitimas kj ************************************************
        # seq_1d, RRl, RRr = read_seq_RR(db_path, all_beats_attr, idx, wl_side, wr_side)
        RRl, RRr = read_RR_arr_from_signal(atr_sample, idx, nl_steps=1, nr_steps=1)
        wl_side = math.floor(RRl[0] * fraction_to_drop_l)  # pakeista all_beats_attr.loc[idx][5] į RRl, kj
        wr_side = math.floor(RRr[0] * fraction_to_drop_r)  # pakeista all_beats_attr.loc[idx][6] į RRr, kj
        seq_1d = read_seq_from_signal(signal, atr_sample, idx, wl_side, wr_side)

        # ************************************* pakeitimas ************************************************
        # wl_side = math.floor(all_beats_attr.loc[idx][5] * fraction_to_drop_l)
        # wr_side = math.floor(all_beats_attr.loc[idx][6] * fraction_to_drop_r)
        RPT = []
        # seq_1d, sample, label, RRl, RRr = read_seq_RR_arr(db_path, all_beats_attr, idx, wl_side, wr_side,nl_RR, nr_RR)
        dictRR = {}
        for tmp_l in range(nl_RR):
            dictRR[keys_RR[tmp_l]] = RRl[tmp_l]
        for tmp_l in range(nl_RR - 1):
            dictRR[keys_RR[tmp_l + nl_RR]] = RRl[tmp_l] / RRl[tmp_l + 1]
            #keys_RR.append('RR_l_' + str(tmp_l) + '/' + 'RR_l_' + str(tmp_l + 1))
        for tmp_r in range(nr_RR):
            dictRR[keys_RR[tmp_r + 2 * nl_RR - 1]] = RRr[tmp_r]
        for tmp_r in range(nr_RR - 1):
            dictRR[keys_RR[tmp_r + 3 * nl_RR - 1]] = RRr[tmp_r] / RRr[tmp_r + 1]
        dictRR['RR_r/RR_l'] = RRl[tmp_r] / RRr[tmp_l]
        
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


        # if (label != None) & (len(ind_low)>=1) & (len(ind_up)>=1):
        if (len(ind_low)>=1) & (len(ind_up)>=1):
            RPT.append(indexes_lower[0][ind_low[0]])  # 1-st is P
            RPT.append(indexes_upper[0][tmp1[0]]) # 2nd is T

            tmp = get_spike_width(fd, fdd, resapmling_points, RPT)
            if not tmp.empty:
                Rythm_Data = Rythm_Data.append(tmp,ignore_index=True)
                print("Processing -- %d; idx -- %d" % (count, idx))
                count -= 1

                dict_full ={'idx': idx, 'seq_size': seq_1d.shape[0], 'label': 0}
                dict_full = dict(dict_full, **dictRR)
                dict_full.update({'wl_side': wl_side, 'wr_side': wr_side, "signal_mean": np.mean(fd), "signal_std": np.std(fd)})

                train_set_stats = train_set_stats.append(dict_full,
                                                       ignore_index=True)

                train_set_points = train_set_points.append(pd.Series(fd), ignore_index=True)
            else:
                omit_idx = omit_idx.append({'idx': idx}, ignore_index=True)
        else:
            omit_idx = omit_idx.append({'idx' : idx}, ignore_index=True)
    train_set_stats = pd.concat([train_set_stats, Rythm_Data, train_set_points], axis=1)
    return train_set_stats, omit_idx


def get_beat_features_fda_vu_v1_vasara(signal, atr_sample, idx):
    # count = len(train_set_idx)
    # count = len(idx_lst)
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

# II. train_set_stats": 'idx', 'seq_size', 'label', 'RRl', 'RRr', 'RRl/RRr', 'wl_side', 'wr_side',
# 'signal_mean', 'signal_std' - viso 10 požymių

            dict_full ={'idx': idx, 'seq_size': seq_1d.shape[0], 'label': 0}
            dict_full = dict(dict_full, **dictRR)
            dict_full.update({'wl_side': wl_side, 'wr_side': wr_side, "signal_mean": np.mean(fd), "signal_std": np.std(fd)})
            train_set_stats = pd.DataFrame(dict_full)

# III. train_set_points: '0', '1', '2', ... , '197', '198', '199' - viso 200 požymių
            
            train_set_points = pd.Series(fd).to_frame().T
        else:
            omit_idx = pd.DataFrame([{'idx': idx}])
    else:
        omit_idx = pd.DataFrame([{'idx': idx}])

    train_set_stats = pd.concat([train_set_stats, Rythm_Data, train_set_points], axis=1)
    return train_set_stats, omit_idx

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

def get_beat_features_set_fda_vu_v1(signal, df_rpeaks, idx_lst):
# Apskaičiuojami užduotų EKG signalo pūpsnių (per idx_lst) požymiai ir iš jų
# suformuojamas požymių dataframe masyvas

    all_beats =  {'N':0, 'S':1, 'V':2, 'U':3, 'F':3}
    
    beat_features_set = pd.DataFrame()
    omit_idx_set = pd.DataFrame()

    atr_sample = df_rpeaks['sampleIndex'].to_numpy()
    atr_symbol = df_rpeaks['annotationValue'].to_numpy()
    # Jei pasitaiko symbol 'U' arba 'F', pūpsniui suteikiame klasę 3, kurią vėliau apvalysime  
    test_labels = np.array([all_beats[symbol] for symbol in atr_symbol])
    
    # print("\nGet beat features set from signal:")
    for idx in idx_lst:
        beat_features, omit_idx = get_beat_features_fda_vu_v1_vasara(signal, atr_sample, idx)
        # beat_features_set = beat_features_set.append(beat_features, ignore_index=True)
        if omit_idx.empty:
            beat_features['label'] = test_labels[idx]
            beat_features_set = pd.concat([beat_features_set, beat_features])
        else:
            omit_idx_set = pd.concat([omit_idx_set, omit_idx])

    # Konvertuojame int pozymius į float64
    beat_features_set['RRl'] = beat_features_set['RRl'].astype(float)
    beat_features_set['RRr'] = beat_features_set['RRr'].astype(float)

    return beat_features_set, omit_idx_set

# pd.set_option("display.max_rows", 6000, "display.max_columns",200)
# pd.set_option('display.width', 1000)

import warnings
# warnings.filterwarnings("ignore")

my_os = sys.platform
print("OS in my system : ", my_os)

if my_os != 'linux':
    OS = 'Windows'
else:
    OS = 'Ubuntu'

# Pasiruošimas

# Bendras duomenų aplankas, kuriame patalpintas subfolderis name_db

if OS == 'Windows':
    # Duomenu_aplankas = "F:\DI\Data\MIT&ZIVE"   # variantas: Windows
    Duomenu_aplankas = 'D:\DI\Data\MIT&ZIVE'   # variantas: Windows
else:
    Duomenu_aplankas = '/home/povilas/Documents/kardio'  # arba variantas: UBUNTU, be Docker

# jei variantas Docker pasirenkame:
# Duomenu_aplankas = '/Data/MIT&ZIVE'

#  MIT2ZIVE duomenų aplankas
db_folder = 'MIT2ZIVE'

# Nuoroda į DUOM_TST duomenų aplanką
db_path = Path(Duomenu_aplankas, db_folder)

# Nuoroda į aplanką su EKG įrašais (.npa) ir anotacijomis (.json)
db_path = Path(db_path, 'records')  # all_beats_attr šiame aplanke dar su apskaičiuotais R

# Anotacijoms priskirtos klasės
selected_beats = {'N': 0, 'S': 1, 'V': 2}

print("\nBendras MIT ir Zive duomenų aplankas: ", Duomenu_aplankas)
print("MIT2ZIVE EKG įrašų aplankas: ", db_folder)


from scipy import stats
import datetime

# Įvairios operacijos, naudojant EKG įrašus ir all_beats_attr.csv

 # Nuskaitome pūpsnių atributų failą
file_path = Path(db_path, 'all_beats_attr.csv')
all_beats_attr = pd.read_csv(file_path, index_col=0)
#

# beats_skiped = 1
#create training set:
# train_set_idx = pd.read_csv(Path(db_path, 'train_ind_lst_tst.csv'), header = None, index_col=0)
# print(train_set_idx)

 # Nuskaitome EKG įrašą (npy formatu)

SubjCode = 100
sign_raw = read_rec(db_path, SubjCode)
signal_length = sign_raw.shape[0]
signal = sign_raw

# Surandame ir išvedame įrašo atributus
file_name = get_filename(db_path, SubjCode)
userNr, recNr = split_SubjCode(SubjCode)
print(f"\nSubjCode: {SubjCode} userNr: {userNr:>2} file_name: {file_name:>2} signal_length: {signal_length}")

# Filtruojame signalą
# signal = signal_filter(signal=sign_raw, sampling_rate=200, lowcut=0.2, method="butterworth", order=5)

# Nuskaitome paciento anotacijas ir jų indeksus

atr_sample, atr_symbol = read_rec_attrib(db_path, SubjCode)
print(atr_sample)

df_rpeaks = read_df_rpeaks(db_path, SubjCode)
print(df_rpeaks.head(10))

idx_lst = [2, 3, 4, 5, 6, 7, 8, 9, 10]
print(idx_lst)

train_set_data, omitted = get_beat_features_set_fda_vu_v1(signal, df_rpeaks, idx_lst)
# train_set_data, omitted = apply_FDA_vasara(signal, idx_lst, atr_sample)
# train_set_data, omitted = apply_FDA_vasara(train_set_idx, all_beats_attr)

print("\n", "train_set_data DATA:")
train_set_data.info()
print(list(train_set_data.columns))
# print(train_set_data.keys())
print("\n")
print(train_set_data.head())
print("\n")
# print(train_set_data['label'].value_counts())
# train_set_data.to_csv(Path(db_path, 'train_data_RR_1_MIT_v2.csv'), index = False)
# omitted.to_csv(Path(db_path, 'train_data_omitted_idx_RR_1_v2.csv'), index = False)

#create validation set:
# test_set_idx = pd.read_csv(Path(db_path, 'validate_ind_lst.csv'), header = None, index_col=0)
# test_set_data, omitted = apply_FDA_vasara_vasara(test_set_idx, all_beats_attr)
# test_set_data.to_csv(Path(db_path, 'test_data_RR_1_MIT.csv'), index = False)
# omitted.to_csv(Path(db_path, 'test_data_omitted_idx_RR_1.csv'), index = False)


print("\nPabaiga")