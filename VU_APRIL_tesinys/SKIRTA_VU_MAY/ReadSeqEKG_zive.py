import pandas as pd
import numpy as np
from pathlib import Path
import sys, os
import math
import random

import scipy.signal
import skfda
from skfda import FDataGrid
from skfda.preprocessing.smoothing import BasisSmoother
from skfda.representation.basis import BSpline, Fourier
from matplotlib import pyplot as plt


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


def get_seq_start_end(signal_length, i_sample, window_left_side, window_right_side):
    # Nustatome išskiriamos EKG sekos pradžią ir pabaigą
    seq_start = i_sample - window_left_side
    seq_end = i_sample + window_right_side
    if (seq_start < 0 or seq_end > signal_length):
        # print("\nseq_start: ", seq_start, " seq_end: ", seq_end)
        return (None, None)
    else:
        return (seq_start, seq_end)



def read_RR_arr(all_beats_attr, idx, nl_steps=0, nr_steps=0):  
    # Pateikia seką RRl_arr iš nl_steps RR reikšmių tarp iš eilės einančių R dantelių į kairę nuo einamo R dantelio
    # ir seką RRr_arr nr_steps RR reikšmių tarp iš eilės einančių R dantelių į dešinę nuo einamo R dantelio dantelio.
    # Seka iš kairės RRl_arr prasideda nuo tolimiausio nuo R dantelio atskaitymo ir jai pasibaigus,
    # toliau ją pratesia RRl_arr.

    SubjCode = get_SubjCode(idx, all_beats_attr)

    #  **************************** Tikrinimai ******************************************************************
    # Tikriname, ar skaičiuodami RR neišeisime už all_beats_attr ribų
    if (idx + nr_steps) >= len(all_beats_attr):
        txt = f"Klaida 1! idx, nl_steps: {idx}, {nr_steps} Skaičiuojant RR viršijama pūpsnių atributo masyvo riba."
        raise Exception(txt)
        # Reikia mažinti nr_steps arba koreguoti viršutinę idx ribą

    if ((idx - nl_steps) < 0):
        txt = f"Klaida 2! idx, nl_steps: {idx}, {nl_steps} Skaičiuojant RR išeinama už pūpsnių atributo masyvo ribų."
        raise Exception(txt)
        # Reikia mažinti nl_steps arba didinti apatinę idx ribą

    # Tikriname, ar skaičiuodami RR neišeisime už EKG įrašo SubjCode ribų
    SubjCodeRight = get_SubjCode(idx + nr_steps, all_beats_attr)
    if (SubjCodeRight != SubjCode):
        txt = f"Klaida 3! idx, nl_steps: {idx}, {nr_steps} Skaičiuojant RR viršijama EKG įrašo {SubjCode} viršutinė riba."
        raise Exception(txt)
        # Reikia mažinti nr_steps arba mažinti max idx

    SubjCodeLeft = get_SubjCode(idx - nl_steps, all_beats_attr)
    if (SubjCodeLeft != SubjCode):
        txt = f"Klaida 4! idx, nl_steps: {idx}, {nl_steps} Skaičiuojant RR išeinama už EKG įrašo {SubjCode} apatinės ribos."
        raise Exception(txt)
        # Reikia koreguoti nl_steps arba koreguoti apatinę idx ribas

    # **************************** Tikrinimų pabaiga ******************************************************************

    # Suformuojame RR sekas kairėje ir dešinėje idx atžvilgiu
    if (nl_steps != 0):
        RRl_arr = np.zeros(shape=(nl_steps), dtype=int)
        for i in range(nl_steps):
            RRl_arr[nl_steps - i - 1] = all_beats_attr.loc[idx - i, 'sample'] - all_beats_attr.loc[
                idx - i - 1, 'sample']
    else:
        RRl_arr = None

    if (nr_steps != 0):
        RRr_arr = np.zeros(shape=(nr_steps), dtype=int)
        for i in range(nr_steps):
            RRr_arr[i] = all_beats_attr.loc[idx + i + 1, 'sample'] - all_beats_attr.loc[idx + i, 'sample']
    else:
        RRr_arr = None

    return RRl_arr, RRr_arr

def read_seq_RR(rec_dir, all_beats_attr, idx, wl_side, wr_side):  
    # Nuskaito ir pateikia EKG seką apie R dantelį seq: reikšmiu kiekis wl_side - iš kairės pusės,
    # reikšmiu kiekis wr_side - iš dešinės pusės, R dantelio vietą EKG įraše sample,
    # ir atitinkamo pūpsnio klasės numerį label: 0, 1, 2.

    row = all_beats_attr.loc[idx]
    SubjCode = get_SubjCode(idx, all_beats_attr)

    file_path = Path(rec_dir, str(SubjCode) + '.npy')

    signal = np.load(file_path, mmap_mode='r')
    signal_length = signal.shape[0]
    (seq_start, seq_end) = get_seq_start_end(signal_length, row['sample'], wl_side, wr_side)

    # **************************** Tikrinimai ******************************************************************
    # Tikriname, ar sekos langas neišeina už įrašo ribų
    if (seq_start == None or seq_end == None):
        raise Exception(f"Klaida! {idx}: Sekos lango rėžiai už EKG įrašo {SubjCode} ribų.")
        # Reikia mažinti wl_side ar wr_side, arba koreguoti idx ribas
    else:
        seq = signal[seq_start:seq_end]
        sample = row['sample']
        label = row['label']

    return seq, sample, label




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


def apply_FDA_signal_v2(rec_dir, train_set_idx, all_beats_attr):
        # //////////////////////////////////////////////////////////////////////////////////////////// pakeitimas kj
    # randomlist = random.sample(range(0, len(all_beats_attr)), 3)
    #basis = BSpline(n_basis=40, domain_range=(0,1), order=3)
    # basis = Fourier(n_basis=70, domain_range=(0,1))
    #smoother = BasisSmoother(basis = basis, return_basis=True, method='svd')
    count = len(train_set_idx)
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
    for idx in train_set_idx.index:
        #idx =7883

        # //////////////////////////////////////////////////////////////////////////////////////////// pakeitimas kj
        RRl, RRr = read_RR_arr(all_beats_attr, idx, nl_steps=nl_RR, nr_steps=nr_RR)
        wl_side = math.floor(RRl[0] * fraction_to_drop_l)  # pakeista all_beats_attr.loc[idx][5] į RRl, kj
        wr_side = math.floor(RRr[0] * fraction_to_drop_r)  # pakeista all_beats_attr.loc[idx][6] į RRr, kj
        seq_1d, sample, label = read_seq_RR(rec_dir, all_beats_attr, idx, wl_side, wr_side)

        # //////////////////////////////////////////////////////////////////////////////////////////// pakeitimas kj

        RPT = []
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


        if (label != None) & (len(ind_low)>=1) & (len(ind_up)>=1):
            RPT.append(indexes_lower[0][ind_low[0]])  # 1-st is P
            RPT.append(indexes_upper[0][tmp1[0]]) # 2nd is T

            tmp = get_spike_width(fd, fdd, resapmling_points, RPT)
            if not tmp.empty:
                Rythm_Data = Rythm_Data.append(tmp,ignore_index=True)
                print("\n")
                print("Pprocessing -- %d; idx -- %d" % (count, idx))
                count -= 1

                dict_full ={'idx': idx, 'seq_size': seq_1d.shape[0], 'label': label}
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



my_os = sys.platform
print("OS in my system : ", my_os)

if my_os != 'linux':
    OS = 'Windows'
else:
    OS = 'Ubuntu'

# Pasiruošimas

# Bendras duomenų aplankas, kuriame patalpintas subfolderis name_db

if OS == 'Windows':
    # Duomenu_aplankas = 'C:\DI\Data\MIT&ZIVE'  # variantas: Windows
    Duomenu_aplankas = os.getcwd()  # variantas: Windows

else:
    Duomenu_aplankas = '/home/povilas/Documents/kardio'  # arba variantas: UBUNTU, be Docker

# jei variantas Docker pasirenkame:
# Duomenu_aplankas = '/Data/MIT&ZIVE'

print(Duomenu_aplankas)

#  MIT2ZIVE duomenų aplankas
db_folder = 'DUOM_SKIRTI_VU'

# Nuoroda į DUOM_TST duomenų aplanką
db_path = Path(Duomenu_aplankas, db_folder)

# Anotacijoms priskirtos klasės
selected_beats = {'N': 0, 'S': 1, 'V': 2}

print("\nBendras MIT ir Zive duomenu aplankas: ", Duomenu_aplankas)
print("MIT2ZIVE EKG irasu aplankas: ", db_folder)


from scipy import stats
import datetime

# Įvairios operacijos, naudojant EKG įrašus ir all_beats_attr.csv

 # Nuskaitome pūpsnių atributų failą
file_path = Path(db_path, 'all_beats_attr_z.csv')
all_beats_attr = pd.read_csv(file_path, index_col=0)


beats_skiped = 1
#create training set:
train_set_idx = pd.read_csv(Path(db_path, 'train_ind_lst_sutrumpintas.csv'), header = None, index_col=0)
train_set_data, omitted = apply_FDA_signal_v2(db_path, train_set_idx, all_beats_attr)
#train_set_data, omitted = get_raw_data(train_set_idx, all_beats_attr)
train_set_data.to_csv(Path(db_path, 'train_data_RR_1_MIT.csv'), index = False)
omitted.to_csv(Path(db_path, 'train_data_omitted_idx_RR_1.csv'), index = False)

# #create validation set:
# test_set_idx = pd.read_csv(Path(db_path, 'validate_ind_lst.csv'), header = None, index_col=0)
# test_set_data, omitted = apply_FDA_signal_v2(db_path, test_set_idx, all_beats_attr)
# #test_set_data, omitted = get_raw_data(test_set_idx, all_beats_attr)
# test_set_data.to_csv(Path(db_path, 'test_data_RR_1_MIT.csv'), index = False)
# omitted.to_csv(Path(db_path, 'test_data_omitted_idx_RR_1.csv'), index = False)


print("\nPabaiga"), 