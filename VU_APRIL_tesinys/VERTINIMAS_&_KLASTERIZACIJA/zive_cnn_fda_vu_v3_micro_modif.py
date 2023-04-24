# Senesnis get_beat_features_fda_vu.py variantas, skaičiuojantis požymius užduotai pūpsnių
# sekai ind_lst
# Perdirbtas iš zive_cnn_fda_vu_v1.py, išmetant Rl/Rr santykį iš požymių

import pandas as pd
import numpy as np
import scipy.signal
from skfda import FDataGrid
import math
import keras, pickle, json
from pathlib import Path
from bitstring import BitArray

def zive_read_file_1ch(filename):
    f = open(filename, "r")
    a = np.fromfile(f, dtype=np.dtype('>i4'))
    ADCmax=0x800000
    Vref=2.5
    b = (a - ADCmax/2)*2*Vref/ADCmax/3.5*1000
    ecg_signal = b - np.mean(b)
    return ecg_signal

def zive_read_file_3ch(filename):
    f = open(filename, "r")
    a = np.fromfile(f, dtype=np.uint8)

    b = BitArray(bytes=a)
    d = np.array(b.unpack('intbe:32, intbe:24, intbe:24,' * int(len(a)/10)))
    # print (len(d))
    # printhex(d[-9:])

    ADCmax=0x800000
    Vref=2.5

    b = (d - ADCmax/2)*2*Vref/ADCmax/3.5*1000
    #b = d
    ch1 = b[0::3]
    ch2 = b[1::3]
    ch3 = b[2::3]
    start = 0#5000+35*200
    end = len(ch3)#start + 500*200
    ecg_signal = ch3[start:end]
    ecg_signal = ecg_signal - np.mean(ecg_signal)
    return ecg_signal



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


def get_beat_features_fda_vu_vasara_v1(signal, atr_sample, idx):
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

    # print(f'idx: {idx}')
    # print(f'get keys_RR: {keys_RR}')

    train_set_stats = pd.DataFrame()
    train_set_points = pd.DataFrame()
    Rythm_Data = pd.DataFrame()
    omit_idx = None
    # omit_idx = pd.DataFrame()

    RRl_arr, RRr_arr = read_RR_arr_from_signal(atr_sample, idx, nl_steps=1, nr_steps=1)
    wl_side = math.floor(RRl_arr[0] * fraction_to_drop_l)  # pakeista all_beats_attr.loc[idx][5] į RRl, kj
    wr_side = math.floor(RRr_arr[0] * fraction_to_drop_r)  # pakeista all_beats_attr.loc[idx][6] į RRr, kj
    
    # print(f'wl_side: {wl_side} wr_side: {wr_side}')

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
    # print(f'dictRR: {dictRR}')
    
    fd = FDataGrid(data_matrix=seq_1d)

    fdd = fd.derivative()
    fd = fd.evaluate(samples)
    fd = fd.reshape(resapmling_points)

    fdd = fdd.evaluate(samples)
    fdd = fdd.reshape(resapmling_points)

    RPT.append(fd.argmax()) #0-th  element is R
    # print(f'RPT: {RPT}')

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
    
        # print(f'RPT: {RPT}')

        Rythm_Data = get_spike_width(fd, fdd, resapmling_points, RPT)
        # print(f'get Rythm_Data {Rythm_Data}')

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
            # omit_idx = pd.DataFrame({'idx': idx}, index=[0])
            omit_idx = idx
    else:
        # omit_idx = pd.DataFrame({'idx': idx}, index=[0])
        omit_idx = idx

    # print(f'\ntrain_set_stats:')
    # print(train_set_stats.head())
    
    # print(f'\nRythm_Data:')
    # print(Rythm_Data.head())
    
    # print(f'\ntrain_set_points:')
    # print(train_set_points.head())
    
    # print(f'\nomit_idx:')
    # print(omit_idx.head())


    train_set_stats = pd.concat([train_set_stats, Rythm_Data, train_set_points], axis=1)

    # print(f'\ntrain_set_stats_united:')
    # print(train_set_stats.head())
    
    # print(f'\nomit_idx:')
    # print(omit_idx.head())

    return train_set_stats, omit_idx


def get_beat_features_set_fda_vu_vasara_v1(signal, atr_sample, idx_lst):
# Apskaičiuojami užduotų EKG signalo pūpsnių (per idx_lst) požymiai 

    beat_features_set = pd.DataFrame()
    omitted_lst = []

    # print("\nGet beat features set from signal:")
    for idx in idx_lst:
        beat_features, omit_idx = get_beat_features_fda_vu_vasara_v1(signal, atr_sample, idx)
        if omit_idx == None:
            beat_features_set = pd.concat([beat_features_set, beat_features])
        else:
            omitted_lst.append(omit_idx)

    # Konvertuojame int pozymius į float64
    beat_features_set['RR_l_0'] = beat_features_set['RR_l_0'].astype(float)
    beat_features_set['RR_r_0'] = beat_features_set['RR_r_0'].astype(float)

    beat_features_set = beat_features_set.set_index('idx')
    beat_features_set.columns = beat_features_set.columns.astype(str)

    return beat_features_set, omitted_lst

def get_normalized_data(data_frame, model_dir, all_features):
       # Suformuojame indeksų sąrašą. Formuodami sąrašą eliminuojame pirmą ir paskutinį indeksą
    
    # Nuskaitome scaler objectą
    path_scaler = Path(model_dir, 'scaler.pkl')  
    scaler = pickle.load(open(path_scaler,'rb'))
  
    # paruošiame požymių masyvą 
    data_frame = data_frame[all_features]
  
    data_array = scaler.transform(data_frame)
    # print('data_array.shape:', data_array.shape)

    return data_array


def predict_cnn_fda_vu_vasara_v2(test_x, model_dir):
# ************************************* funkcijai ***************************************************************

#  Iėjimo parametrai:   data_frame, atr_sample
#                       
#  Išėjimo parametrai: pred_y, kai kurie neatpažinti : 'U':3
    
    x_test = test_x.reshape((test_x.shape[0], test_x.shape[1], 1))

    # nuskaitome modelio parametrus
    model_path = Path(model_dir, 'best_model_final_2.h5')
    model = keras.models.load_model(model_path)
    
    # Pūpsnių klasių atpažinimas
    predictions = model.predict(x_test, verbose=0)
    predictions_y = np.argmax(predictions, axis=1)
    return predictions_y


def get_pred_labels_modif(predictions, data_frame_index, omitted_lst, atr_sample):
    
    if (len(data_frame_index) != len(predictions)):
        print("klaida funkcijoje get_pred_labels!")

    # Sužymimi neatpažinti pūpsniai - klasė 3:'U'
    pred_y = np.zeros(len(atr_sample), dtype=int)
    pred_y[0] = 3
    pred_y[len(atr_sample)-1] = 3

    if (len(omitted_lst) != 0):
        pred_y[omitted_lst] = 3

# Sužymimi atpažinti pūpsniai 
    pred_y[data_frame_index] = predictions
    
    # (unique,counts) = np.unique(pred_y, return_counts=True)
    # total = counts.sum()
    # print("pred_y:", unique, counts, total)

    return pred_y


def classify_cnn_fda_vu_vasara_v2_modif(signal, atr_sample, model_dir, all_features):
    # Tai vedantysis skriptas, į kurį kreipiasi analyse.py

    # Šitas gabaliukas ateina iš predict_cnn_fda_vu_vasara_v1
    idx_lst = list(range(1, len(atr_sample)-1))
    # print("len(atr_sample):",  len(atr_sample))
    
    # Formuojame iš pūpsnių požymių masyvą
    data_frame, omitted = get_beat_features_set_fda_vu_vasara_v1(signal, atr_sample, idx_lst)
    # Surandame pūpsnių indeksus, kuriems požymiai apaskaičiuoti, ir kuriems - ne 
    data_frame_index = data_frame.index
    omitted_index = pd.Index(omitted)
    
    # print("omitted.index:", list(omitted_index))
    # print("0", data_frame.head())
    # print("0", data_frame.tail())

    # Normalizuojame masyvą su apskaičiuotais požymiais
    data_array = get_normalized_data(data_frame, model_dir, all_features)

    # Atpažistame pūpsnių ML klases
    predictions = predict_cnn_fda_vu_vasara_v2(data_array, model_dir)

    # Suformuojame pūpsnių ML anotacijas  
    pred = get_pred_labels_modif(predictions, data_frame_index, omitted_index, atr_sample)
    
    return pred



