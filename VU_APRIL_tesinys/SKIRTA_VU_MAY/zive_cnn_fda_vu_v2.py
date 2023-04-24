import pandas as pd
import numpy as np
import scipy.signal
from skfda import FDataGrid
import math
import keras, pickle
from pathlib import Path
from bitstring import BitArray


# def zive_read_file_1ch(filename):
#     f = open(filename, "r")
#     a = np.fromfile(f, dtype=np.dtype('>i4'))
#     ADCmax=0x800000
#     Vref=2.5
#     b = (a - ADCmax/2)*2*Vref/ADCmax/3.5*1000
#     ecg_signal = b - np.mean(b)
#     return ecg_signal

# def zive_read_file_3ch(filename):
#     f = open(filename, "r")
#     a = np.fromfile(f, dtype=np.uint8)

#     b = BitArray(bytes=a)
#     d = np.array(b.unpack('intbe:32, intbe:24, intbe:24,' * int(len(a)/10)))
#     # print (len(d))
#     # printhex(d[-9:])

#     ADCmax=0x800000
#     Vref=2.5

#     b = (d - ADCmax/2)*2*Vref/ADCmax/3.5*1000
#     #b = d
#     ch1 = b[0::3]
#     ch2 = b[1::3]
#     ch3 = b[2::3]
#     start = 0#5000+35*200
#     end = len(ch3)#start + 500*200
#     ecg_signal = ch3[start:end]
#     ecg_signal = ecg_signal - np.mean(ecg_signal)
#     return ecg_signal





def read_RR_arr_from_signal(atr_sample, idx, nl_steps, nr_steps):
# Nuskaito ir pateikia EKG seką apie R dantelį seq: reikšmiu kiekis wl_side - iš kairės pusės, 
# reikšmiu kiekis wr_side - iš dešinės pusės, R dantelio vietą EKG įraše sample,
# ir atitinkamo pūpsnio klasės numerį label: 0, 1, 2.
# Taip pat pateikia seką RRl_arr iš nl_steps RR reikšmių tarp iš eilės einančių R dantelių į kairę nuo einamo R dantelio 
# ir seką RRr_arr nr_steps RR reikšmių tarp iš eilės einančių R dantelių į dešinę nuo einamo R dantelio dantelio.
# Seka iš kairės RRl_arr prasideda nuo tolimiausio nuo R dantelio atskaitymo ir jai pasibaigus,
# toliau ją pratesia RRr_arr. 

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
        # ret = ret. append({"P_val": orig[P], "Q_val":orig[Q], "R_val": orig[R], "S_val": orig[S], "T_val":orig[T],
        #                    "P_pos":P * 1./ reample_points, "Q_pos":Q * 1./ reample_points,
        #                    "R_pos":R * 1./ reample_points, "S_pos":S * 1./ reample_points,
        #                    "T_pos": T * 1./ reample_points,
        #                    "QRS":QRS * 1./ reample_points, "PR":PR * 1./ reample_points,
        #                    "ST":ST * 1./ reample_points, "QT":QT * 1./ reample_points}, ignore_index=True)   KJ: append pakeistas į concat

        pozym = pd.DataFrame([{"P_val": orig[P], "Q_val":orig[Q], "R_val": orig[R], "S_val": orig[S], "T_val":orig[T],
                           "P_pos":P * 1./ reample_points, "Q_pos":Q * 1./ reample_points,
                           "R_pos":R * 1./ reample_points, "S_pos":S * 1./ reample_points,
                           "T_pos": T * 1./ reample_points,
                           "QRS":QRS * 1./ reample_points, "PR":PR * 1./ reample_points,
                           "ST":ST * 1./ reample_points, "QT":QT * 1./ reample_points}])

        ret = pd.concat([ret, pozym], axis = 0)
        # ret.reset_index()
        return ret
    else:
        return pd.DataFrame()


def apply_FDA_modified(signal, idx_lst, atr_sample):
# def apply_FDA_modified(train_set_idx, all_beats_attr):
    # randomlist = random.sample(range(0, len(all_beats_attr)), 3)
    #basis = BSpline(n_basis=40, domain_range=(0,1), order=3)
    # basis = Fourier(n_basis=70, domain_range=(0,1))
    #smoother = BasisSmoother(basis = basis, return_basis=True, method='svd')
    count = len(idx_lst)
    resapmling_points = 200
    fraction_to_drop_l = 0.7
    fraction_to_drop_r = 0.7
    show_period = 100
    samples = np.linspace(0, 1, resapmling_points)
    train_set_stats = pd.DataFrame(columns=['idx', 'seq_size', 'label', 'RRl', 'RRr', "RRl/RRr", 'wl_side', 'wr_side', "signal_mean", "signal_std"])

    train_set_points = pd.DataFrame()
    Rythm_Data = pd.DataFrame()
    omit_idx = pd.DataFrame()

    i_count = 0  # Derinimui

    # print("\nProcessing:")
    for idx in idx_lst:
        #idx =7883
        # ************************************* pakeitimas kj ************************************************
        # seq_1d, RRl, RRr = read_seq_RR(db_path, all_beats_attr, idx, wl_side, wr_side)
        RRl_arr, RRr_arr = read_RR_arr_from_signal(atr_sample, idx, nl_steps=1, nr_steps=1)
        RRl = RRl_arr[0]
        RRr = RRr_arr[0]
        wl_side = math.floor(RRl * fraction_to_drop_l)  # pakeista all_beats_attr.loc[idx][5] į RRl, kj
        wr_side = math.floor(RRr * fraction_to_drop_r)  # pakeista all_beats_attr.loc[idx][6] į RRr, kj
        seq_1d = read_seq_from_signal(signal, atr_sample, idx, wl_side, wr_side)

        # ************************************* pakeitimas ************************************************

        RPT = []
        fd = FDataGrid(data_matrix=seq_1d)
        #fd_smooth = smoother.fit_transform(fd)
        #plt.figure()
        #plt.plot(fd_smooth)

        fdd = fd.derivative()
        fd = fd.evaluate(samples)  
        fd = fd.reshape(resapmling_points)

        fdd = fdd.evaluate(samples)
        fdd = fdd.reshape(resapmling_points)
        #plt.figure()
        #plt.plot(fd)
        #plt.show()
        #plt.figure()
        #plt.plot(fdd)
        #plt.show()
        RPT.append(fd.argmax()) #0-th  element is R
        #if idx == 2195:
        #    print ("stop")

        indexes_lower = scipy.signal.argrelextrema(fdd[math.floor(RPT[0]*0.5):RPT[0] - math.floor(RPT[0]*0.05)], comparator=np.greater, order=3)
        indexes_lower = tuple([math.floor(RPT[0]*0.5) + 1 + x for x in indexes_lower])
        values_lower = fd[indexes_lower]
        ind_low = np.argpartition(values_lower, -1)[-1:]
        ind_low[::-1].sort()

        indexes_upper = scipy.signal.argrelextrema(fdd[RPT[0] + 2:math.floor(RPT[0]*1.8)], comparator=np.greater, order=3) + RPT[0] + 2
        values_upper = fd[indexes_upper[0]]
        ind_up = np.argpartition(values_upper, -1)[-1:]
        tmp1 = np.sort(ind_up)

        #plt.plot(fdd)

        if (len(ind_low)>=1) & (len(ind_up)>=1):
        # if (label != None) & (len(ind_low)>=1) & (len(ind_up)>=1):  # Išmęstas label, kj
            RPT.append(indexes_lower[0][ind_low[0]])  # 1-st is P
            RPT.append(indexes_upper[0][tmp1[0]]) # 2nd is T

            tmp = get_spike_width(fd, fdd, resapmling_points, RPT)
            if not tmp.empty:
                # Rythm_Data = Rythm_Data.append(tmp,ignore_index=True)
                #   KJ: append pakeistas į concat
                Rythm_Data = pd.concat([Rythm_Data, tmp])
            #if (label != None) & (Rythm_Data['Q_pos'] != ) & (Rythm_Data["S_pos"] != None):
                #spike_width = (R_end - R_start) * 1. / resapmling_points
                
                # ///////////////////// Pakeitimas kj ////////////////////////////////////////////////////////
                # print("\n")
                # print("Pprocessing -- %d; idx -- %d" % (count, idx))

                # if (idx%show_period == 0):
                # if (idx < 10):
                #     print(" count,idx:", count, ",", idx, end =' ')
                # ///////////////////// Pakeitimas ////////////////////////////////////////////////////////

                count -= 1
                # train_set_stats = train_set_stats.append({'idx': idx, 'seq_size': seq_1d.shape[0], 'label': 0,
                #                                         'RRl': RRl, 'RRr': RRr, 'RRl/RRr': RRl / RRr,
                #                                         'wl_side': wl_side, 'wr_side': wr_side,
                #                                         "signal_mean": np.mean(fd), "signal_std": np.std(fd)},
                #                                        ignore_index=True)
                #   KJ: append pakeistas į concat 

                pozym = pd.DataFrame([{'idx': idx, 'seq_size': seq_1d.shape[0], 'label': 0,
                                                        'RRl': RRl, 'RRr': RRr, 'RRl/RRr': RRl / RRr,
                                                        'wl_side': wl_side, 'wr_side': wr_side,
                                                        "signal_mean": np.mean(fd), "signal_std": np.std(fd)}])

                train_set_stats = pd.concat([train_set_stats, pozym])
                #train_set_stats = train_set_stats.append(train_set_stats, ignore_index=False)
                # train_set_points = train_set_points.append(pd.Series(fd), ignore_index=True) KJ: append pakeistas į concat
                train_set_points = pd.concat([train_set_points, pd.DataFrame(fd)])

            else:
                # omit_idx = omit_idx.append({'idx': idx}, ignore_index=True)
                #  KJ: append pakeistas į concat
                pozym = pd.DataFrame([{'idx': idx}])
                omit_idx = pd.concat([omit_idx, pozym])
        else:
            # omit_idx = omit_idx.append({'idx' : idx}, ignore_index=True)
            #  KJ: append pakeistas į concat
            pozym = pd.DataFrame([{'idx': idx}])
            omit_idx = pd.concat([omit_idx, pozym])

    
        if (i_count == 3):
            print('\ntrain_set_stats:\n')
            print(train_set_stats)
            print('\nRythm_Data:\n')
            print(Rythm_Data)
            print('\ntrain_set_points:\n')
            print(train_set_points)
        i_count += 1

    # train_set_stats = pd.concat([train_set_stats, Rythm_Data, train_set_points], axis=1) KJ: append pakeistas į concat
    train_set_stats = pd.concat([train_set_stats, Rythm_Data, train_set_points], axis=0)
    
    print('\ntrain_set_stats jungtinis:\n')
    print(train_set_stats.head(3))
    
    return train_set_stats, omit_idx


def classify_cnn_fda_vu_v1(signal, atr_sample, model_dir, prediction_labels):
    #  Iėjimo parametrai:   signal - ekg reikšmių masyvas numpy
    #                       atr_sample - pūpsnių rpeak reikšmių sąrašas list 
    #                       model_dir - aplanko su modelio ir scaler_train parametrais vardas
    #                       prediction_labels - klasių simbolinių vardų sąrašas list ['N', 'S', 'V', 'U']                  
    #  Išėjimo parametrai: classification - pūpsniams priskirtų klasių simbolinių vardų sąrašas list

    pred_y = predict_cnn_fda_vu_v1(signal, atr_sample, model_dir)
    classification = get_pred_symbols(pred_y, atr_sample, prediction_labels)
    return classification


def get_pred_symbols(pred_y, atr_sample, prediction_labels):
    classification=[]
    for i, i_sample in enumerate(atr_sample):
        # if (pred_y[i] != 3):
        pred_symb = prediction_labels[pred_y[i]]
        classification.append({'sample':i_sample, 'annotation':pred_symb})    
    return classification


def predict_cnn_fda_vu_v1(signal, atr_sample, model_dir):
# ************************************* funkcijai ***************************************************************
    #  Iėjimo parametrai:   signal - ekg reikšmių masyvas numpy
    #                       atr_sample - pūpsnių rpeak reikšmių (indeksų) sąrašas list 
    #                       model_dir - aplanko su modelio ir scaler_train parametrais vardas
    #                       
    #  Išėjimo parametrai: pred_y -  pūpsniams priskirtų klasių numerių sąrašas numpy, kurie neatpažinti == 3

    # Suformuojame indeksų sąrašą. Formuodami sąrašą eliminuojame pirmą ir paskutinį indeksą

    all_features = ['RRl', 'RRr', 'RRl/RRr',
                'signal_mean', 'signal_std', 'P_val', 'Q_val', 'R_val', 'S_val', 'T_val',
                'P_pos', 'Q_pos', 'R_pos', 'S_pos', 'T_pos', 'QRS', 'PR', 'ST', 'QT', '0', '1', '2',
                '3', '4', '5', '6', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32',
                '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46',
                '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60',
                '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74',
                '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88',
                '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', '100', '101', '102',
                '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114',
                '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '125', '126',
                '127', '128', '129', '130', '131', '132', '133', '134', '135', '136', '137', '138',
                '139', '140', '141', '142', '143', '144', '145', '146', '147', '148', '149', '150',
                '151', '152', '153', '154', '155', '156', '157', '158', '159', '160', '161', '162',
                '163', '164', '165', '166', '167', '168', '169', '170', '171', '172', '173', '174',
                '175', '176', '177', '178', '179', '180', '181', '182', '183', '184', '185', '186',
                '187', '188', '189', '190', '191', '192', '193', '194', '195', '196', '197', '198',
                '199']
   
    # nuskaitome modelio parametrus
    model_path = Path(model_dir, 'best_model_final_2.h5')
    model = keras.models.load_model(model_path)

    # Nuskaitome scaler objectą
    path_scaler = Path(model_dir, 'scaler.pkl')  
    scaler = pickle.load(open(path_scaler,'rb'))

    idx_lst = list(range(1, len(atr_sample)-1))

    # Formuojame iš pūpsnių požymių masyvą
    data_frame, omitted = apply_FDA_modified(signal, idx_lst, atr_sample)
    data_frame = data_frame.set_index('idx')
    data_frame.columns = data_frame.columns.astype(str)

    # paruošiame požymių masyvą klasifikatoriui
    data_frame_init = data_frame[all_features]
    data_array = scaler.transform(data_frame_init)
    test_x = data_array
    x_test = test_x.reshape((test_x.shape[0], test_x.shape[1], 1))

    # Pūpsnių klasių atpažinimas
    predictions = model.predict(x_test)
    predictions_y = np.argmax(predictions, axis=1)

# Sužymimi neatpažinti pūpsniai - klasė 3:'U'
    pred_y = np.zeros(len(atr_sample), dtype=int)
    pred_y[0] = 3
    pred_y[len(atr_sample)-1] = 3

    if (omitted.empty != True):
        idxs = list(omitted['idx'].astype('int'))
        for i in range(len(idxs)):
            pred_y[idxs[i]] = 3
            
# Sužymimi atpažinti pūpsniai 
    selected = data_frame.index.astype('int')
    pred_y[selected] = predictions_y
    return pred_y


