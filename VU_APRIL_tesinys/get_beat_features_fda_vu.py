from skfda import FDataGrid
import math
import scipy.signal
import keras
import numpy as np
import pandas as pd

def get_RR_arr(atr_sample, idx, nl_steps, nr_steps):
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


def get_seq(signal, atr_sample, idx, wl_side, wr_side):
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
        ret = pd.DataFrame([{"P_val": orig[P], "Q_val":orig[Q], "R_val": orig[R], "S_val": orig[S], "T_val":orig[T],
                           "P_pos":P * 1./ reample_points, "Q_pos":Q * 1./ reample_points,
                           "R_pos":R * 1./ reample_points, "S_pos":S * 1./ reample_points,
                           "T_pos": T * 1./ reample_points,
                           "QRS":QRS * 1./ reample_points, "PR":PR * 1./ reample_points,
                           "ST":ST * 1./ reample_points, "QT":QT * 1./ reample_points}])
        return ret
    else:
        return pd.DataFrame()

def get_beat_features_fda_vu_v1(signal, atr_sample, idx):
# def apply_FDA_modified(train_set_idx, all_beats_attr):
    # randomlist = random.sample(range(0, len(all_beats_attr)), 3)
    #basis = BSpline(n_basis=40, domain_range=(0,1), order=3)
    # basis = Fourier(n_basis=70, domain_range=(0,1))
    #smoother = BasisSmoother(basis = basis, return_basis=True, method='svd')
    # count = len(idx_lst)
    resapmling_points = 200
    fraction_to_drop_l = 0.7
    fraction_to_drop_r = 0.7
    show_period = 100
    samples = np.linspace(0, 1, resapmling_points)
    train_set_stats = pd.DataFrame(columns=['idx', 'seq_size', 'label', 'RRl', 'RRr', "RRl/RRr", 'wl_side', 'wr_side',
                                           "signal_mean", "signal_std"])

    train_set_points = pd.DataFrame()
    Rythm_Data = pd.DataFrame()
    omit_idx = pd.DataFrame()

      #idx =7883
    # ************************************* pakeitimas kj ************************************************
    # seq_1d, RRl, RRr = read_seq_RR(db_path, all_beats_attr, idx, wl_side, wr_side)
    RRl_arr, RRr_arr = get_RR_arr(atr_sample, idx, nl_steps=1, nr_steps=1)
    RRl = RRl_arr[0]
    RRr = RRr_arr[0]
    wl_side = math.floor(RRl * fraction_to_drop_l)  # pakeista all_beats_attr.loc[idx][5] į RRl, kj
    wr_side = math.floor(RRr * fraction_to_drop_r)  # pakeista all_beats_attr.loc[idx][6] į RRr, kj
    seq_1d = get_seq(signal, atr_sample, idx, wl_side, wr_side)

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
#    
#   I. Rythm_Data: 'P_val', 'Q_val', 'R_val', 'S_val', 'T_val', 'P_pos', 'Q_pos', 'R_pos',
#  'S_pos', 'T_pos', 'QRS', 'PR', 'ST', 'QT' - viso 14 požymių
# 
        RPT.append(indexes_lower[0][ind_low[0]])  # 1-st is P
        RPT.append(indexes_upper[0][tmp1[0]]) # 2nd is T

        Rythm_Data = get_spike_width(fd, fdd, resapmling_points, RPT)
        if not Rythm_Data.empty:
# 
# II. train_set_stats": 'idx', 'seq_size', 'label', 'RRl', 'RRr', 'RRl/RRr', 'wl_side', 'wr_side',
# 'signal_mean', 'signal_std' - viso 10 požymių
#  
            train_set_stats = pd.DataFrame([{'idx': idx, 'seq_size': seq_1d.shape[0], 'label': 0,
                                                    'RRl': RRl, 'RRr': RRr, 'RRl/RRr': RRl / RRr,
                                                    'wl_side': wl_side, 'wr_side': wr_side,
                                                    "signal_mean": np.mean(fd),
                                                    "signal_std": np.std(fd)}])
# 
# III. train_set_points: '0', '1', '2', ... , '197', '198', '199' - viso 200 požymių
#  
            #train_set_stats = train_set_stats.append(train_set_stats, ignore_index=False)
            # print(f"\nfd: {fd}  {type(fd)}  {fd.shape}")
            

            train_set_points = pd.Series(fd).to_frame().T
            # train_set_points = train_set_points.append(pd.Series(fd), ignore_index=True)
        else:
            omit_idx = pd.DataFrame([{'idx': idx}])
    else:
        omit_idx = pd.DataFrame([{'idx': idx}])

    train_set_stats = pd.concat([train_set_stats, Rythm_Data, train_set_points], axis=1)
    return(train_set_stats, omit_idx)    
