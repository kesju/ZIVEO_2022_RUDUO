#zive_util.py>

import numpy as np
import pandas as pd
from numpy import loadtxt
import neurokit2 as nk
from neurokit2 import signal_filter
# import math
import os, sys
from pathlib import Path

from bitstring import BitArray
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import json
# import plotly.express as px
# import scipy.signal

import tensorflow as tf
from tensorflow import keras

def get_rec_Id(rec_dir, userNr, recordingNr):
    # Patikriname, ar df_transl egzistuoja. 
    if (userNr < 1000):
         return userNr, recordingNr
    file = Path(rec_dir, 'df_transl.csv')
    if (file.exists()):
        # Nuskaitome vardų žodyną iš rec_dir aplanko
        file_path = Path(rec_dir, 'df_transl.csv')
        df_transl = pd.read_csv(file_path, index_col=0)
#       print(df_transl) 
         # Panaudodami df masyvą df_transl su įrašų numeriais iš įrašų eilės numerių gauname ZIVE numerius
        row = df_transl.loc[(df_transl['userNr'] == userNr) & (df_transl['recordingNr'] == recordingNr)]
        if row.empty:
            print("Klaida!")
            return None, None
        else:
            return row['userId'].values[0], row['recordingId'].values[0]
    else:
        print("df_transl neegzistuoja")


def get_rec_Nr(rec_dir, userId, recordingId):
    # Patikriname, ar df_transl egzistuoja. 
    file = Path(rec_dir, 'df_transl.csv')
    if (file.exists()):
        # Nuskaitome vardų žodyną iš rec_dir aplanko
        file_path = Path(rec_dir, 'df_transl.csv')
        df_transl = pd.read_csv(file_path, index_col=0)
        # Panaudodami df masyvą df_transl su įrašų numeriais iš ZIVE numerių gauname įrašų eilės numerius
        row = df_transl.loc[(df_transl['userId'] == userId) & (df_transl['recordingId'] == recordingId)]
        if row.empty:
            print("Neegzistuoja!")
            return None, None
        else:
            return row['userNr'].values[0], row['recordingNr'].values[0]

def get_rec_userId(rec_dir, userNr):
    # Patikriname, ar ne mit2zive
    if (userNr < 1000):
         return userNr
         
    # Patikriname, ar df_transl egzistuoja. 
    file = Path(rec_dir, 'df_transl.csv')
    if (file.exists()):
        # Nuskaitome vardų žodyną iš rec_dir aplanko
        file_path = Path(rec_dir, 'df_transl.csv')
        df_transl = pd.read_csv(file_path, index_col=0)
        #  print(df_transl) 
        # Panaudodami df masyvą df_transl su įrašų numeriais iš įrašų eilės numerių gauname ZIVE numerius
        row = df_transl.loc[(df_transl['userNr'] == userNr)]
        if row.empty:
            print("Klaida!")
            return None
        else:
            return row['userId'].values[0]
    else:
        print("df_transl neegzistuoja")
    
def get_rec_userNr(rec_dir, userId):
    # Reiktų patikrinti, ar ne mit2zive
    # ......................

    # Patikriname, ar df_transl egzistuoja. 
    file = Path(rec_dir, 'df_transl.csv')
    if (file.exists()):
        # Nuskaitome vardų žodyną iš rec_dir aplanko
        file_path = Path(rec_dir, 'df_transl.csv')
        df_transl = pd.read_csv(file_path, index_col=0)
        #  print(df_transl) 
        # Panaudodami df masyvą df_transl su įrašų numeriais iš įrašų eilės numerių gauname ZIVE numerius
        row = df_transl.loc[(df_transl['userId'] == userId)]
        if row.empty:
            print("Klaida!")
            return None
        else:
            return row['userNr'].values[0]   # userNr = df.iloc[0,2]
    else:
        print("df_transl neegzistuoja")

def get_rec_recNr(rec_dir, recordingId):
    # Reiktų patikrinti, ar ne mit2zive
    # ......................

    # Patikriname, ar df_transl egzistuoja. 
    file = Path(rec_dir, 'df_transl.csv')
    if (file.exists()):
        # Nuskaitome vardų žodyną iš rec_dir aplanko
        file_path = Path(rec_dir, 'df_transl.csv')
        df_transl = pd.read_csv(file_path, index_col=0)
        # Panaudodami df masyvą df_transl su įrašų numeriais iš ZIVE numerių gauname įrašų eilės numerius
        row = df_transl.loc[df_transl['recordingId'] == recordingId]
        if row.empty:
            print("Neegzistuoja!")
            return None, None
        else:
            return row['userNr'].values[0], row['recordingNr'].values[0]


def get_beat_attributes(idx, all_beats_attr):
    row = all_beats_attr.loc[idx]
    return row['userNr'], row['recordingNr'], row['label'], row['symbol'] 


def split_SubjCode(SubjCode):
    # SubjCode = userNr + recordingNr
    # pvz. SubjCode = 10002
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

def create_SubjCode(userNr, recordingNr):
    # SubjCode = userNr + recordingNr
    # pvz. SubjCode = 10002
    if (userNr < 1000):
        return userNr
    else:        
        str_code = str(userNr) + str(recordingNr)
        SubjCode = int(str_code)
        return SubjCode

def get_SubjCode(idx, all_beats_attr):
    row = all_beats_attr.loc[idx]
    SubjCode = create_SubjCode(row['userNr'],  row['recordingNr'])
    return SubjCode
    

def get_ind_list(all_beats_attr, userNr, recordingNr):
    index_beats_attr = all_beats_attr.index
    selected_ind = index_beats_attr[(all_beats_attr['userNr']==userNr) & (all_beats_attr['recordingNr']==recordingNr)]
    return selected_ind.to_list()

def read_rec(rec_dir, SubjCode):
    file_path = Path(rec_dir, str(SubjCode) + '.npy')
    signal = np.load(file_path, mmap_mode='r')
    print(f"SubjCode: {SubjCode}  signal.shape: {signal.shape}")
    return signal

def read_seq(rec_dir, all_beats_attr, idx, wl_side, wr_side):
    # nuskaito EKG seką apie R dantelį: wl_side - iš kairės pusės, wr_side - iš dešinės pusės
    row = all_beats_attr.loc[idx]
    SubjCode = create_SubjCode(row['userNr'],  row['recordingNr'])
    
    file_path = Path(rec_dir, str(SubjCode) + '.npy')
    signal = np.load(file_path, mmap_mode='r')    
    signal_length = signal.shape[0]
    (seq_start, seq_end)  = get_seq_start_end(signal_length, row['sample'], wl_side, wr_side)
        
    # Praleidžiame per trumpas sekas įrašo pradžioje ir pabaigoje
    if (seq_start == None or seq_end == None):
        return None, None 
    else:    
        seq = signal[seq_start:seq_end]
        sample = row['sample']
        label = row['label']
    return seq, sample, label


def read_seq_RR(rec_dir, all_beats_attr, idx, wl_side, wr_side):
# Nuskaito ir pateikia EKG seką apie R dantelį: wl_side - iš kairės pusės, wr_side - iš dešinės pusės,
# klasės numerį: 0, 1, 2.
# Taip pat pateikia RRl - EKG reikšmių skaičių nuo R dantelio iki prieš tai buvusio R dantelio,
# ir RRr - reikšmių skaičių nuo R dantelio iki sekančio R dantelio

    row = all_beats_attr.loc[idx]
    SubjCode = create_SubjCode(row['userNr'],  row['recordingNr'])

    file_path = Path(rec_dir, str(SubjCode) + '.npy')
    signal = np.load(file_path, mmap_mode='r')    
    signal_length = signal.shape[0]
    (seq_start, seq_end)  = get_seq_start_end(signal_length, row['sample'], wl_side, wr_side)

    RRl = row['RRl']
    RRr = row['RRr']

    # Ignoruojame per trumpas sekas įrašo pradžioje ir pabaigoje,
    # o taip pat pūpsnius, kuriems RRl ir RRr == -1
    if (seq_start == None or seq_end == None or RRl== -1 ):
        return None, None, None, None 
    else:    
        seq = signal[seq_start:seq_end]
        label = row['label']
    return seq, label, RRl, RRr

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


def plot_seq(rec_dir, all_beats_attr, idx, wl_side, wr_side, window_left_side_ext, window_right_side_ext):
# 'Išpjauname' užduoto ilgio seką ir sukuriame jos vaizdą
    row = all_beats_attr.loc[idx]
    SubjCode = create_SubjCode(row['userNr'],  row['recordingNr'])
    sample = row['sample']
    macro_annotation = row['symbol']
    # print(userNr, sample)

    fig = plt.figure(facecolor=(1, 1, 1), figsize=(18,3)) 
    ax = read_show_seq_ext_mit2zive(rec_dir, SubjCode, sample, wl_side, wr_side, 
                                                                window_left_side_ext, window_right_side_ext)
    if (ax == None):
        print(f"Sekai SubjCode: {str(SubjCode)}  idx: {idx} negali suformuoti išplėstinio vaizdo")
    else:
        txt = f"'SubjCode:' {str(SubjCode)}  'idx:' {str(idx)}  'macro_annotation:' {macro_annotation}"
        plt.title(txt)
        plt.show()    

def read_show_seq_ext_mit2zive(rec_dir, SubjCode, i_sample, win_ls, win_rs, win_ls_ext, win_rs_ext):
# Išpjauna užduoto ilgio seką iš mit2zive įrašo ir sukuria jos vaizdą su anotacijomis

# rec_dir - paciento EKG įrašų aplankas
# subject - paciento EKG įrašo numeris - int
# i_sample - R dantelio, kurio atžvilgiu formuojama seka, indeksas viso EKG įrašo reikšmių masyve - int
# win_ls - klasifikuojamo EKG segmento plotis iki R pūpsnio (iš kairės) 
# win_rs - klasifikuojamo EKG segmento plotis nuo R pūpsnio (iš dešinės)
# win_ls_ext - vaizduojamo EKG segmento plotis iki R pūpsnio (iš kairės) 
# win_rs_ext - vaizduojamo EKG segmento plotis už R pūpsnio (iš dešinės) 

    ax = plt.gca()
    
 # Nuskaitome visą paciento įrašą 
    file_path = Path(rec_dir, str(SubjCode) + '.npy')
    signal = np.load(file_path, mmap_mode='r')    
    signal_length = signal.shape[0]

    # Nuskaitome paciento anotacijas ir jų indeksus
    file_path = Path(rec_dir, str(SubjCode) + '.json')

    if (SubjCode > 1000):
        with open(file_path,'r', encoding='UTF-8', errors = 'ignore') as f:
            data = json.loads(f.read())
        df = pd.json_normalize(data, record_path =['rpeaks'])
    else:
        df = pd.read_json(file_path, orient = 'records')

    atr_sample = df['sampleIndex'].to_numpy()
    atr_symbol = df['annotationValue'].to_numpy()

    # surandame užduoto ilgio sekos pradžią ir pabaigą,
    # jei reikia - koreguojame
    seq_start, seq_end = get_seq_start_end(signal_length, i_sample, win_ls_ext, win_rs_ext)
    if (seq_start == None or seq_end == None):
        print("klaida!")
        return None

    # Išskiriame seką
    sequence = signal[seq_start:seq_end]

    # # suformuojame anotacijų žymes
    beat_symbols,beat_locs = get_symbol_list(atr_symbol,atr_sample, seq_start, seq_end)

    # deltax ir deltay simbolių pozicijų koregavimui
    min = np.amin(sequence)
    max = np.amax(sequence)
    deltay = (max - min)/20
    deltax = len(sequence)/100

    # suformuojame vaizdą
    x = np.arange(0, len(sequence), 1)
    ax.plot(x, sequence, color="#6c3376", linewidth=2)
    left_mark = i_sample - seq_start - win_ls
    right_mark = i_sample - seq_start + win_rs
    ax.axvline(x = left_mark, color = 'b', linestyle = 'dotted')
    ax.axvline(x = right_mark, color = 'b', linestyle = 'dotted')
    for i in range(len(beat_locs)):
        ax.annotate(beat_symbols[i],(beat_locs[i]-deltax,sequence[beat_locs[i]]+deltay))
    ax.set_ylim([min, max+2*deltay])
    
    return(ax)


def runtime(s):
    hours, remainder = divmod(s, 3600)
    minutes, seconds = divmod(remainder, 60)
    print('Runtime: {:02}:{:02}:{:02}'.format(int(hours), int(minutes), int(seconds)))

# Maksimumo vietos lyginimui 
def max_place(subject, atr_sample, sign, sample_from, sample_to):  
    for i in range(len(atr_sample)):
        index_s = atr_sample[i]
        if (index_s < sample_from or index_s >sample_to):
            continue
        print(f"\n{i}, index_s: {index_s}  max: {sign[index_s]:.3f}")
        for j in range(-5,5):
            index_y = index_s+j
            formatted_float = "{:.3f}".format(sign[index_y])
            print(index_y, formatted_float, end =' ')
        
def create_dir(parent_dir):
    # Sukuriami rekursyviškai aplankai, jei egzistuoja - tai nekuria
    # https://smallbusiness.chron.com/make-folders-subfolders-python-38545.html

    try:
        os.makedirs(parent_dir)
        print("Directory '%s' created successfully" % parent_dir)
        # print("Directory {:s} created successfully".format(parent_dir)
    except OSError as error:
        print("Directory '%s' already exists" % parent_dir)


def create_subdir(parent_dir, names_lst):
    # Sukuriami subdirektoriai su nurodytais pavadinimais
    for name in names_lst:
        # sukuriami aplankai EKG sekų vaizdams
        sub_dir = os.path.join(parent_dir, name)
        try:
            os.makedirs(sub_dir)
            print("Directory '%s' created successfully" % sub_dir)
        except OSError as error:
            print("Directory '%s' already exists" % sub_dir)

def get_rev_dictionary(dictionary):
    rev_dict = {value : key for (key, value) in dictionary.items()}
    return rev_dict


def get_symbol_list(atr_symbols, atr_samples, seq_start, seq_end):
    # Surenkame išpjautos EKG sekos anotacijas ir jų indeksus sekoje
    # ir patalpiname sąraše.
    beat_locs = []
    beat_symbols = []

    for i in range(len(atr_samples)):
        if atr_samples[i] > seq_start and atr_samples[i] < seq_end:
            beat_symbols.append(atr_symbols[i])
            beat_locs.append(atr_samples[i]-seq_start)   
            # beat_locs.append(atr_samples[i])   

def get_seq_start_end(signal_length,i_sample,window_left_side,window_right_side):
    # Nustatome išskiriamos EKG sekos pradžią ir pabaigą
    seq_start = i_sample - window_left_side
    seq_end = i_sample + window_right_side
    if (seq_start < 0 or seq_end > signal_length):
        # print("\nseq_start: ", seq_start, " seq_end: ", seq_end)
        return (None,None)
    else:    
        return (seq_start, seq_end)


def split_seq_file_name(seq_file_name):
    # file_name = "M" + userNr + "_" + registrationNr + "_" + str(seq_nr) + "_" + atr_symbol[i]
    # pvz. seq_file_name = 'M100_0_0_N'

    lst = seq_file_name.split('_',3)
    frag = lst[0]
    userNr = int(lst[0][1:])
    recordingNr = int(lst[1])
    seq_nr = int(lst[2])
    atr_symbol = lst[3] 
    return userNr, recordingNr, seq_nr, atr_symbol

def get_freq_unique_values(y, cols_pattern=None):
  # y - numpy array
  # cols_pattern - pvz. ['N','S','V']
  (unique, counts) = np.unique(y, return_counts=True)
  if (cols_pattern is not None):
    return cols_pattern, counts, int(counts.sum())
  else:
    return unique, counts, int(counts.sum())


def get_annotations_table(all_beats_attr, ind_lst=None, cols_pattern=None):
#  Skaičiuoja anotacijų pasiskirstymą per pacientus ir jų įrašus
#  ind_lst - indeksų sąrašas, kuriuos reikia įtraukti į skaičiavimą

    # Jei užduotas ind_lst, tai atsirenkame beats tik užduotus ind_lst sąraše
    if (ind_lst is not None):
        selected_beats_attr = all_beats_attr.loc[all_beats_attr.index[ind_lst]].copy()
    else:
        selected_beats_attr = all_beats_attr.copy()
    # print(selected_bearead_seqts_attr)

    selected_beats_attr['SubjCodes'] = selected_beats_attr['userNr'].astype(str) + selected_beats_attr['recordingNr'].astype(str)
    # print(selected_beats_attr)

    labels_table = pd.crosstab(index= selected_beats_attr['SubjCodes'], columns= selected_beats_attr['symbol'], margins=True)

    if (cols_pattern is not None):
        cols = list(labels_table.columns)
        cols_ordered = [s for s in cols_pattern if s in cols]
        labels_table = labels_table[cols_ordered]
    
    labels_sums = labels_table.sum(axis=1) 

    return labels_table, labels_sums

def print_annotations_table(labels_table, labels_sums, Flag1 = False, Flag2 = False):    
    if (Flag1):
        # Sausdinti visą lentelę
        print(labels_table)
    else:    
        count = labels_table.loc['All']
        d = count.to_dict()
        print(str(d)[1:-1])

    if (Flag2):    
        print("\n")
        print(labels_sums)
    else:
        print("Total: ", labels_sums.loc['All'])


def anotaciju_pasiskirstymas(seq_attr, cols_pattern=None):
    #  Anotacijų pasiskirstymas per visas sekas
    labels_table = pd.crosstab(index=seq_attr['subject'], columns=seq_attr['symbol'])
    if (cols_pattern is not None):
        cols = list(labels_table.columns)
        cols_ordered = [s for s in cols_pattern if s in cols]
        labels_table = labels_table[cols_ordered]
    # print(cols_ordered)
    print(labels_table)
    # print(labels_table.info())
    suma = labels_table.sum(axis = 0) 
    print('\nsum:',' '*3,str(suma.tolist())[1:-1])
    total = suma.sum()
    print('total: ', total)

    print("\nall shape" , seq_attr.shape)
    mem = sys.getsizeof(seq_attr)
    print("Size of all: ", mem, " bytes")


def anotaciju_pasiskirstymas_v2(all_beats_attr, ind_lst=None, cols_pattern=None):
#  Skaičiuoja anotacijų pasiskirstymą
#  ind_lst - indeksų sąrašas, kuriuos reikia įtraukti į skaičiavimą

    if (ind_lst is not None):
        tmp_beats_attr = all_beats_attr.loc[all_beats_attr.index[ind_lst]]
        labels_table = pd.crosstab(index=tmp_beats_attr['userNr'], columns=tmp_beats_attr['symbol'], margins=True)
    else:
        labels_table = pd.crosstab(index=all_beats_attr['userNr'], columns=all_beats_attr['symbol'], margins=True)

    if (cols_pattern is not None):
        cols = list(labels_table.columns)
        cols_ordered = [s for s in cols_pattern if s in cols]
        labels_table = labels_table[cols_ordered]
    
    labels_sums = labels_table.sum(axis=1) 

    return labels_table, labels_sums

def load_dict(path):
    # reading the data from the file
    with open(path) as f:
        data = f.read()
        # reconstructing the data as a dictionary
        js = json.loads(data)
        f.close()
    return(js)
    
def get_subj_seq_nr(seq_file_name):
    # pvz. seq_file_name = '210_255_NN'
    lst = seq_file_name.split('_',2)
    subject = int(lst[0])
    seq_nr = int(lst[1])
    return subject, seq_nr

def get_test_labels(file_names,labels):
# sudarome "mokytojo" priskirtų klasių sąrašą y_test.
# File_names - failų su sekomis vardai, labels - failas tipo dictionary
#  su failų vardais ir klasių numeriais
    y_test = []
    for name in file_names:
        label = labels[name]
        y_test.append(label)
    return y_test

def anotaciju_pasiskirstymas_agr(seq_attr, cols_pattern=None, grouping_dict=None):

    # Anotacijų papildomas grupavimas
    if (grouping_dict is not None):
        seq_attr_grouped = seq_attr.copy()
        seq_attr_grouped["symbol"].replace(grouping_dict, inplace=True)
    else:
        seq_attr_grouped = seq_attr

    #  Anotacijų pasiskirstymas per visas sekas
    labels_table = pd.crosstab(index=seq_attr_grouped['userNr'], columns=seq_attr_grouped['symbol'])
    
    # Stulpelių sutvarkymas
    if (cols_pattern is not None):
        cols = list(labels_table.columns)
        cols_ordered = [s for s in cols_pattern if s in cols]
        labels_table = labels_table[cols_ordered]

    # print(cols_ordered)
    print(labels_table)
    # print(labels_table.info())
    suma = labels_table.sum(axis = 0) 
    print('\nsum:',' '*3,str(suma.tolist())[1:-1])
    total = suma.sum()
    print('total: ', total)

    print("\nall shape" , seq_attr.shape)
    mem = sys.getsizeof(seq_attr)
    print("Size of all: ", mem, " bytes")
    return labels_table

def correct_partition(partition, batches):
    partition_len = len(partition)
    new_partition_len = int(partition_len/batches)*batches
    new_partition = partition[:new_partition_len]     
    return partition_len, new_partition_len, new_partition

def confusion_matrix_modified(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes,n_classes), dtype=int)
    length = len(y_true)
    for i in range(length):
        cm[y_true[i],y_pred[i]] +=1
    return cm

# Iš classif_util.py

def cm2df(cm, labels):
    df = pd.DataFrame()
    # rows
    for i, row_label in enumerate(labels):
        rowdata={}
        # columns
        for j, col_label in enumerate(labels): 
            rowdata[col_label]=cm[i,j]
        df = df.append(pd.DataFrame.from_dict({row_label:rowdata}, orient='index'))
    return df[labels]

def show_confusion_matrix(cnf_matrix, class_names):
    df = cm2df(cnf_matrix, class_names)
    print('Confusion Matrix')
    print(df)
    print("\n")

    flag_of_zero_values = False
    for i in range(len(class_names)):
        if (cnf_matrix[i,i] == 0):
            flag_of_zero_values = True

    if flag_of_zero_values != True:
        cnf_matrix_n = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
        print('Normalized Confusion Matrix')
        df = cm2df(cnf_matrix_n, class_names)
        pd.options.display.float_format = "{:,.3f}".format
        print(df)
    else:
        print('Zero values! Cannot calculate Normalized Confusion Matrix')


def zive_read_df_rpeaks(db_path, recordingId):
    file_path = Path(db_path, recordingId + '.json')
    with open(file_path,'r') as f:
        data = json.loads(f.read())
    df_rpeaks = pd.json_normalize(data, record_path =['rpeaks'])
    return df_rpeaks


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


def get_symbol_list(atr_symbols, atr_samples, seq_start, seq_end):
    # Surenkame išpjautos EKG sekos anotacijas ir jų indeksus sekoje
    # ir patalpiname sąraše.
    beat_locs = []
    beat_symbols = []

    for i in range(atr_samples.shape[0]):
        if atr_samples[i] > seq_start and atr_samples[i] < seq_end:
            beat_symbols.append(atr_symbols[i])
            beat_locs.append(atr_samples[i]-seq_start)   
            # beat_locs.append(atr_samples[i])   

    return (beat_symbols,beat_locs)

def get_recordingId_list(db_path, flag_incl):
# Jei flag_incl == False, į požymį incl nereaguoja, sąrašą išvęda visiems EKG įrašams.
# Jei flag_incl == True, sąrašą išvęda tik tiems įrašams, kuriems incl ==1
    file_path = Path(db_path, 'list.json')
    with open(file_path,'r') as f:
        data = json.loads(f.read())

    df_list = pd.json_normalize(data, record_path =['data'])

    if (flag_incl):
        recordingId_list = list(df_list[(df_list['incl'] == flag_incl)]['recordingId'])
    else:
        recordingId_list = list(df_list['recordingId'])

    return recordingId_list


def zive_read_df_data(file_path, name):
    df_data = pd.DataFrame()
    path = Path(file_path)
    # https://www.askpython.com/python-modules/check-if-file-exists-in-python
    if (path.exists()):
        with open(file_path,'r', encoding='UTF-8', errors = 'ignore') as f:
            data = json.loads(f.read())
        df_data = pd.json_normalize(data, record_path =[name])
    return df_data

def zive_read_json_data(file_path):
    path = Path(file_path)
    # https://www.askpython.com/python-modules/check-if-file-exists-in-python
    if (path.exists()):
        with open(file_path,'r', encoding='UTF-8', errors = 'ignore') as f:
            data = json.loads(f.read())
    return data    

def get_df_noises_frag(df_noises_orig, start, end):
# Koreguojame df_noises
    df_noises = df_noises_orig.copy()
    if (df_noises_orig.empty != True):
        df_noises.drop(df_noises[(df_noises['startIndex'] < start) & (df_noises['endIndex'] <= start)].index, inplace=True)
        df_noises.drop(df_noises[(df_noises['startIndex'] >= end) & (df_noises['endIndex'] > end)].index, inplace=True)
        if (df_noises.empty != True):
            for idx, row in df_noises.iterrows():
                if ((start > row['startIndex']) & (start <= row['endIndex'])):
                    row['startIndex'] = start
                if ((end < row['endIndex']) & (end >= row['startIndex'])):
                    row['endIndex'] = end    
    return df_noises    


def read_show_seq_ext_zive(db_path, recordingId, i_sample, win_ls, win_rs, win_ls_ext, win_rs_ext):
# Išpjauna užduoto ilgio seką iš mit2zive įrašo ir sukuria jos vaizdą su anotacijomis

# db_path - paciento EKG įrašų aplankas
# recordingId - paciento EKG įrašo Id - int
# i_sample - R dantelio, kurio atžvilgiu formuojama seka, indeksas viso EKG įrašo reikšmių masyve - int
# win_ls - klasifikuojamo EKG segmento plotis iki R pūpsnio (iš kairės) 
# win_rs - klasifikuojamo EKG segmento plotis nuo R pūpsnio (iš dešinės)
# win_ls_ext - vaizduojamo EKG segmento plotis iki R pūpsnio (iš kairės) 
# win_rs_ext - vaizduojamo EKG segmento plotis už R pūpsnio (iš dešinės) 

    ax = plt.gca()

 # Nuskaitome visą paciento įrašą 
    file_path = Path(db_path, recordingId)
    signal = zive_read_file_1ch(file_path)
    signal_length = signal.shape[0] 

    # print("recordingId=", recordingId,  "signal_length",  signal_length)
    # print("file_path =", file_path )

    # Nuskaitome paciento anotacijas ir jų indeksus
    filepath = Path(db_path, recordingId + '.json')
    df_rpeaks = zive_read_df_data(filepath, 'rpeaks')
    if (df_rpeaks.empty == True):
        print(f'Annotation file for {recordingId} does not exist or rpeaks is empty!')
    else:
        atr_sample = df_rpeaks['sampleIndex'].to_numpy()
        atr_symbol = df_rpeaks['annotationValue'].to_numpy()

    # surandame užduoto ilgio sekos pradžią ir pabaigą,
    # jei reikia - koreguojame
    seq_start, seq_end = get_seq_start_end(signal_length, i_sample, win_ls_ext, win_rs_ext)
    if (seq_start == None or seq_end == None):
        # print("klaida!")
        return None

    # Išskiriame seką
    sequence = signal[seq_start:seq_end]

    # # suformuojame anotacijų žymes
    beat_symbols,beat_locs = get_symbol_list(atr_symbol,atr_sample, seq_start, seq_end)

    # deltax ir deltay simbolių pozicijų koregavimui
    min = np.amin(sequence)
    max = np.amax(sequence)
    deltay = (max - min)/20
    deltax = len(sequence)/100

    # suformuojame vaizdą
    x = np.arange(0, len(sequence), 1)
    ax.plot(x, sequence, color="#6c3376", linewidth=2)
    left_mark = i_sample - seq_start - win_ls
    right_mark = i_sample - seq_start + win_rs
    ax.axvline(x = left_mark, color = 'b', linestyle = 'dotted')
    ax.axvline(x = right_mark, color = 'b', linestyle = 'dotted')
    for i in range(len(beat_locs)):
        ax.annotate(beat_symbols[i],(beat_locs[i]-deltax,sequence[beat_locs[i]]+deltay))
    ax.set_ylim([min, max+2*deltay])
    
    return(ax)


def show_seq_zive_noise_pred(signal, atr_sample, atr_symbol, atr_symbol_pred, df_noises_orig, frag_start, frag_end, win_ls, win_rs, win_flag=False):

# win_ls - klasifikuojamo EKG segmento plotis iki R pūpsnio (iš kairės) 
# win_rs - klasifikuojamo EKG segmento plotis nuo R pūpsnio (iš dešinės)

    # Išskiriame fragmentą
    fragment = signal[frag_start:frag_end]
    # gražiname seką 2d: sequence.shape(seq_end-seq_start, 1)

    # suformuojame anotacijų žymes, beat_locs - indeksas sekoje 
    beat_symbols, beat_locs = get_symbol_list(atr_symbol,atr_sample, frag_start, frag_end)
    pred_symbols, pred_locs = get_symbol_list(atr_symbol_pred,atr_sample, frag_start, frag_end)
    df_noises_frag = get_df_noises_frag(df_noises_orig, frag_start, frag_end)

    # deltax ir deltay simbolių pozicijų koregavimui
    min = np.amin(fragment)
    max = np.amax(fragment)
    deltay = (max - min)/20
    deltax1 = len(fragment)/100

    # suformuojame vaizdą
    ax = plt.gca()

    # Įrašo fragmento vaizdas
    # x = np.arange(0, len(fragment), 1)
    x = np.arange(frag_start, frag_end, 1)
    ax.plot(x, fragment, color="#6c3376", linewidth=2)
    left, right = ax.get_xlim()
    for i in range(len(beat_locs)):
        # Anotacijos
        ax.annotate(beat_symbols[i],xy=(beat_locs[i]+frag_start-deltax1,fragment[beat_locs[i]]+deltay))
        # Automatinės anotacijos
        if (beat_symbols[i] != pred_symbols[i]):
            ax.annotate(pred_symbols[i],xy=(beat_locs[i]+frag_start-deltax1,fragment[beat_locs[i]]+deltay), xycoords='data',
    bbox=dict(boxstyle="circle, pad=0.4", fc="none", ec="red"), xytext=(20, 5), textcoords='offset points', ha='center')

    # parodomos triukšmo vietos
    if (df_noises_frag.empty != True):
        for idx, row in df_noises_frag.iterrows():
            ax.axvspan(row['startIndex'], row['endIndex'], facecolor='lightgray')
    
    # Sekų rėžiai
    if win_flag:
        left_mark = beat_locs[i] + frag_start - win_ls
        right_mark = beat_locs[i] + frag_start + win_rs
        centr_mark = beat_locs[i] + frag_start
        ax.axvline(x = left_mark, color = 'b', linestyle = 'dotted')
        ax.axvline(x = right_mark, color = 'b', linestyle = 'dotted')
        ax.axvline(x = centr_mark, color = 'b', linestyle = 'dotted')

    ax.set_ylim([min, max+2*deltay])

    return(ax)

def show_seq_zive_noise(signal, atr_sample, atr_symbol, df_noises_orig, frag_start, frag_end, win_ls, win_rs, win_flag=False):

# win_ls - klasifikuojamo EKG segmento plotis iki R pūpsnio (iš kairės) 
# win_rs - klasifikuojamo EKG segmento plotis nuo R pūpsnio (iš dešinės)

    # Išskiriame fragmentą
    fragment = signal[frag_start:frag_end]
    # gražiname seką 2d: sequence.shape(seq_end-seq_start, 1)

    # suformuojame anotacijų žymes, beat_locs - indeksas sekoje 
    beat_symbols, beat_locs = get_symbol_list(atr_symbol,atr_sample, frag_start, frag_end)
    df_noises_frag = get_df_noises_frag(df_noises_orig, frag_start, frag_end)

    # deltax ir deltay simbolių pozicijų koregavimui
    min = np.amin(fragment)
    max = np.amax(fragment)
    deltay = (max - min)/20
    deltax1 = len(fragment)/100

    # suformuojame vaizdą
    ax = plt.gca()

    # Įrašo fragmento vaizdas
    # x = np.arange(0, len(fragment), 1)
    x = np.arange(frag_start, frag_end, 1)
    ax.plot(x, fragment, color="#6c3376", linewidth=2)
    left, right = ax.get_xlim()
    for i in range(len(beat_locs)):
        # Anotacijos
        ax.annotate(beat_symbols[i],xy=(beat_locs[i]+frag_start-deltax1,fragment[beat_locs[i]]+deltay))

    # parodomos triukšmo vietos
    if (df_noises_frag.empty != True):
        for idx, row in df_noises_frag.iterrows():
            ax.axvspan(row['startIndex'], row['endIndex'], facecolor='lightgray')
    
    # Sekų rėžiai
    if win_flag:
        left_mark = beat_locs[i] + frag_start - win_ls
        right_mark = beat_locs[i] + frag_start + win_rs
        centr_mark = beat_locs[i] + frag_start
        ax.axvline(x = left_mark, color = 'b', linestyle = 'dotted')
        ax.axvline(x = right_mark, color = 'b', linestyle = 'dotted')
        ax.axvline(x = centr_mark, color = 'b', linestyle = 'dotted')

    ax.set_ylim([min, max+2*deltay])

    return(ax)


def show_seq_ext_zive(signal, atr_sample, atr_symbol, frag_start, frag_end, win_ls, win_rs, win_flag=False):

# win_ls - klasifikuojamo EKG segmento plotis iki R pūpsnio (iš kairės) 
# win_rs - klasifikuojamo EKG segmento plotis nuo R pūpsnio (iš dešinės)

    # Išskiriame fragmentą
    fragment = signal[frag_start:frag_end]
    # gražiname seką 2d: sequence.shape(seq_end-seq_start, 1)

    # suformuojame anotacijų žymes, beat_locs - indeksas sekoje 
    beat_symbols,beat_locs = get_symbol_list(atr_symbol,atr_sample, frag_start, frag_end)

    # deltax ir deltay simbolių pozicijų koregavimui
    min = np.amin(fragment)
    max = np.amax(fragment)
    deltay = (max - min)/20
    deltax = len(fragment)/100

    # suformuojame vaizdą
    ax = plt.gca()

    # Įrašo fragmento vaizdas
    # x = np.arange(0, len(fragment), 1)
    x = np.arange(frag_start, frag_end, 1)
    ax.plot(x, fragment, color="#6c3376", linewidth=2)
    for i in range(len(beat_locs)):
        # Anotacijos
        ax.annotate(beat_symbols[i],(beat_locs[i]+frag_start-deltax,fragment[beat_locs[i]]+deltay))

    #     # Sekų rėžiai
        if (win_flag):
            left_mark = beat_locs[i] + frag_start - win_ls
            right_mark = beat_locs[i] + frag_start + win_rs
            centr_mark = beat_locs[i] + frag_start
            ax.axvline(x = left_mark, color = 'b', linestyle = 'dotted')
            ax.axvline(x = right_mark, color = 'b', linestyle = 'dotted')
            ax.axvline(x = centr_mark, color = 'b', linestyle = 'dotted')

    ax.set_ylim([min, max+2*deltay])

    return(ax)


def show_seq_ext_zive_pred(signal, atr_sample, atr_symbol, atr_symbol_pred, frag_start, frag_end, win_ls, win_rs, win_flag=False):

# win_ls - klasifikuojamo EKG segmento plotis iki R pūpsnio (iš kairės) 
# win_rs - klasifikuojamo EKG segmento plotis nuo R pūpsnio (iš dešinės)

    beat_pred_symbols = ['E', 'Z']
    beat_pred_locs = [100,200]

    # Išskiriame fragmentą
    fragment = signal[frag_start:frag_end]
    # gražiname seką 2d: sequence.shape(seq_end-seq_start, 1)

    # suformuojame anotacijų žymes, beat_locs - indeksas sekoje 
    beat_symbols, beat_locs = get_symbol_list(atr_symbol,atr_sample, frag_start, frag_end)
    pred_symbols, pred_locs = get_symbol_list(atr_symbol_pred,atr_sample, frag_start, frag_end)


    # deltax ir deltay simbolių pozicijų koregavimui
    min = np.amin(fragment)
    max = np.amax(fragment)
    deltay = (max - min)/20
    deltax1 = len(fragment)/100
    deltax2 = len(fragment)/50

    # suformuojame vaizdą
    ax = plt.gca()

    # Įrašo fragmento vaizdas
    # x = np.arange(0, len(fragment), 1)
    x = np.arange(frag_start, frag_end, 1)
    ax.plot(x, fragment, color="#6c3376", linewidth=2)
    for i in range(len(beat_locs)):
        # Anotacijos
        ax.annotate(beat_symbols[i],(beat_locs[i]+frag_start-deltax1,fragment[beat_locs[i]]+deltay))
        # Automatinės anotacijos
        if (beat_symbols[i] != pred_symbols[i]):
            ax.annotate(pred_symbols[i],xy=(beat_locs[i]+frag_start-deltax1,fragment[beat_locs[i]]+deltay), xycoords='data',
    bbox=dict(boxstyle="circle, pad=0.4", fc="none", ec="red"), xytext=(20, 5), textcoords='offset points', ha='center')
   
    #     # Sekų rėžiai
        if (win_flag):
            left_mark = beat_locs[i] + frag_start - win_ls
            right_mark = beat_locs[i] + frag_start + win_rs
            centr_mark = beat_locs[i] + frag_start
            ax.axvline(x = left_mark, color = 'b', linestyle = 'dotted')
            ax.axvline(x = right_mark, color = 'b', linestyle = 'dotted')
            ax.axvline(x = centr_mark, color = 'b', linestyle = 'dotted')

    ax.set_ylim([min, max+2*deltay])

    return(ax)


class DataGenerator_CNN_M(keras.utils.Sequence):
# Pritaikytas modeliui CNN  (generuoja X 3-dimensijų),
#  klasių skaičiui lygu M (naudoja keras.utils.to_categorical)

    'Generates data for Keras'
    def __init__(self, data_path, list_IDs, labels, batch_size=64, dim=200, n_channels=1, n_classes=3, shuffle=False):
        'Initialization'
        self.dim = dim
        self.n_channels = n_channels
        self.batch_size = batch_size
        self.data_path = data_path
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(len(self.list_IDs) / self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        # X = np.empty((self.batch_size, self.dim, self.n_channels))
        X = np.empty((self.batch_size, self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)
        
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Nuskaitome duomenis
            file_path = os.path.join(self.data_path,ID) + '.csv'
            # file_path = self.data_path + "\" + ID + '.csv'
            seq1d = loadtxt(file_path, delimiter=',')

            # Normalizacija
            seq2d = np.reshape(seq1d,(-1,1))
            scaler = StandardScaler()
            signal = scaler.fit_transform(seq2d)

            X[i,:,0] = signal[:,0]
            # X = X.reshape((X.shape[0], X.shape[1], 1))
            # print(X.shape)
            y[i] = self.labels[ID]
        
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
        # print(X.shape,y.shape)
        # y = to_categorical(y)
        # print(y)
        # return X, y
