from pickle import FALSE
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import scipy ,json, sys, math

import neurokit2 as nk
from neurokit2 import signal_filter, signal_resample

from pathlib import Path, PurePath

# from wfdb_plot import plot_items
import itertools

from icecream import ic



def read_seq(rec_dir, all_beats_attr, idx, wl_side, wr_side):
# nuskaito EKG seką apie R dantelį: wl_side - iš kairės pusės, wr_side - iš dešinės pusės
    row = all_beats_attr.loc[idx]

    file_path = Path(rec_dir, str(row['userNr']) + '.npa')

    # with open(file_path, "rb") as f:
        # signal = np.load(f) 

    signal = np.load(file_path, mmap_mode='r')

    signal_length = signal.shape[0]

    (seq_start, seq_end)  = get_seq_start_end(signal_length, row['sample'], wl_side, wr_side)
        
    # Praleidžiame per trumpas sekas įrašo pradžioje ir pabaigoje
    if (seq_start == None or seq_end == None):
        return None, None 
    else:    
        seq = signal[seq_start:seq_end]
        label = row['label']
    return seq, label


def get_seq_start_end(signal_length,i_sample,window_left_side,window_right_side):
    # Nustatome išskiriamos EKG sekos pradžią ir pabaigą
    seq_start = i_sample - window_left_side
    seq_end = i_sample + window_right_side
    if (seq_start < 0 or seq_end > signal_length):
        # print("\nseq_start: ", seq_start, " seq_end: ", seq_end)
        return (None,None)
    else:    
        return (seq_start, seq_end)


def plot_seq(rec_dir, all_beats_attr, idx, wl_side, wr_side, window_left_side_ext, window_right_side_ext):
# 'Išpjauname' užduoto ilgio seką ir sukuriame jos vaizdą
    row = all_beats_attr.loc[idx]
    userNr = str(row['userNr'])
    sample = row['sample']
    # print(userNr, sample)

    fig = plt.figure(facecolor=(1, 1, 1), figsize=(18,3)) 
    ax = read_show_seq_ext_mit2zive(rec_dir, userNr, sample, wl_side, wr_side, 
                                                                window_left_side_ext, window_right_side_ext)
    if (ax == None):
        print(f"Sekai userNr: {userNr}  idx: {idx} negali suformuoti išplėstinio vaizdo")
    else:
        txt = f"{'userNr:'} {str(userNr)}  {'idx:'} {str(idx)}"
        plt.title(txt)
        plt.show()    


def create_set_from_rec(rec_dir, all_beats_attr, ind_lst, window_left_side, window_right_side):
    # Panaudojant sekų atributų freimą, sukuriamas užduoto ilgio sekų ir klasių numerių masyvai, 
    # tinkami klasifikatoriaus mokymui ir tikslumo vertinimui
    
    seq_length = window_left_side + window_right_side

    # Suformuojame failus pildymui
    set_len = len(ind_lst)
    X = np.empty((set_len, seq_length))
    y = np.empty((set_len), dtype=int)

    # Pildymas
    count = 0
    for idx in ind_lst:
        seq_1d, label = read_seq(rec_dir, all_beats_attr, idx, window_left_side, window_right_side)
        if (label != None):
            X[count,:] = seq_1d 
            y[count] = label
            count +=1
        else:
            print("Klaida!", idx)    

    # Koreguojame ilgius 
    X_set = np.resize(X, (count, seq_length))
    y_set = np.resize(y, (count))
    return X_set, y_set



def create_set_from_rec_corrected(rec_dir, all_beats_attr, ind_lst, wl_side, wr_side, batches):
# Įėjimo parametrai:   
    # rec_dir - aplankas su duomenimis
    # all_beats_attr - pūpsnių atributai
    # ind_lst - atrinktų sekų indeksų sąrašas (list)
    # wl_side - kairioji R atžvilgiu sekos pusė
    # wr_side - dešinė R atžvilgiu sekos pusė
# Išėjimo parametrai:
    # X_set - numpy masyvas su sekomis
    # y_set - numpy masyvas su sekų klasių numeriais
    # ind_lst - koreguotas indeksų sąrašas (list) 

    all_set_len = len(ind_lst)

    # Koreguojame mokymo imties dydį, kad santykis su batch size būtų sveikas skaičius
    set_len = math.floor(all_set_len/batches)*batches
    del ind_lst[set_len:]
    X_set, y_set = create_set_from_rec(rec_dir, all_beats_attr, ind_lst, wl_side, wr_side)

    return X_set, y_set, ind_lst


def runtime(s):
    hours, remainder = divmod(s, 3600)
    minutes, seconds = divmod(remainder, 60)
    print('Runtime: {:02}:{:02}:{:02}'.format(int(hours), int(minutes), int(seconds)))


def ecg_record_plot(sign, sampfrom, sampto, title=None, figsize=(18,3)):
    # Atvaizduoja vienmatę seką numpy array sign
    # Atvaizduoja visą seką, start - pirmos vaizduojamos reikšmės indeksas 
   
    plt.figure(figsize=figsize)
    if title is not None:
        plt.title(title)

    len = sign.shape[0]
    if (sampto > len):
        sampto=len

    x = np.arange(sampfrom, sampto, 1)
    plt.plot(x, sign[sampfrom:sampto], color="#6c3376", linewidth=2)
    plt.show()
    # plt.close()

def plot_sign(signal, sampfrom, sampto, ann_samp=None, ann_symb=None, title=None, figsize=(18,3)):
    # Atvaizduoja vienmatę seką numpy array signal su anotacijomis
    
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    if title is not None:
        plt.title(title)

    len = signal.shape[0]
    if (sampto > len):
        sampto=len
        
    x = np.arange(sampfrom, sampto, 1)
    axes.plot(x,signal[sampfrom:sampto],color="#6c3376", linewidth=2)

    if ann_samp is not None and ann_symb is not None:
        # Paimame anotacijas iš užduoto įrašo intervalo
        kept_inds = np.intersect1d(np.where(ann_samp>=sampfrom), np.where(ann_samp<=sampto))
        ann_samp = ann_samp[kept_inds]
        ann_symb = ann_symb[kept_inds]
        y = signal[ann_samp]
        # Vizualizuojame anotacijas
        for i in range(ann_samp.shape[0]):
            x_c = ann_samp[i]
            y_c = y[i]
            axes.annotate(ann_symb[i],(x_c,y_c),xytext=(x_c-10, y_c+1),arrowprops=dict(arrowstyle="->",connectionstyle="arc"))
    plt.show(block=False)
    # plt.pause(3)
    plt.close()
    
def ecg_record_stat(sign,subject):
# Statistika 
    result = pd.Series(sign).describe()
    skew_val = scipy.stats.skew(sign, axis=0)
    data_stat = {"subject":subject,"count":result[0], "mean":result[1],
    "std":result[2], "min":result[3], "max":result[7], "skirt":result[7]-result[3],
    "skew":skew_val,"25%":result[4], "50%":result[5], "75%":result[6]}
    return data_stat

def ecg_record_hist(sign):
# Histograma
    hist, bin_edges = np.histogram(sign, bins=20)
    # print(hist)
    fig, ax = plt.subplots()
    ax.hist(sign, bin_edges, cumulative=False)
    ax.set_xlabel('x')
    ax.set_ylabel('Frequency')
    plt.show()

def zive_read_file_1ch(filename):
    f = open(filename, "r")
    a = np.fromfile(f, dtype=np.dtype('>i4'))
    ADCmax=0x800000
    Vref=2.5
    b = (a - ADCmax/2)*2*Vref/ADCmax/3.5*1000
    ecg_signal = b - np.mean(b)
    return ecg_signal

def mit2zive_read_df_data(file_path, name):
    df_data = pd.DataFrame()
    path = Path(file_path)
    # https://www.askpython.com/python-modules/check-if-file-exists-in-python
    if (path.exists()):
        with open(file_path,'r', encoding='UTF-8', errors = 'ignore') as f:
            data = json.loads(f.read())
        df_data = pd.json_normalize(data, record_path =[name])
    return df_data

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
    
def pasiskirtymas_klasese(y, cols_pattern=None):
  # y - numpy array
  # cols_pattern - pvz. ['N','S','V']
  (unique, counts) = np.unique(y_train, return_counts=True)
  if (cols_pattern is not None):
    print(cols_pattern, counts, 'sum=', counts.sum())
  else:
    print(unique, counts, 'sum =', counts.sum())


def show_seq_mit2zive_pred(signal, atr_sample, atr_symbol, atr_symbol_pred, frag_start, frag_end, win_ls, win_rs, win_flag=False):

# win_ls - klasifikuojamo EKG segmento plotis iki R pūpsnio (iš kairės) 
# win_rs - klasifikuojamo EKG segmento plotis nuo R pūpsnio (iš dešinės)

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

def show_seq_mit2zive(signal, atr_sample, atr_symbol, frag_start, frag_end, win_ls, win_rs, win_flag=False):

# win_ls - klasifikuojamo EKG segmento plotis iki R pūpsnio (iš kairės) 
# win_rs - klasifikuojamo EKG segmento plotis nuo R pūpsnio (iš dešinės)

    # Išskiriame fragmentą
    fragment = signal[frag_start:frag_end]
    # gražiname seką 2d: sequence.shape(seq_end-seq_start, 1)

    # suformuojame anotacijų žymes, beat_locs - indeksas sekoje 
    beat_symbols, beat_locs = get_symbol_list(atr_symbol,atr_sample, frag_start, frag_end)

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


def read_show_seq_ext_mit2zive(rec_dir, subject, i_sample, win_ls, win_rs, win_ls_ext, win_rs_ext):
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
    file_path = Path(rec_dir, str(subject) + '.npa')
    with open(file_path, "rb") as f:
        signal = np.load(f) 
    signal_length = signal.shape[0]

    # Nuskaitome paciento anotacijas ir jų indeksus
    file_path = Path(rec_dir, str(subject) + '.json')
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

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix',
cmap=plt.cm.Blues):

# Confusion matrix, Rakesh Rajpurohit
# https://medium.com/@rakeshrajpurohit/confusion-matrix-469248ed0397

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    # else:
        # print('Confusion matrix, without normalization')

    # print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")
                
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def get_test_labels(file_names,labels):
# sudarome "mokytojo" priskirtų klasių sąrašą y_test.
# File_names - failų su sekomis vardai, labels - failas tipo dictionary
#  su failų vardais ir klasių numeriais
    y_test = []
    for name in file_names:
        label = labels[name]
        y_test.append(label)
    return y_test

def correct_partition(partition, batches):
    partition_len = len(partition)
    new_partition_len = int(partition_len/batches)*batches
    new_partition = partition[:new_partition_len]     
    return partition_len, new_partition_len, new_partition

import tensorflow as tf

from tensorflow import keras
from numpy import loadtxt
import os
from sklearn.preprocessing import StandardScaler

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

# Iš mit_bih_util.py

def get_seq_start_end(signal_length,i_sample,window_left_side,window_right_side):
    # Nustatome išskiriamos EKG sekos pradžią ir pabaigą
    seq_start = i_sample - window_left_side
    seq_end = i_sample + window_right_side
    if (seq_start < 0 or seq_end > signal_length):
        # print("\nseq_start: ", seq_start, " seq_end: ", seq_end)
        return (None,None)
    else:    
        return (seq_start, seq_end)


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

    return (beat_symbols,beat_locs)

# Iš mit_bih_util.py

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

def get_subj_seq_nr(seq_file_name):
    # pvz. seq_file_name = '210_255_NN'
    lst = seq_file_name.split('_',2)
    subject = int(lst[0])
    seq_nr = int(lst[1])
    return subject, seq_nr

def split_seq_file_name(seq_file_name):
    # file_name = "M" + userNr + "_" + registrationNr + "_" + str(seq_nr) + "_" + atr_symbol[i]
    # pvz. seq_file_name = 'M100_0_0_N'

    lst = seq_file_name.split('_',3)
    frag = lst[0].split('M')
    userNr = int(frag[1])
    recordingNr = int(lst[1])
    seq_nr = int(lst[2])
    atr_symbol = lst[3] 
    return userNr, recordingNr, seq_nr, atr_symbol


def load_dict(path):
    # reading the data from the file
    with open(path) as f:
        data = f.read()
        # reconstructing the data as a dictionary
        js = json.loads(data)
        f.close()
    return(js)