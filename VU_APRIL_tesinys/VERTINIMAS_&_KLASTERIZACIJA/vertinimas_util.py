import numpy as np
import neurokit2 as nk
from neurokit2 import signal_filter
from bitstring import BitArray
import pandas as pd
import os, sys
import matplotlib.pyplot as plt
import json
from pathlib import Path


def zive_read_file_1ch(filename):
    f = open(filename, "r")
    a = np.fromfile(f, dtype=np.dtype('>i4'))
    ADCmax=0x800000
    Vref=2.5
    b = (a - ADCmax/2)*2*Vref/ADCmax/3.5*1000
    ecg_signal = b - np.mean(b)
    return ecg_signal


def read_signal(rec_dir, filename):
    """
    Tinka EKG įrašų skaitymui tiek zive, tiek mit2zive atveju.
    zive atveju filename pvz. 1621694.321, 1621694.321.json
    mit2zive atveju, pvz. 100.000, 100.000.json - dalis iki taško ne ilgesnė
    už 4 simbolius

    Parameters
    ------------
        rec_dir: string
        filename: string
    Return
    -----------
        signl: numpy array, float
    """   
    file_path = Path(filename)
    name = file_path.stem
    file_path = Path(rec_dir, filename)
    
    if len(name) < 7:
        with open(file_path, "rb") as f:
            signl_loaded = np.load(f) 
        return signl_loaded
    else:        
        signl_loaded = zive_read_file_1ch(file_path)
        return signl_loaded


def AnalyseHeartrate(ecg_signal_df):
    _, rpeaks = nk.ecg_peaks(ecg_signal_df['orig'], sampling_rate=200, method="neurokit", correct_artifacts=False)
    ret = {'rpeaks':rpeaks['ECG_R_Peaks'].tolist()}
    return ret 

def anotacijos(df, symbol):
    df_anot = df.loc[df['annotationValue'] == symbol]
    keys_anot = df_anot['sampleIndex'].values.tolist()
    values_anot = df_anot['annotationValue'].values.tolist()
    lst_anot = {keys_anot[i]: values_anot[i] for i in range(len(keys_anot))}
    return lst_anot

def get_seq_start_end(signal_length, i_sample, window_left_side, window_right_side):
    # Nustatome išskiriamos EKG sekos pradžią ir pabaigą
    seq_start = i_sample - window_left_side
    seq_end = i_sample + window_right_side
    if (seq_start < 0 or seq_end > signal_length):
        # print("\nseq_start: ", seq_start, " seq_end: ", seq_end)
        return (None,None)
    else:    
        return (seq_start, seq_end)

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
            # Analizuojamo intervalo centras 
            centr_mark = beat_locs[i] + frag_start
            ax.axvline(x = centr_mark, color = 'b', linestyle = 'dotted')

            # Kairė analizuojamo intervalo riba - raudona
            left_mark = centr_mark - win_ls
            if (left_mark < frag_start):
                left_mark = frag_start
            # ax.axvline(x = left_mark, color = 'r', linestyle = 'dotted')

            # Dešinė analizuojamo intervalo riba - žalia
            right_mark = centr_mark + win_rs
            if (right_mark > frag_end):
                right_mark = frag_end
            # ax.axvline(x = right_mark, color = 'g', linestyle = 'dotted')

    ax.set_ylim([min, max+2*deltay])

    return(ax)

# ////////////////////////////////////////////////////////////////////////////////////////

def runtime(s):
    hours, remainder = divmod(s, 3600)
    minutes, seconds = divmod(remainder, 60)
    print('Runtime: {:02}:{:02}:{:02}'.format(int(hours), int(minutes), int(seconds)))

def count_lines_enumrate(file_name):
# Funkcija csv failo eilučių skaičiaus suradimui
# https://insightsndata.com/6-ways-to-find-number-of-lines-from-a-csv-file-in-python-b22eb63f7f7c

    fp = open(file_name,'r')
    for line_count, line in enumerate(fp):
        pass
    return line_count


def create_dir(parent_dir):
    """
    Sukuriami rekursyviškai aplankai, jei egzistuoja - tai nekuria
    https://smallbusiness.chron.com/make-folders-subfolders-python-38545.html
    Parameters
    ------------
        parent_dir: str
    """

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

def get_freq_unique_values(y, cols_pattern=None):
  # y - numpy array
  # cols_pattern - pvz. ['N','S','V']
  (unique, counts) = np.unique(y, return_counts=True)
  if (cols_pattern is not None):
    return cols_pattern, counts, int(counts.sum())
  else:
    return unique, counts, int(counts.sum())


        
def get_seq_start_end(signal_length,i_sample,window_left_side,window_right_side):
    # Nustatome išskiriamos EKG sekos pradžią ir pabaigą
    seq_start = i_sample - window_left_side
    seq_end = i_sample + window_right_side
    if (seq_start < 0 or seq_end > signal_length):
        # print("\nseq_start: ", seq_start, " seq_end: ", seq_end)
        return (None,None)
    else:    
        return (seq_start, seq_end)



def get_label_sums(labels, dict_all_beats):
    """
    input: labels - numpy array (0,0,0,1,0,....)
           dict_all_beats - dict {'N':0, 'S':1, 'V':2, 'U':3}
    output: label_sums - numpy.array (30, 10, 0, 5)
            total - int, number of labels         
    """
    label_sums = np.zeros(len(dict_all_beats), dtype = int)    
    (unique, counts) = np.unique(labels, return_counts=True)
    total = counts.sum()
    len_unique = len(unique)
    for i in range(len_unique):
        u = unique[i]
        label_sums[u] = counts[i]
    return label_sums, total    

def get_rid_off_class_3(test_y, pred_y):
# pred_y turi būti tokio pat ilgio, kaip ir test_y
    if (len(test_y) != len(pred_y)):
        raise Exception(f"Klaida! Nesutampa test_y ir pred_y ilgiai")     

    lst_rid_off  = []
    for i in range(len(test_y)):
        flag = (test_y[i] == 3) or (pred_y[i] == 3)
        if (flag):
           lst_rid_off.append(i)
    # https://www.codingem.com/numpy-remove-array-element/
    test_y = np.delete(test_y, lst_rid_off)
    pred_y = np.delete(pred_y, lst_rid_off)

# Atsikračius pūpsnių su klase = 3 ir suformavus masyvus, pred_y turi būti tokio pat ilgio, kaip ir test_y
    if (len(test_y) != len(pred_y)):
        raise Exception(f"Klaida! Nesutampa test_y ir pred_y ilgiai")     

    return test_y, pred_y


    
def zive_read_df_rpeaks(db_path, file_name):
    file_path = Path(db_path, file_name + '.json')
    with open(file_path,'r', encoding="utf8") as f:
        data = json.loads(f.read())
    df_rpeaks = pd.json_normalize(data, record_path =['rpeaks'])
    return df_rpeaks

def zive_read_df_data(db_path, file_name, name):
    file_path = Path(db_path, file_name + '.json')
    # df_data = pd.DataFrame()
    # path = Path(file_path)
    # https://www.askpython.com/python-modules/check-if-file-exists-in-python
    with open(file_path,'r', encoding='UTF-8', errors = 'ignore') as f:
        data = json.loads(f.read())
    df_data = pd.json_normalize(data, record_path =[name])
    return df_data

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


def get_symbol_list_modified(atr_symbols, atr_samples, seq_start, seq_end):
    # modifikuota versija get_symbol_list
    # atr_symbols - turi būti list, o ne numpy.array
    # atr_samples - turi būti list, o ne numpy.array
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


def get_annotations_table(all_beats_attr, ind_lst=None, cols_pattern=None):

    """
    Atnaujintas variantas, po to, kaip padaryti pakeitimai failų varduose 2022 03 26

    Skaičiuoja anotacijų pasiskirstymą per pacientus ir jų įrašus
    ind_lst - indeksų sąrašas, kuriuos reikia įtraukti į skaičiavimą
    Parameters
    ------------
        all_beats_attr: dataframe
        ind_lst: list
        cols_pattern: list
    Return
    -----------
        labels_table: dataframe
        labels_sums: list
    """   

    if (ind_lst is not None):
        selected_beats_attr = all_beats_attr.loc[all_beats_attr.index[ind_lst]].copy()
    else:
        selected_beats_attr = all_beats_attr.copy()
    # print(selected_beats_attr)

    selected_beats_attr['SubjCodes'] =  selected_beats_attr['userNr'].astype(str) + selected_beats_attr['recordingNr'].astype(str)
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
        # išvęsti visą lentelę
        print(labels_table)
    else:    
        count = labels_table.loc['All']
        d = count.to_dict()
        print(str(d)[1:-1])

    if (Flag2):
        # išvęsti sumarinius rodiklius
        print("\n")
        print(labels_sums)
    else:
        print("Total: ", labels_sums.loc['All'])

def get_annotations_distribution(df_list, rec_dir, beats_annot):
    # Nustatomas anotacijų pasiskirstymas per visus įrašus.
    # Pasiruošimas ciklui per pacientų įrašus
    labels_rec_all = pd.DataFrame(columns=beats_annot.keys(),dtype=int)
    labels_rec_all.insert(0,"file_name",0)
    labels_rec_all = labels_rec_all.astype({"file_name":str})
    labels_rec_all.insert(1,"userId",0)
    labels_rec_all = labels_rec_all.astype({"userId":str})

    labels_rec = []

    # Ciklas per pacientų įrašus
    for ind in df_list.index:
        file_name = df_list.loc[ind, "file_name"]
        labels_rec = np.zeros(labels_rec_all.shape[1], dtype=int)
        file_name = df_list.loc[ind, "file_name"]

        #  load data using Python JSON module
        file_path = Path(rec_dir, file_name + '.json')
        with open(file_path,'r', encoding='UTF-8', errors = 'ignore') as f:
            data = json.loads(f.read())

        df_rpeaks = pd.json_normalize(data, record_path =['rpeaks'])

        atr_sample = df_rpeaks['sampleIndex'].to_numpy()
        atr_symbol = df_rpeaks['annotationValue'].to_numpy()

        # Ciklas per visas paciento įrašo anotacijas (simbolius)
        for symbol in atr_symbol:
            # Gaunamas anotacijos simbolio numeris anotacijų sąraše
            label = beats_annot.get(symbol)
            if (label == None):
                continue
            labels_rec[label] +=1

        # Sumuojame į bendrą masyvą
        dict = {'file_name':file_name, 'userId': data['userId'], 'N':labels_rec[0], 'S':labels_rec[1],
         'V':labels_rec[2], 'U':labels_rec[3]}
        # labels_rec_all = labels_rec_all.append(dict, ignore_index = True) # taisomas
        labels_rec_all = pd.concat([labels_rec_all, pd.DataFrame([dict])], axis = 0)

    # Ciklo per pacientų įrašus pabaiga
    return labels_rec_all


def cm2df(cm, labels):
    df = pd.DataFrame()
    # rows
    for i, row_label in enumerate(labels):
        rowdata={}
        # columns
        for j, col_label in enumerate(labels): 
            rowdata[col_label]=cm[i,j]
        row_df = pd.DataFrame.from_dict({row_label:rowdata}, orient='index')
        df = pd.concat([df, row_df])
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

def confusion_matrix_modified(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes,n_classes), dtype=int)
    length = len(y_true)
    for i in range(length):
        cm[y_true[i],y_pred[i]] +=1
    return cm




