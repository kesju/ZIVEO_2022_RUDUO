# Moduliai perdirbti iš 9_selected_annotations_graphs_zive&mit.ipynb

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_sequences_min_max(signal, df_gr, w_side):
# Ieškomi min ir max EKG reikšmės klasterio pūpsniuose
    min_clt = 0
    max_clt = 0
    for index, row in df_gr.iterrows():
        seq_start = row['atr_sample'] - w_side
        if (seq_start < 0):
            continue
        seq_end = row['atr_sample'] + w_side
        if (seq_end >= len(signal)):
            continue
        sequence = signal[seq_start:seq_end] 

        # deltax ir deltay simbolių pozicijų koregavimui
        min = np.amin(sequence)
        if (min < min_clt):
            min_clt = min
        max = np.amax(sequence)
        if (max > max_clt):
            max_clt = max
    return min_clt, max_clt

def show_beats(signal, df_gr, w_side, min, max, fig_width, fig_height, max_graphs=None):
# Atvaizduojami klasterio pūpsniai kiekvienas atskirai
    """
    Parameters
    ------------
        signal: numpy array, float
        df_gr: dataframe after grouping by 'cluster'
        w_side: window width from rpeak, int
        min: float
        max: float
        fig_width: int
        fig_height: int
        max_graphs: max of graphs, int
    """

    seq_nr = 0
    for index, row in df_gr.iterrows():
        # Formuojami grafiniai vaizdai
        seq_start = row['atr_sample'] - w_side
        if (seq_start < 0):
            continue
        seq_end = row['atr_sample'] + w_side
        if (seq_end >= len(signal)):
            continue
        beat_loc = row['atr_sample'] - seq_start
        beat_symbol = row['atr_symbol']
        sequence = signal[seq_start:seq_end] 
        
        dict_attr = {'seq_nr':seq_nr, 'rpeak':row['atr_sample'], 'symbol':beat_symbol}
        print(dict_attr)

        deltay = (max - min)/20
        deltax = len(sequence)/100

        # suformuojame vaizdą
        fig = plt.figure(facecolor=(1, 1, 1), figsize=(fig_width, fig_height))
        ax = plt.gca()
        x = np.arange(0, len(sequence), 1)
        ax.plot(x, sequence, color="#6c3376", linewidth=2)
        ax.set_ylim([min, max+2*deltay])
        ax.annotate(beat_symbol, (beat_loc - deltax,sequence[beat_loc] + deltay))

        seq_nr += 1
        plt.show()
        if (max_graphs != None):
            if (seq_nr >= max_graphs):
                break

def show_beats_in_same_plot(signal, df_gr, w_side,  min, max, fig_width, fig_height):
# Atvaizduojami klasterio pūpsniai klojant vienas ant kito
    """
    Parameters
    ------------
        signal: numpy array, float
        df_gr: dataframe after grouping by 'cluster'
        w_side: window width from rpeak to one side
        min: float
        max: float
        fig_width: int
        fig_height: int
    """
    deltay = (max - min)/20

# Formuojami pūpsnių grafiniai vaizdai ir klojami vienas ant kito
    fig = plt.figure(facecolor=(1, 1, 1), figsize=(fig_width, fig_height))
    ax = plt.gca()
    seq_nr = 0
    for index, row in df_gr.iterrows():
        # Formuojami grafiniai vaizdai
        seq_start = row['atr_sample'] - w_side
        if (seq_start < 0):
            continue
        seq_end = row['atr_sample'] + w_side
        if (seq_end >= len(signal)):
            continue
        # beat_rpeak = row['atr_sample']
        # beat_symbol = row['atr_symbol']
        sequence = signal[seq_start:seq_end] 
        
        # suformuojame vaizdą
        x = np.arange(0, len(sequence), 1)
        ax.plot(x, sequence, color="#6c3376", linewidth=2)
        ax.set_ylim([min, max+2*deltay])
        seq_nr += 1
    plt.show()
