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


def apply_FDA(train_set_idx, all_beats_attr):
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

    for idx in train_set_idx:
        #idx =7883
        wl_side = math.floor(all_beats_attr.loc[idx][5] * fraction_to_drop_l)
        wr_side = math.floor(all_beats_attr.loc[idx][6] * fraction_to_drop_r)
        seq_1d, sample, label, RRl, RRr = read_seq_RR_arr(db_path, all_beats_attr, idx, wl_side, wr_side,nl_RR, nr_RR)
        
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

        # print(f'dictRR: {dictRR} ')

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


