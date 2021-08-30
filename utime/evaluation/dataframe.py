import numpy as np
import pandas as pd


def get_eval_df(sequencer):
    ids = [ss.identifier for ss in sequencer.pairs]
    classes = ["mean"] + ["cls {}".format(i) for i in range(sequencer.n_classes)]
    return pd.DataFrame(columns=ids, index=classes)

def get_eval_plasma_states_df(sequencer):
    ids = [ss.identifier for ss in sequencer.dataset]
    classes = ["mean"] + ["cls {}".format(i) for i in range(sequencer.no_classes)]
    return pd.DataFrame(columns=ids, index=classes)

def add_to_eval_df(eval_dict, id_, values):
    mean = np.nanmean(values)
    values = [mean] + list(values)
    eval_dict[id_] = values


def with_grand_mean_col(eval_dict, col_name="Grand mean"):
    means = np.mean(eval_dict, axis=1)
    eval_dict[col_name] = means
    cols = list(eval_dict.columns)
    cols.append(cols.pop(cols.index(col_name)))
    return eval_dict.loc[:, cols]


def log_eval_df_to_screen(eval_dict, logger, round=4, txt=None):
    log = "[*] {}".format(txt or "EVALUATION RESULTS")
    logger("\n" + log)
    logger("-"*len(log))
    logger(eval_dict.round(round))
    logger("-"*len(log))


def log_eval_df_to_file(eval_dict, out_csv_file=None, out_txt_file=None, round=4):
    if out_csv_file:
        with open(out_csv_file, "w+") as out_csv:
            out_csv.write(eval_dict.to_csv())
    if out_txt_file:
        with open(out_txt_file, "w+") as out_txt:
            out_txt.write(eval_dict.round(round).to_string())


def log_eval_df(eval_dict, logger, out_csv_file=None, out_txt_file=None,
                round=4, txt=None):
    log_eval_df_to_screen(eval_dict, logger, round, txt)
    log_eval_df_to_file(eval_dict, out_csv_file, out_txt_file, round)
