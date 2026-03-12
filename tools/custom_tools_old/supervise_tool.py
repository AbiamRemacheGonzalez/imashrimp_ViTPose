from pylab import *
from .base_tool import create_custom_file
import os


def save_supervise_pck(train_h, val_h, pck_name, path_to_save):
    pck_max = max(val_h)
    x = linspace(0, len(train_h), len(train_h))
    fig, axs = plt.subplots(1, 1, figsize=(10, 6))
    plt.plot(x, train_h, label='Train ' + pck_name, color='blue')
    plt.plot(x, val_h, label='Eval ' + pck_name, color='darkorange')
    plt.legend(handlelength=2.5)
    plt.axvline(x=val_h.index(pck_max) + 1, color='darkorange', linestyle='--')
    plt.text(val_h.index(pck_max) + 1 + 0.2, plt.ylim()[1] * 0.5 - 0.04,
             f'PCK={round(pck_max, 3)}\nEP={val_h.index(pck_max) + 1}', color='#C26B23')
    create_custom_file(os.path.join(path_to_save, "2_Max_PCK_eval  = " + str(round(pck_max, 3)) + "; EP = " + str(
        val_h.index(pck_max) + 1)) + ".txt", "")
    xlabel('Epochs')
    ylabel(pck_name)
    title(pck_name)
    plt.savefig(os.path.join(path_to_save, "1_PCK_GRPHC.png"))
    plt.clf()


def save_supervise_loss(train_h, val_info, loss_name, path_to_save, thr=0.1):
    breaks = find_stops(train_h.copy(), val_info.copy(), thr=thr)
    loss_min = min(val_info)
    fig, axs = plt.subplots(1, 1, figsize=(10, 6))
    x = linspace(0, len(train_h), len(train_h))
    plt.plot(x, train_h, label='Train ', color='blue')
    plt.plot(x, val_info, label='Eval ', color='darkorange')
    plt.legend(handlelength=1.5)
    count = 0.6
    for b in breaks:
        b += 1
        plt.axvline(x=b, color='darkorange', linestyle='--')
        plt.text(b + 0.2, plt.ylim()[1] * count, f'EP={b}', color='#C26B23')
        count -= 0.03
    plt.axvline(x=val_info.index(loss_min) + 1, color='darkorange', linestyle='--')
    plt.text(val_info.index(loss_min) + 1 + 0.2, plt.ylim()[1] * count,
             f'MIN={val_info.index(loss_min) + 1}', color='#C26B23')
    create_custom_file(
        os.path.join(path_to_save, "2_Min_" + loss_name + "_eval = " + str(round(loss_min, 6)) + "; EP = " + str(
            val_info.index(loss_min) + 1)) + ".txt", "")
    xlabel('Epochs')
    ylabel('Loss')
    title(loss_name)
    plt.savefig(os.path.join(path_to_save, "1_" + loss_name + ".png"))
    plt.clf()


def find_stops(tl, el, thr=0.1):
    e_min = 1.
    t_min = 1.
    stops = []
    for idx in range(len(tl)):
        ctl = tl[idx]
        cel = el[idx]

        if t_min >= ctl:
            t_min = ctl

        if e_min >= cel:
            e_min = cel
        else:
            diff = cel - e_min
            per = diff / e_min
            if per > thr:
                t_diff = ctl - t_min
                t_per = t_diff / t_min
                if t_per < thr:
                    stops.append(el.index(cel))
                    print(f"{el.index(cel)} = {per}")
    return stops


def find_stops_out_dated(l1, l, thr=0.1):
    min = 1.
    stops = []
    for ele in l:
        if min >= ele:
            min = ele
        else:
            diff = ele - min
            per = diff / min
            if per > thr:
                stops.append(l.index(ele))
                print(f"{l.index(ele)} = {per}")
    return stops
