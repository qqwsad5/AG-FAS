import math
import numpy as np

def eval_state(probs, labels, thr):
    predict = probs >= thr
    TN = np.sum((labels == 0) & (predict == False))
    FN = np.sum((labels == 1) & (predict == False))
    FP = np.sum((labels == 0) & (predict == True))
    TP = np.sum((labels == 1) & (predict == True))
    return TN, FN, FP, TP

def calculate(probs, labels, threshold=0.5):
    TN, FN, FP, TP = eval_state(probs, labels, threshold)
    APCER = 1.0 if (FP + TN == 0) else FP / float(FP + TN)
    NPCER = 1.0 if (FN + TP == 0) else FN / float(FN + TP)
    ACER = (APCER + NPCER) / 2.0
    ACC = (TP + TN) / labels.shape[0]
    return APCER, NPCER, ACER, ACC

def get_best_ACER(probs, labels, grid_density = 10000):
    thresholds = get_threshold(probs, grid_density)
    min_ACER = 1.0
    min_ACER_states = []
    for thr in thresholds:
        APCER, NPCER, ACER, ACC = calculate(probs, labels, thr)
        if ACER <= min_ACER:
            min_ACER = ACER
            min_ACER_states = [APCER, NPCER, ACER, ACC, thr]
    return min_ACER_states

def calculate_threshold(probs, labels, threshold):
    TN, FN, FP, TP = eval_state(probs, labels, threshold)
    ACC = (TP + TN) / labels.shape[0]
    return ACC

def get_threshold(probs, grid_density):
    Min, Max = min(probs), max(probs)
    thresholds = []
    for i in range(grid_density + 1):
        thresholds.append(0.0 + i * 1.0 / float(grid_density))
    thresholds.append(1.1)
    return thresholds

def get_EER_states(probs, labels, grid_density = 10000):
    thresholds = get_threshold(probs, grid_density)
    min_dist = 1.0
    min_dist_states = []
    FRR_list = []
    FAR_list = []
    for thr in thresholds:
        TN, FN, FP, TP = eval_state(probs, labels, thr)
        if(FN + TP == 0):
            FRR = TPR = 1.0
            FAR = FP / float(FP + TN)
            TNR = TN / float(TN + FP)
        elif(FP + TN == 0):
            TNR = FAR = 1.0
            FRR = FN / float(FN + TP)
            TPR = TP / float(TP + FN)
        else:
            FAR = FP / float(FP + TN)
            FRR = FN / float(FN + TP)
            TNR = TN / float(TN + FP)
            TPR = TP / float(TP + FN)
        dist = math.fabs(FRR - FAR)
        FAR_list.append(FAR)
        FRR_list.append(FRR)
        if dist <= min_dist:
            min_dist = dist
            min_dist_states = [FAR, FRR, thr]
    EER = (min_dist_states[0] + min_dist_states[1]) / 2.0
    thr = min_dist_states[2]
    return EER, thr, FRR_list, FAR_list

def get_HTER_at_thr(probs, labels, thr):
    TN, FN, FP, TP = eval_state(probs, labels, thr)
    if (FN + TP == 0):
        FRR = 1.0
        FAR = FP / float(FP + TN)
    elif(FP + TN == 0):
        FAR = 1.0
        FRR = FN / float(FN + TP)
    else:
        FAR = FP / float(FP + TN)
        FRR = FN / float(FN + TP)
    HTER = (FAR + FRR) / 2.0
    return HTER

# TPR @ FPR
def get_TPR_at_FPR_states(probs, labels, grid_density = 10000, FPR_list = [0.1, 0.01, 0.001]):
    thresholds = get_threshold(probs, grid_density)
    min_dist = [1.0 for _ in range(len(FPR_list))]
    min_dist_states = [[] for _ in range(len(FPR_list))]
    FRR_list = []
    FAR_list = []
    for thr in thresholds:
        TN, FN, FP, TP = eval_state(probs, labels, thr)
        if(FN + TP == 0):
            FRR = TPR = 1.0
            FPR = FAR = FP / float(FP + TN)
            TNR = TN / float(TN + FP)
        elif(FP + TN == 0):
            TNR = FPR = FAR = 1.0
            FRR = FN / float(FN + TP)
            TPR = TP / float(TP + FN)
        else:
            FPR = FAR = FP / float(FP + TN)
            FRR = FN / float(FN + TP)
            TNR = TN / float(TN + FP)
            TPR = TP / float(TP + FN)
        dist = [math.fabs(FPR - FPR_list[i]) for i in range(len(FPR_list))]
        for i in range(len(FPR_list)):
            if dist[i] <= min_dist[i]:
                min_dist[i] = dist[i]
                min_dist_states[i] = [TPR, FPR, thr]
    return min_dist_states