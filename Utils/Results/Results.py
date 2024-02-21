import csv
import numpy as np
import os

class ResultsClass:
    def __init__(self, c_cm_list) -> None:
        self.pcms_list = []
        self.meanFPs = []
        self.meanFNs = []
        self.meanErrors= []
        self.meanFP_stds = []
        self.meanFN_stds = []
        self.meanError_stds= []
        for i in range(len(c_cm_list)):
            pcm = self.get_pcms(c_cm_list[i])
            self.pcms_list.append(pcm)
            self.meanFPs.append(self.get_meanFP(pcm))
            self.meanFNs.append(self.get_meanFN(pcm))
            self.meanErrors.append(self.get_mean_error(pcm))
            self.meanFP_stds.append(self.get_meanFP_std(pcm))
            self.meanFN_stds.append(self.get_meanFN_std(pcm))
            self.meanError_stds.append(self.get_mean_error_std(pcm))

        self.CIC = self.get_CIC(self.pcms_list)
        self.CIC_std = self.get_CIC_std(self.pcms_list, len(self.pcms_list))

    def get_pcms(self, confusion_matrix_list):
        tot = self.get_tot_elements(confusion_matrix_list[0])
        pcms = []
        for i in confusion_matrix_list:
            tn = (i[0, 0] / tot) * 100
            fn = (i[1, 0] / tot) * 100
            tp = (i[1, 1] / tot) * 100
            fp = (i[0, 1] / tot) * 100
            pcm = np.array(
                [[tn, fp], [fn, tp]]
            )
            pcms.append(pcm)
        return pcms

    def get_tot_elements(self, cm):
        tot = cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[1, 1]
        return tot

    def get_statistics_pcm(self, pcms):
        count_tn = 0
        count_fn = 0
        count_tp = 0
        count_fp = 0
        stdtn_i = 0
        stdtp_i = 0
        stdfn_i = 0
        stdfp_i = 0
        for i in pcms:
            tn = i[0, 0]
            fn = i[1, 0]
            tp = i[1, 1]
            fp = i[0, 1]

            count_tn += tn
            count_fn += fn
            count_fp += fp
            count_tp += tp

        mean_tn = count_tn / len(pcms)
        mean_fn = count_fn / len(pcms)
        mean_tp = count_tp / len(pcms)
        mean_fp = count_fp / len(pcms)

        mean_matrix = np.array(
            [
                [mean_tn, mean_fp],
                [mean_fn, mean_tp],
            ]
        )

        for i in pcms:
            tn = i[0, 0]
            fn = i[1, 0]
            tp = i[1, 1]
            fp = i[0, 1]
            stdtn_i += (tn - mean_tn)**2
            stdfn_i += (fn - mean_fn)**2
            stdtp_i += (tp - mean_tp)**2
            stdfp_i += (fp - mean_fp)**2
        std_matrix = np.array(
            [stdtn_i, stdfp_i],
            [stdfn_i, stdtp_i]
        )
        std_matrix = std_matrix / np.sqrt(len(pcms)-1)
        return mean_matrix, std_matrix

    def get_accuracy(self, pcm):
        return pcm[0,0]+pcm[1,1]
    
    def get_error(self, pcm):
        return 1-self.get_accuracy(pcm)
    
    
    # FP
    def get_meanFP(self, pcms):
        meanpcm = self.get_statistics_pcm(pcms)[0] 
        return meanpcm[0, 1]
    
    def get_meanFP_std(self, pcms):
        meanpcm_std = self.get_statistics_pcm(pcms)[1] 
        return meanpcm_std[0, 1]
    
    #FN
    def get_meanFN(self, pcms):
        meanpcm = self.get_statistics_pcm(pcms)[0] 
        return meanpcm[1, 0]
    
    def get_meanFN_std(self, pcms):
        meanpcm_std = self.get_statistics_pcm(pcms)[1] 
        return meanpcm_std[1, 0]
        
    #TP
    def get_meanTP(self, pcms):
        meanpcm = self.get_statistics_pcm(pcms)[0] 
        return meanpcm[0, 0]
    
    def get_meanTP_std(self, pcms):
        meanpcm_std = self.get_statistics_pcm(pcms)[1] 
        return meanpcm_std[0, 0]
    
    #TN
    def get_meanTN(self, pcms):
        meanpcm = self.get_statistics_pcm(pcms)[0] 
        return meanpcm[1, 1]
    
    def get_meanTN_std(self, pcms):
        meanpcm_std = self.get_statistics_pcm(pcms)[1] 
        return meanpcm_std[1, 1]
    
    # ERROR
    def get_mean_error(self, pcms):
        mean_pcms = self.get_statistics_pcm(pcms)[0]
        return self.get_error(mean_pcms)
    
    def get_error_std(self, std_pcm):
        return std_pcm[0,0]+std_pcm[1,1]
    
    def get_mean_error_std(self, pcms):
        std_meanpcm = self.get_statistics_pcm(pcms)[1]
        return self.get_error_std(std_meanpcm)
    
    # CIC
    def get_CIC(self, c_pcms):
        errors = []
        for pcm in c_pcms:
            errors.append(self.get_error(pcm))
        return np.sum(errors)- min(errors)

    def get_CIC_std(self, c_pcms, n_cultures=3):
        if len(c_pcms)!=n_cultures:
            print("The number of confusion matrices does not correspond to the number of cultures")
        culture_mean_pcms, culture_mean_std_pcms = [], []
        for i in range(n_cultures):
            culture_mean_pcm, culture_mean_std_pcm = self.get_statistics_pcm(c_pcms[n_cultures])
            culture_mean_pcms.append(culture_mean_pcm)
            culture_mean_std_pcms.append(culture_mean_std_pcm)
        std_errors = []
        errors = []
        for std_pcm in culture_mean_std_pcms:
            std_errors.append(self.get_error_std(std_pcm))
        for pcm in culture_mean_pcms:
            errors.append(self.get_error(pcm))
        i = errors.index(min(errors))
        return np.sum(std_errors) + std_errors[i]
    
    # Precision = TP / (TP+FP)
    def get_meanPrecision(self, pcms):
        meantp = self.get_meanTP(pcms)
        meanfp = self.get_meanFP(pcms)
        return meantp / (meantp + meanfp)
    
    def get_meanPrecision_std(self, pcms):
        # stdFP = |dPrec/dtp|stdtp + |dPrec/dfp|stdfp = 
        # = (FP/(FP+TP)^2)*STDTP + (TP/(FP+TP)^2)*STDFP
        meantp = self.get_meanTP(pcms)
        stdtp = self.get_meanTP_std(pcms)
        meanfp = self.get_meanFP(pcms)
        stdfp = self.get_meanFP_std(pcms)
        denominator = (meanfp + meantp)**2
        numerator = meanfp*stdtp + meantp*stdfp
        return numerator/denominator
    
    # Precision = TP / (TP+FN)
    def get_meanRecall(self, pcms):
        meantp = self.get_meanTP(pcms)
        meanfn = self.get_meanFN(pcms)
        return meantp / (meantp + meanfn)
    
    def get_meanRecall_std(self, pcms):
        # stdFP = |dPrec/dtp|stdtp + |dPrec/dfn|stdfn = 
        # = (FN/(FN+TP)^2)*STDTP + (TP/(FN+TP)^2)*STDFN
        meantp = self.get_meanTP(pcms)
        stdtp = self.get_meanTP_std(pcms)
        meanfn = self.get_meanFN(pcms)
        stdfn = self.get_meanFN_std(pcms)
        denominator = (meanfn + meantp)**2
        numerator = meanfn*stdtp + meantp*stdfn
        return numerator/denominator



class ResultsPathClass:
    def buildPath(basePath, standard, alg, lamp, culture, augment, adversary, lambda_index, taugment, tadversary, tgaug, teps, t_cult, out):
        if standard:
            basePath = "./STD/" + alg
        else:
            basePath = "./MIT/" + alg
        if lamp:
            if culture == 0:
                c = "/LC/"
            elif culture == 1:
                c = "/LF/"
            elif culture == 2:
                c = "/LT/"
            else:
                c = "/LC/"
        else:
            if culture == 0:
                c = "/CI/"
            elif culture == 1:
                c = "/CJ/"
            elif culture == 2:
                c = "/CS/"
            else:
                c = "/CI/"
        basePath = basePath + c
        if augment:
            if adversary:
                aug = "TOTAUG/"
            else:
                aug = "STDAUG/"
        else:
            if adversary:
                aug = "AVD/"
            else:
                aug = "NOAUG/"
        basePath = basePath + aug + str(lambda_index) + "/"
        
        if taugment:
            if tadversary:
                testaug = f"TTOTAUG/G_AUG={tgaug}/EPS={teps}/"
            else:
                testaug = f"TSTDAUG/G_AUG={tgaug}/"
        else:
            if tadversary:
                testaug = f"TAVD/EPS={teps}/"
            else:
                testaug = f"TNOAUG/"
        testaug = testaug + f"CULTURE{t_cult}/"

        basePath = basePath + testaug + "out " + str(out) + ".csv"