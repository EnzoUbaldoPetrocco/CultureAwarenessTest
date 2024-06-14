#!/usr/bin/env python
__author__ = "Enzo Ubaldo Petrocco"
import sys

sys.path.insert(1, "../../")
import numpy as np
import os
from Utils.FileManager.FileManager import FileManagerClass
import pandas as pd
import xarray as xr


class ResultsClass:
    def __init__(self, c_cm_list) -> None:
        """
        This function uses the list of confusion matrices for creating ERRs and CICs
        :param c_cm_list: list of confusion matrices
        """
        self.pcms_list = []
        self.meanFPs = []
        self.meanFNs = []
        self.meanErrors = []
        self.meanFP_stds = []
        self.meanFN_stds = []
        self.meanError_stds = []
        if len(c_cm_list) == 0:
            print("List empty")
        else:
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

    def to_df(self, w=None):
        """
        This function converts the analysis to Pandas DataFrame
        """
        def convert_to_percentage(value):
            """
            This function converts a value to percentages, using 2 decimals
            """
            return f"{round(value*100, 4)}%"

        if len(self.pcms_list) > 0:
            ls = []
            for i in range(3):
                ls.append(convert_to_percentage(self.meanErrors[i]))
                ls.append(convert_to_percentage(self.meanError_stds[i]))
                

            ERR = convert_to_percentage(np.mean(self.meanErrors))
            ls.append(ERR)
            ERRstd = convert_to_percentage(np.sum(self.meanError_stds))
            ls.append(ERRstd)

            wERR = convert_to_percentage(np.average(self.meanErrors, weights=w))
            ls.append(wERR)
            ls.append(ERRstd)

            ls.append(convert_to_percentage(self.CIC))
            ls.append(convert_to_percentage(self.CIC_std))
            columns = []
            for i in range(3):
                columns.append(f"ERR^CULTURE {i}")
                columns.append(f"ERR^CULTURE {i} std")
            columns.append("ERR")
            columns.append("ERR std")
            columns.append("W_ERR")
            columns.append("W_ERR std")
            columns.append("CIC")
            columns.append("CIC std")
            ls = np.expand_dims(np.asarray(ls, dtype=object), 0)
            df = pd.DataFrame(ls, columns=columns)
            return df

    def print(self):
        """
        Prints the results
        """
        if len(self.pcms_list) > 0:
            for i in range(3):
                print(
                    f"For culture={i}: ERR={self.meanErrors[i]:.4f}"
                    + "\u00B1"
                    + f"{self.meanError_stds[i]:.4f}"
                )
            print(f"CIC={self.CIC:.4f}" + "\u00B1" + f"{self.CIC_std:.4f}")

    def get_pcms(self, confusion_matrix_list):
        """
        Given a confusion matrix list this function returns a percentage confusion matrix list
        """
        if np.shape(confusion_matrix_list)[0] >= 1:
            tot = self.get_tot_elements(confusion_matrix_list[0])
            pcms = []
            confusion_matrix_list = np.asarray(confusion_matrix_list)
            for i in confusion_matrix_list:
                tn = i[0, 0] / tot
                fn = i[1, 0] / tot
                tp = i[1, 1] / tot
                fp = i[0, 1] / tot
                pcm = np.array([[tn, fp], [fn, tp]])
                pcms.append(pcm)
        else:
            pcms = []
        return pcms

    def get_tot_elements(self, cm):
        """
        This function gets the total number of elements in a confusion matrix
        :param cm: confusion matrix
        """
        if len(cm) <= 0:
            return -1
        cm = np.asarray(cm)
        tot = cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[1, 1]
        return tot

    def get_statistics_pcm(self, pcms):
        """
        This function gets the mean matrix and matrix of standard deviations
        :param pcms: list of percentage confusion matrices
        :return matrix of mean values, matrix of standard deviations.
        """
        if len(pcms) > 0:
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
                stdtn_i += (tn - mean_tn) ** 2
                stdfn_i += (fn - mean_fn) ** 2
                stdtp_i += (tp - mean_tp) ** 2
                stdfp_i += (fp - mean_fp) ** 2
            std_matrix = np.array([[stdtn_i, stdfp_i], [stdfn_i, stdtp_i]])
            std_matrix = np.sqrt(std_matrix / len(pcms))
            std_matrix = np.power(std_matrix, 2)
            #print(mean_matrix)
            #print(std_matrix)
            return mean_matrix, std_matrix
        else:
            return [], []

    def get_accuracy(self, pcm):
        """
        Given a percentage confusion matrix, it returns the accuracy
        :param pcm: confusion matrix
        :return accuracy
        """
        if len(pcm) <= 0:
            return -1
        return pcm[0][0] + pcm[1][1]

    def get_error(self, pcm): 
        """
        Given a percentage confusion matrix, it returns the error
        :param pcm: confusion matrix
        :return error
        """
        if len(pcm) <= 0:
            return -1
        return 1 - self.get_accuracy(pcm)
    
    # FP
    def get_meanFP(self, pcms):
        """
        Given a percentage confusion matrix, it returns the False Positive Mean
        :param pcm: confusion matrix
        :return False Positive Mean
        """
        if len(pcms) <= 0:
            return -1
        meanpcm = self.get_statistics_pcm(pcms)[0]
        return meanpcm[0, 1]

    def get_meanFP_std(self, pcms):
        """
        Given a percentage confusion matrix, it returns the False Positive Standard Deviation
        :param pcm: confusion matrix
        :return False Positive Standard Deviation
        """
        if len(pcms) <= 0:
            return 0
        meanpcm_std = self.get_statistics_pcm(pcms)[1]
        return meanpcm_std[0, 1]

    # FN
    def get_meanFN(self, pcms):
        """
        Given a percentage confusion matrix, it returns the False Negative Mean
        :param pcm: confusion matrix
        :return False Negative Mean
        """
        if len(pcms) <= 0:
            return -1
        meanpcm = self.get_statistics_pcm(pcms)[0]
        return meanpcm[1, 0]

    def get_meanFN_std(self, pcms):
        """
        Given a percentage confusion matrix, it returns the False Negative Standard Deviation
        :param pcm: confusion matrix
        :return False Negative Standard Deviation
        """
        if len(pcms) <= 0:
            return 0
        meanpcm_std = self.get_statistics_pcm(pcms)[1]
        return meanpcm_std[1, 0]

    # TP
    def get_meanTP(self, pcms):
        """
        Given a percentage confusion matrix, it returns the True Positive Mean
        :param pcm: confusion matrix
        :return True Positive Mean
        """
        if len(pcms) <= 0:
            return -1
        meanpcm = self.get_statistics_pcm(pcms)[0]
        return meanpcm[0, 0]

    def get_meanTP_std(self, pcms):
        """
        Given a percentage confusion matrix, it returns the True Positive Standard Deviation
        :param pcm: confusion matrix
        :return True Positive Standard Deviation
        """
        if len(pcms) <= 0:
            return 0
        meanpcm_std = self.get_statistics_pcm(pcms)[1]
        return meanpcm_std[0, 0]

    # TN
    def get_meanTN(self, pcms):
        """
        Given a percentage confusion matrix, it returns the True Negative Mean
        :param pcm: confusion matrix
        :return True Negative Mean
        """
        if len(pcms) <= 0:
            return -1
        meanpcm = self.get_statistics_pcm(pcms)[0]
        return meanpcm[1, 1]

    def get_meanTN_std(self, pcms):
        """
        Given a percentage confusion matrix, it returns the True Negative Standard Deviation
        :param pcm: confusion matrix
        :return True Negative Standard Deviation
        """
        if len(pcms) <= 0:
            return 0
        meanpcm_std = self.get_statistics_pcm(pcms)[1]
        return meanpcm_std[1, 1]

    # ERROR
    def get_mean_error(self, pcms):
        """
        Given a list of percentage confusion matrix, it returns the mean error
        :param pcm: list of confusion matrices
        :return mean error
        """
        if len(pcms) <= 0:
            return -1
        mean_pcms = self.get_statistics_pcm(pcms)[0]
        return self.get_error(mean_pcms)
    

    def get_error_std(self, std_pcm):
        """
        Given a percentage confusion matrix, it returns the mean error standard deviation
        :param pcm: confusion matrix
        :return mean error standard deviation
        """
        if len(std_pcm) <= 0:
            return 0
        return std_pcm[0, 0] + std_pcm[1, 1]

    def get_mean_error_std(self, pcms):
        """
        Given a list of percentage confusion matrice, it returns the mean error
        :param pcm: list of confusion matrices
        :return mean error standard deviation
        """
        if len(pcms) <= 0:
            return 0
        std_meanpcm = self.get_statistics_pcm(pcms)[1]
        return self.get_error_std(std_meanpcm)

    # CIC
    def get_CIC(self, c_pcms):
        """
        Given a list of percentage confusion matrices subdivided in cultures it returns the CIC metric. 
        With CIC = 1/|C| * sum |ERR^C-min(ERR^C)|
        :param c_pcms: list of percentage confusion matrices subdivided in cultures
        :return CIC
        """
        # 1/|C| * sum |ERR^C-min(ERR^C)|
        if len(c_pcms) <= 0:
            return -1
        c_errors = []
        for c_pcm in c_pcms:
            errors = []
            for pcm in c_pcm:
                errors.append(self.get_error(pcm))
            c_errors.append(errors)
        ers = []
        for c_error in c_errors:
            if len(c_error) <= 0:
                ers.append(-1)
            else:
                ers.append(np.mean(c_error))
        res = 0
        for i in range(len(ers)):
            res += np.abs(ers[i] - min(ers))  # |ERR^C - min(ERR^C)|
        res = res / len(ers)
        return res

    def get_CIC_std(self, c_pcms, n_cultures=3):
        """
        Given a list of percentage confusion matrices subdivided in cultures it returns the CIC metric. 
        With CIC standard deviation
        :param c_pcms: list of percentage confusion matrices subdivided in cultures
        :param n_cultures: number of cultures
        :return CIC standard deviation
        """
        if len(c_pcms) <= 0:
            return 0
        if len(c_pcms) != n_cultures:
            print(
                "The number of confusion matrices does not correspond to the number of cultures"
            )
        culture_mean_pcms = []
        culture_mean_std_pcms = []
        for i in range(len(c_pcms)):
            culture_mean_pcm, culture_mean_std_pcm = self.get_statistics_pcm(c_pcms[i])
            culture_mean_pcms.append(culture_mean_pcm)
            culture_mean_std_pcms.append(culture_mean_std_pcm)
        std_errors = []
        errors = []

        for std_pcm in culture_mean_std_pcms:
            std_errors.append(self.get_error_std(std_pcm))
        for pcm in culture_mean_pcms:
            errors.append(self.get_error(pcm))

        i = errors.index(min(errors))
        return (np.sum(std_errors) - std_errors[i]) / len(errors)

    # Precision = TP / (TP+FP)
    def get_meanPrecision(self, pcms):
        if len(pcms) <= 0:
            return -1
        meantp = self.get_meanTP(pcms)
        meanfp = self.get_meanFP(pcms)
        return meantp / (meantp + meanfp)

    def get_meanPrecision_std(self, pcms):
        if len(pcms) <= 0:
            return 0
        # stdFP = |dPrec/dtp|stdtp + |dPrec/dfp|stdfp =
        # = (FP/(FP+TP)^2)*STDTP + (TP/(FP+TP)^2)*STDFP
        meantp = self.get_meanTP(pcms)
        stdtp = self.get_meanTP_std(pcms)
        meanfp = self.get_meanFP(pcms)
        stdfp = self.get_meanFP_std(pcms)
        denominator = (meanfp + meantp) ** 2
        numerator = meanfp * stdtp + meantp * stdfp
        return numerator / denominator

    # Precision = TP / (TP+FN)
    def get_meanRecall(self, pcms):
        if len(pcms) <= 0:
            return -1
        meantp = self.get_meanTP(pcms)
        meanfn = self.get_meanFN(pcms)
        return meantp / (meantp + meanfn)

    def get_meanRecall_std(self, pcms):
        if len(pcms) <= 0:
            return 0
        # stdFP = |dPrec/dtp|stdtp + |dPrec/dfn|stdfn =
        # = (FN/(FN+TP)^2)*STDTP + (TP/(FN+TP)^2)*STDFN
        meantp = self.get_meanTP(pcms)
        stdtp = self.get_meanTP_std(pcms)
        meanfn = self.get_meanFN(pcms)
        stdfn = self.get_meanFN_std(pcms)
        denominator = (meanfn + meantp) ** 2
        numerator = meanfn * stdtp + meantp * stdfn
        return numerator / denominator


class ResAcquisitionClass:
    def buildPath(
        self,
        basePath,
        standard,
        alg,
        lamp,
        culture,
        percent,
        augment,
        adversary,
        lambda_index,
        taugment,
        tadversary,
        tgaug,
        teps,
        t_cult=0,
        out=0,
        g_augment=0
    ):
        if standard:
            basePath = basePath + "STD/" + alg
        else:
            basePath = basePath + "MIT/" + alg
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
        basePath = basePath + c + str(percent) + "/"
        if augment:
            if adversary:
                aug = f"TOTAUG/g={g_augment}/"
            else:
                aug = f"STDAUG/g={g_augment}/"
        else:
            if adversary:
                aug = "AVD/"
            else:
                aug = "NOAUG/"

        basePath = basePath + aug
        if not standard:
            basePath = basePath + str(lambda_index) + "/"
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

        basePath = basePath + testaug
        if standard:
            basePath = basePath + "res.csv"
        else:
            basePath = basePath + "out " + str(out) + ".csv"
        return basePath

    def get_cm_list(self, path):
        try:
            fm = FileManagerClass(path, create=False)
            cm_list = fm.readcms()
        except:
            cm_list = []
            print(
                f"Error in function 'acquire_cm_list' while trying to read from path: {path}"
            )
        return cm_list

    def get_cm_structure(self, basePath):
        standards = [1]
        alg = "DL"
        lamps = [0, 1]
        cultures = [0, 1, 2]
        percents = [0.05]
        augments = [0, 1]
        g_augments = [ 0.05, 0.1, 0.5, 0.75]
        adversary = [0]
        lambda_indeces = range(-1, 13)
        taugments = [0]
        tadversaries = [0]
        test_g_augs = [0.01, 0.05, 0.1]
        test_eps = [0.0005, 0.001, 0.005]
        t_cults = [0, 1, 2]

        for standard in standards:
            for lamp in lamps:
                for culture in cultures:
                    for percent in percents:
                        for augment in augments:
                            if augment:
                                for g_augment in g_augments:
                                    for adv in adversary:
                                        for adv in adversary:
                                            if standard == 0:
                                                for lambda_index in lambda_indeces:
                                                    for taugment in taugments:
                                                        for tadversary in tadversaries:
                                                            for tgaug in test_g_augs:
                                                                for teps in test_eps:
                                                                    tcultsl = []
                                                                    for t_cult in t_cults:
                                                                        path = self.buildPath(
                                                                            basePath,
                                                                            standard,
                                                                            alg,
                                                                            lamp,
                                                                            culture,
                                                                            percent,
                                                                            augment,
                                                                            adv,
                                                                            lambda_index,
                                                                            taugment,
                                                                            tadversary,
                                                                            tgaug,
                                                                            teps,
                                                                            t_cult,
                                                                            t_cult,
                                                                            g_augment=g_augment
                                                                        )
                                                                        outsl = self.get_cm_list(
                                                                            path
                                                                        )
                                                                        tcultsl.append(outsl)

                                                                    tempst = path.split("/")
                                                                    tempst2 = "./"
                                                                    for i in range(
                                                                        3, len(tempst) - 2
                                                                    ):
                                                                        tempst2 += tempst[i] + "/"
                                                                    st = tempst2
                                                                    dir = os.path.dirname(st)
                                                                    mkdir(dir)
                                                                    rc = ResultsClass(
                                                                        np.asarray(tcultsl)
                                                                    )
                                                                    weights = [percent]*len(cultures)
                                                                    weights[culture]= weights[culture]/percent
                                                                    data = rc.to_df(weights)
                                                                    data.to_csv(st + "res.csv")
                                            else:
                                                for taugment in taugments:
                                                    for tadversary in tadversaries:
                                                        for tgaug in test_g_augs:
                                                            for teps in test_eps:
                                                                tcultsl = []
                                                                for t_cult in t_cults:
                                                                    path = self.buildPath(
                                                                        basePath,
                                                                        standard,
                                                                        alg,
                                                                        lamp,
                                                                        culture,
                                                                        percent,
                                                                        augment,
                                                                        adv,
                                                                        0,
                                                                        taugment,
                                                                        tadversary,
                                                                        tgaug,
                                                                        teps,
                                                                        t_cult,
                                                                        t_cult,
                                                                        g_augment=g_augment
                                                                    )
                                                                    outsl = self.get_cm_list(path)
                                                                    tcultsl.append(outsl)

                                                                tempst = path.split("/")
                                                                tempst2 = "./"
                                                                for i in range(3, len(tempst) - 2):
                                                                    tempst2 += tempst[i] + "/"
                                                                st = tempst2
                                                                dir = os.path.dirname(st)
                                                                mkdir(dir)
                                                                rc = ResultsClass(
                                                                    np.asarray(tcultsl)
                                                                )
                                                                weights = [percent]*len(cultures)
                                                                weights[culture]= weights[culture]/percent
                                                                data = rc.to_df(weights)
                                                                data.to_csv(st + "res.csv")
                            else:
                                for adv in adversary:
                                    if standard == 0:
                                        for lambda_index in lambda_indeces:
                                            for taugment in taugments:
                                                for tadversary in tadversaries:
                                                    for tgaug in test_g_augs:
                                                        for teps in test_eps:
                                                            tcultsl = []
                                                            for t_cult in t_cults:
                                                                path = self.buildPath(
                                                                    basePath,
                                                                    standard,
                                                                    alg,
                                                                    lamp,
                                                                    culture,
                                                                    percent,
                                                                    augment,
                                                                    adv,
                                                                    lambda_index,
                                                                    taugment,
                                                                    tadversary,
                                                                    tgaug,
                                                                    teps,
                                                                    t_cult,
                                                                    t_cult,
                                                                )
                                                                outsl = self.get_cm_list(
                                                                    path
                                                                )
                                                                tcultsl.append(outsl)

                                                            tempst = path.split("/")
                                                            tempst2 = "./"
                                                            for i in range(
                                                                3, len(tempst) - 2
                                                            ):
                                                                tempst2 += tempst[i] + "/"
                                                            st = tempst2
                                                            dir = os.path.dirname(st)
                                                            mkdir(dir)
                                                            rc = ResultsClass(
                                                                np.asarray(tcultsl)
                                                            )
                                                            weights = [percent]*len(cultures)
                                                            weights[culture]= weights[culture]/percent
                                                            data = rc.to_df(weights)
                                                            data.to_csv(st + "res.csv")
                                    else:
                                        for taugment in taugments:
                                            for tadversary in tadversaries:
                                                for tgaug in test_g_augs:
                                                    for teps in test_eps:
                                                        tcultsl = []
                                                        for t_cult in t_cults:
                                                            path = self.buildPath(
                                                                basePath,
                                                                standard,
                                                                alg,
                                                                lamp,
                                                                culture,
                                                                percent,
                                                                augment,
                                                                adv,
                                                                0,
                                                                taugment,
                                                                tadversary,
                                                                tgaug,
                                                                teps,
                                                                t_cult,
                                                                t_cult,
                                                            )
                                                            outsl = self.get_cm_list(path)
                                                            tcultsl.append(outsl)

                                                        tempst = path.split("/")
                                                        tempst2 = "./"
                                                        for i in range(3, len(tempst) - 2):
                                                            tempst2 += tempst[i] + "/"
                                                        st = tempst2
                                                        dir = os.path.dirname(st)
                                                        mkdir(dir)
                                                        rc = ResultsClass(
                                                            np.asarray(tcultsl)
                                                        )
                                                        weights = [percent]*len(cultures)
                                                        weights[culture]= weights[culture]/percent
                                                        data = rc.to_df(weights)
                                                        data.to_csv(st + "res.csv")
                                                        



def mkdir(dir):
    try:
        if not os.path.exists(dir):
            print(f"Making directory: {str(dir)}")
            os.makedirs(dir)
    except Exception as e:
        print(f"{dir} Not created")


def main():
    rac = ResAcquisitionClass()
    basepath = "../../Mitigated/"
    rac.get_cm_structure(basepath)


if __name__ == "__main__":
    main()
