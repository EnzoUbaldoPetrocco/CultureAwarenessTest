import sys

sys.path.insert(1, "../../")
import numpy as np
import os
from Utils.FileManager.FileManager import FileManagerClass
import pandas as pd
import xarray as xr


class ResultsClass:
    def __init__(self, c_cm_list) -> None:
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

    def to_df(self):
        if len(self.pcms_list) > 0:
            ls = []
            for i in range(3):
                ls.append(
                    [
                        round(self.meanErrors[i], 4),
                        round(self.meanError_stds[i], 4),
                        round(self.CIC, 4),
                        round(self.CIC_std, 4),
                        round(self.meanFNs[i], 4),
                        round(self.meanFN_stds[i], 4),
                        round(self.meanFPs[i], 4),
                        round(self.meanFP_stds[i], 4),
                    ]
                )
            indeces = [f"CULTURE {i}" for i in range(3)]
            df = pd.DataFrame(
                ls,
                columns=[
                    "ERR",
                    "ERR std",
                    "CIC",
                    "CIC std",
                    "FN",
                    "FN std",
                    "FP",
                    "FP std",
                ],
                index=indeces,
            )
            return df

    def print(self):
        if len(self.pcms_list) > 0:
            for i in range(3):
                print(
                    f"For culture={i}: ERR={self.meanErrors[i]:.4f}"
                    + "\u00B1"
                    + f"{self.meanError_stds[i]:.4f}"
                )
            print(f"CIC={self.CIC:.4f}" + "\u00B1" + f"{self.CIC_std:.4f}")

    def get_pcms(self, confusion_matrix_list):
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
        if len(cm) <= 0:
            return -1
        cm = np.asarray(cm)
        tot = cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[1, 1]
        return tot

    def get_statistics_pcm(self, pcms):
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
            std_matrix = std_matrix / np.sqrt(len(pcms) - 1)
            return mean_matrix, std_matrix
        else:
            return [], []

    def get_accuracy(self, pcm):
        if len(pcm) <= 0:
            return -1
        return pcm[0][0] + pcm[1][1]

    def get_error(self, pcm):
        if len(pcm) <= 0:
            return -1
        return 1 - self.get_accuracy(pcm)

    # FP
    def get_meanFP(self, pcms):
        if len(pcms) <= 0:
            return -1
        meanpcm = self.get_statistics_pcm(pcms)[0]
        return meanpcm[0, 1]

    def get_meanFP_std(self, pcms):
        if len(pcms) <= 0:
            return 0
        meanpcm_std = self.get_statistics_pcm(pcms)[1]
        return meanpcm_std[0, 1]

    # FN
    def get_meanFN(self, pcms):
        if len(pcms) <= 0:
            return -1
        meanpcm = self.get_statistics_pcm(pcms)[0]
        return meanpcm[1, 0]

    def get_meanFN_std(self, pcms):
        if len(pcms) <= 0:
            return 0
        meanpcm_std = self.get_statistics_pcm(pcms)[1]
        return meanpcm_std[1, 0]

    # TP
    def get_meanTP(self, pcms):
        if len(pcms) <= 0:
            return -1
        meanpcm = self.get_statistics_pcm(pcms)[0]
        return meanpcm[0, 0]

    def get_meanTP_std(self, pcms):
        if len(pcms) <= 0:
            return 0
        meanpcm_std = self.get_statistics_pcm(pcms)[1]
        return meanpcm_std[0, 0]

    # TN
    def get_meanTN(self, pcms):
        if len(pcms) <= 0:
            return -1
        meanpcm = self.get_statistics_pcm(pcms)[0]
        return meanpcm[1, 1]

    def get_meanTN_std(self, pcms):
        if len(pcms) <= 0:
            return 0
        meanpcm_std = self.get_statistics_pcm(pcms)[1]
        return meanpcm_std[1, 1]

    # ERROR
    def get_mean_error(self, pcms):
        if len(pcms) <= 0:
            return -1
        mean_pcms = self.get_statistics_pcm(pcms)[0]
        return self.get_error(mean_pcms)

    def get_error_std(self, std_pcm):
        if len(std_pcm) <= 0:
            return 0
        return std_pcm[0, 0] + std_pcm[1, 1]

    def get_mean_error_std(self, pcms):
        if len(pcms) <= 0:
            return 0
        std_meanpcm = self.get_statistics_pcm(pcms)[1]
        return self.get_error_std(std_meanpcm)

    # CIC
    def get_CIC(self, c_pcms):
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
        t_cult,
        out,
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
                aug = "TOTAUG/"
            else:
                aug = "STDAUG/"
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
        standards = [0, 1]
        alg = "DL"
        lamps = [0, 1]
        cultures = [0, 1, 2]
        percents = [0.05, 0.1]
        augments = [0, 1]
        adversary = [0, 1]
        lambda_indeces = range(-1, 13)
        taugments = [0, 1]
        tadversaries = [0, 1]
        test_g_augs = [0.01, 0.05, 0.1]
        test_eps = [0.0005, 0.001, 0.005]
        t_cults = [0, 1, 2]

        structure = []
        for standard in standards:
            lampsl = []
            for lamp in lamps:
                culturesl = []
                for culture in cultures:
                    percentsl = []
                    for percent in percents:
                        augmentsl = []
                        for augment in augments:
                            adversaryl = []
                            for adv in adversary:
                                if standard == 0:
                                    lambda_indecesl = []
                                    for lambda_index in lambda_indeces:
                                        taugmentsl = []
                                        for taugment in taugments:
                                            tadversariesl = []
                                            for tadversary in tadversaries:
                                                test_g_augsl = []
                                                for tgaug in test_g_augs:
                                                    tepsl = []
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
                                                        tepsl.append(tcultsl)
                                                    test_g_augsl.append(tepsl)
                                                tadversariesl.append(test_g_augsl)
                                            taugmentsl.append(tadversariesl)
                                        lambda_indecesl.append(taugmentsl)
                                    adversaryl.append(lambda_indecesl)
                                else:
                                    taugmentsl = []
                                    for taugment in taugments:
                                        tadversariesl = []
                                        for tadversary in tadversaries:
                                            test_g_augsl = []
                                            for tgaug in test_g_augs:
                                                tepsl = []
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
                                                            "",
                                                        )
                                                        outsl = self.get_cm_list(path)
                                                        tcultsl.append(outsl)
                                                    tepsl.append(tcultsl)
                                                test_g_augsl.append(tepsl)
                                            tadversariesl.append(test_g_augsl)
                                        taugmentsl.append(tadversariesl)
                                adversaryl.append(taugmentsl)
                            augmentsl.append(adversaryl)
                        percentsl.append(augmentsl)
                    culturesl.append(percentsl)
                lampsl.append(culturesl)
            structure.append(lampsl)
        return structure

    def save_results(self, cm_list):
        cultures = [0, 1, 2]
        percents = [0.05, 0.1]
        lambda_indeces = range(-1, 13)
        test_g_augs = [0.01, 0.05, 0.1]
        test_eps = [0.0005, 0.001, 0.005]
        for standard in range(len(cm_list)):
            for lamp in range(len(cm_list[standard])):
                for culture in range(len(cm_list[standard][lamp])):
                    for percent in range(len(cm_list[standard][lamp][culture])):
                        for augment in range(
                            len(cm_list[standard][lamp][culture][percent])
                        ):
                            for adv in range(
                                len(cm_list[standard][lamp][culture][percent][augment])
                            ):
                                if standard:
                                    for taugment in range(
                                        len(
                                            cm_list[standard][lamp][culture][percent][
                                                augment
                                            ][adv]
                                        )
                                    ):
                                        for tadversary in range(
                                            len(
                                                cm_list[standard][lamp][culture][
                                                    percent
                                                ][augment][adv][taugment]
                                            )
                                        ):
                                            for tgaug in range(
                                                len(
                                                    cm_list[standard][lamp][culture][
                                                        percent
                                                    ][augment][adv][taugment][
                                                        tadversary
                                                    ]
                                                )
                                            ):
                                                for teps in range(
                                                    len(
                                                        cm_list[standard][lamp][
                                                            culture
                                                        ][percent][augment][adv][
                                                            taugment
                                                        ][
                                                            tadversary
                                                        ][
                                                            tgaug
                                                        ]
                                                    )
                                                ):

                                                        lst = cm_list[standard][lamp][
                                                            culture
                                                        ][percent][augment][adv][
                                                            taugment
                                                        ][
                                                            tadversary
                                                        ][
                                                            tgaug
                                                        ][
                                                            teps
                                                        ]
                                                        sp = np.shape(lst)
                                                        if sp[0] >= len(cultures):
                                                            st = self.buildPath(
                                                                "./",
                                                                standard,
                                                                "DL",
                                                                lamp,
                                                                culture,
                                                                percents[percent],
                                                                augment,
                                                                adv,
                                                                lambda_index,
                                                                taugment,
                                                                tadversary,
                                                                test_g_augs[tgaug],
                                                                test_eps[teps],
                                                                0,
                                                                0,
                                                            )
                                                            tempst = st.split("/")
                                                            tempst2 = ""
                                                            for i in range(
                                                                len(tempst) - 2
                                                            ):
                                                                tempst2 += (
                                                                    tempst[i] + "/"
                                                                )
                                                            st = tempst2
                                                            dir = os.path.dirname(st)
                                                            mkdir(dir)
                                                            rc = ResultsClass(
                                                                np.asarray(lst)
                                                            )
                                                            print(st)
                                                            data = rc.to_df()
                                                            #print(data)
                                                            data.to_csv(st + "res.csv")
                                else:
                                    for lambda_index in range(
                                        len(
                                            cm_list[standard][lamp][culture][percent][
                                                augment
                                            ][adv]
                                        )
                                    ):
                                        for taugment in range(
                                            len(
                                                cm_list[standard][lamp][culture][
                                                    percent
                                                ][augment][adv][lambda_index]
                                            )
                                        ):
                                            tadversarydfs = []
                                            for tadversary in range(
                                                len(
                                                    cm_list[standard][lamp][culture][
                                                        percent
                                                    ][augment][adv][lambda_index][
                                                        taugment
                                                    ]
                                                )
                                            ):
                                                tgaugdfs = []
                                                for tgaug in range(
                                                    len(
                                                        cm_list[standard][lamp][
                                                            culture
                                                        ][percent][augment][adv][
                                                            lambda_index
                                                        ][
                                                            taugment
                                                        ][
                                                            tadversary
                                                        ]
                                                    )
                                                ):
                                                    tepdfs = []
                                                    for teps in range(
                                                        len(
                                                            cm_list[standard][lamp][
                                                                culture
                                                            ][percent][augment][adv][
                                                                lambda_index
                                                            ][
                                                                taugment
                                                            ][
                                                                tadversary
                                                            ][
                                                                tgaug
                                                            ]
                                                        )
                                                    ):

                                                            lst = cm_list[standard][
                                                                lamp
                                                            ][culture][percent][
                                                                augment
                                                            ][
                                                                adv
                                                            ][
                                                                lambda_index
                                                            ][
                                                                taugment
                                                            ][
                                                                tadversary
                                                            ][
                                                                tgaug
                                                            ][
                                                                teps
                                                            ]
                                                            sp = np.shape(lst)
                                                            if sp[0] >= len(cultures):
                                                                st = self.buildPath(
                                                                    "./",
                                                                    standard,
                                                                    "DL",
                                                                    lamp,
                                                                    culture,
                                                                    percents[percent],
                                                                    augment,
                                                                    adv,
                                                                    lambda_indeces[
                                                                        lambda_index
                                                                    ],
                                                                    taugment,
                                                                    tadversary,
                                                                    test_g_augs[tgaug],
                                                                    test_eps[teps],
                                                                    0,
                                                                    0,
                                                                )
                                                                tempst = st.split("/")
                                                                tempst2 = ""
                                                                for i in range(
                                                                    len(tempst) - 2
                                                                ):
                                                                    tempst2 += (
                                                                        tempst[i] + "/"
                                                                    )
                                                                st = tempst2
                                                                dir = os.path.dirname(
                                                                    st
                                                                )
                                                                mkdir(dir)
                                                                rc = ResultsClass(
                                                                    np.asarray(lst)
                                                                )
                                                                data = rc.to_df()
                                                                data.to_csv(
                                                                    st + "res.csv"
                                                                )
                                                                tepdfs.append(data)
                                                    tgaugdfs.append(pd.DataFrame({'tgaugidx':test_eps, 'tepdfs':tepdfs}))
                                                tadversarydfs.append(pd.DataFrame({'tadversaryidx':test_g_augs, 'tgaugdfs':tgaugdfs}))
                                                print(tgaugdfs)
                                            

def mkdir(dir):
    try:
        if not os.path.exists(dir):
            print(f"Making directory: {str(dir)}")
            os.makedirs(dir)
    except Exception as e:
        print(f"{dir} Not created")


def main():
    rac = ResAcquisitionClass()
    cm_list = rac.get_cm_structure("../../Mitigated/")
    rac.save_results(cm_list)


if __name__ == "__main__":
    main()
