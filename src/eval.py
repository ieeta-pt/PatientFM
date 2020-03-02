# -*- coding: utf-8 -*-
"""
    File name: eval.py
    Project: Family History Extraction
    Description: evaluation script of the OHNLP/BioCreative 2018 Task 1: Family history extraction
    Author: Sijia Liu (liu dot sijia at mayo dot edu)
"""

from collections import defaultdict
import argparse
import sys


def parse_s1_output(output_path):
    """
    Parse GS or output file for subtask 1 (entity identification)
    :param output_path: 
    :return: 
    """
    
    out_fm = set()

    out_ob = defaultdict(set)

    with open(output_path) as f:
        for l in f:
            splits = l.strip().split('\t')
            if len(splits) == 4:
                doc, _, fm, side = splits
                out_fm |= set([(doc, fm, side)])
            elif len(splits) == 3:
                doc, _, ob = splits
                out_ob[doc] |= set([ob.lower()])
            else:
                print("Number of fields should be 3 or 4. Got {} instead: ".format(len(splits)))
                print("\"{}\"\t->\t{}".format(l.strip(), splits))
                sys.exit(0)

    return out_fm, out_ob


def parse_s2_output(output_path):
    """
    Parse GS or output file for subtask 2 (family history extraction)

    :param output_path:
    :return:
    """
    out_fm = set()

    out_ob = defaultdict(set)

    with open(output_path) as f:
        for l in f:
            splits = l.strip().split('\t')
            print(splits)
            ''' 
            Example:       
            doc_1	Son	NA	LivingStatus	4
            doc_1	Cousin	Paternal	Observation	choreic dysphonia   No
            '''

            if splits[3] == 'LivingStatus':
                doc, fm, side, _ , status = splits
                out_fm |= set([(doc, fm, side, status)])
            elif splits[3] == 'Observation':
                #M31: added negation for evaluation
                doc, fm, side, _ , ob, negation = splits
                out_ob[(doc, fm, side)] |= set([ob.lower(),negation])
                # Original
                # doc, fm, side, _ , ob = splits
                # out_ob[(doc, fm, side)] |= set([ob.lower()])

    return out_fm, out_ob


def get_pr_f1(tp, fp, fn):
    print("TP\t\tFP\t\tFN\t\t")
    print("{}\t{} \t{}".format(tp, fp, fn))

    precision = tp / float(tp + fp)
    recall = tp / float(tp + fn)

    f1 = 2 * precision * recall / (precision + recall)
    print("Prevision\tRecall\tF1")
    print("{:.4f}\t{:.4f} \t{:.4f}".format(precision, recall, f1))
    print()


def calculate_s1(gs_tsv, pred_tsv, verbose):
    """
    Calculate system performance of subtask 1
    :param gs_tsv:
    :param pred_tsv:
    :param verbose:
    :return: None. The performance are printed in console output
    """
    gs_fm, gs_ob = parse_s1_output(gs_tsv)
    pred_fm, pred_ob = parse_s1_output(pred_tsv)

    tp_fm = 0
    fp_fm = 0
    ignoringSide = False

    falsePostive = {}
    truePostive = {}
    if ignoringSide:
        new_gs_fm = list(set([(x[0], x[1]) for x in gs_fm]))
        new_pred_fm = list(set([(x[0], x[1]) for x in pred_fm]))
        for elm in new_pred_fm:
            fm = (elm[1])
            if elm in new_gs_fm:
                tp_fm += 1
                if fm in truePostive:
                    truePostive[fm] += 1
                else:
                    truePostive[fm] = 1
            else:
                if fm in falsePostive:
                    falsePostive[fm] += 1
                else:
                    falsePostive[fm] = 1
                fp_fm += 1
        fn_fm = len(new_gs_fm) - tp_fm
    else:
        for elm in pred_fm:
            fm = (elm[1], elm[2])
            if elm in gs_fm:
                tp_fm += 1
                if fm in truePostive:
                    truePostive[fm] += 1
                else:
                    truePostive[fm] = 1
            else:
                if fm in falsePostive:
                    falsePostive[fm] += 1
                else:
                    falsePostive[fm] = 1
                if verbose: print(elm)
                fp_fm += 1
        if verbose:
            print("True positives")
            print(sorted(truePostive.items(), key=lambda kv: kv[1]))
            print("False positives")
            print(sorted(falsePostive.items(), key=lambda kv: kv[1]))

        fn_fm = len(gs_fm) - tp_fm

    if verbose:
        print("FM: ")
        get_pr_f1(tp_fm, fp_fm, fn_fm)


    tp_ob = 0
    total_pred_ob = 0

    for doc in pred_ob:
        pred_ob_set = pred_ob[doc]

        gs_ob_set = gs_ob[doc]

        for gob in gs_ob_set:
            for pob in pred_ob_set:
                if set(pob.split()) & set(gob.split()):
                    tp_ob += 1
                    break

        total_pred_ob += len(pred_ob_set)

    total_gs_ob = 0
    for doc in gs_ob:
        total_gs_ob += len(gs_ob[doc])

    fp_ob = total_pred_ob - tp_ob
    fn_ob = total_gs_ob - tp_ob

    if verbose:
        print("OB: ")
        get_pr_f1(tp_ob, fp_ob, fn_ob)

    tp = tp_ob + tp_fm
    fp = fp_ob + fp_fm
    fn = fn_ob + fn_fm

    print("Overall:")
    get_pr_f1(tp, fp, fn)


def calculate_s2(gs_tsv, pred_tsv, verbose):
    """
    Calculate system performance of subtask 2
    :param gs_tsv:
    :param pred_tsv:
    :param verbose:
    :return: None. The performance are printed in console output
    """

    gs_fm, gs_ob = parse_s2_output(gs_tsv)
    pred_fm, pred_ob = parse_s2_output(pred_tsv)

    tp_fm = 0
    fp_fm = 0

    for elm in pred_fm:
        print(elm)
        if elm in gs_fm:
            tp_fm += 1
        else:
            fp_fm += 1

    fn_fm = len(gs_fm) - tp_fm

    if verbose:
        print("FM: ")
        get_pr_f1(tp_fm, fp_fm, fn_fm)

    tp_ob = 0
    total_pred_ob = 0

    for key in pred_ob:
        pred_ob_set = pred_ob[key]

        gs_ob_set = gs_ob[key]

        for gob in gs_ob_set:
            for pob in pred_ob_set:
                if set(pob.split()) & set(gob.split()):
                    tp_ob += 1
                    break

        total_pred_ob += len(pred_ob_set)

    total_gs_ob = 0

    for key in gs_ob:
        total_gs_ob += len(gs_ob[key])

    fp_ob = total_pred_ob - tp_ob
    fn_ob = total_gs_ob - tp_ob

    if verbose:
        print("OB: ")
        get_pr_f1(tp_ob, fp_ob, fn_ob)

    tp = tp_ob + tp_fm
    fp = fp_ob + fp_fm
    fn = fn_ob + fn_fm

    print("Overall:")
    get_pr_f1(tp, fp, fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("subtask", type=int, help="selection of subtasks: \"1\" or \"2\"")
    parser.add_argument("-v", "--verbose",
                        help="includes the performance of " +
                             "family members and observations in the output",
                        action="store_true")

    parser.add_argument("gs", help="Gold Standard file of the selected subtask")
    parser.add_argument("pred", help="Prediction file of the selected subtask")

    args = parser.parse_args()

    if args.subtask == 1:
        calculate_s1(args.gs, args.pred, args.verbose)
    elif args.subtask == 2:
        calculate_s2(args.gs, args.pred, args.verbose)
    else:
        print("Please select the subtask. ")
        parser.print_help()
