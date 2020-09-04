#!/usr/bin/python3
# takes the csv probabilities (given as csv files) of two models
# and combines them to make the final prediction
import os, argparse, sys
import pandas as pd
import re

# import pdb

def choose_index(i):
    # print(rowlist)
    res = rowlist[i].filter(regex='prob').squeeze()

    # choose the maximum valued class
    max_str=res.idxmax()

    return max_str[-1]

def combine_max(rowlist):

    res = rowlist[0]
    for r in rowlist[1:]:
        res = res.filter(regex='prob').add(r.filter(regex='prob'), fill_value=0)

    # choose the maximum valued class
    max_str=(res.max()).idxmax()
    return max_str[-1]

def combine_average(rowlist):
    # print(rowlist)
    res = rowlist[0]
    for r in rowlist[1:]:
        res = res.filter(regex='prob').add(r.filter(regex='prob'), fill_value=0)

    # choose the maximum valued class
    max_str=(res.sum()/2).idxmax()
    return max_str[-1]


def combine_weighted_average(rowlist):

    # weight with the max prediction value
    for i in range(len(rowlist)):
        rowlist[i] = rowlist[i].filter(regex='prob')
        rowlist[i] = rowlist[i].max() * rowlist[i]

    res = rowlist[0]
    for r in rowlist[1:]:
        res = res.add(r, fill_value=0)

    # choose the maximum valued class
    max_str=(res.sum()/2).idxmax()
    return max_str[-1]

if __name__ == '__main__':

    parser=argparse.ArgumentParser()
    parser.add_argument('-f', '--files', nargs='+', help='List of space separated file paths.', required=True)
    parser.add_argument('-s', '--strategy', help='Strategy name: average, max, weighted, first, second.', default='average')
    parser.add_argument('-o', '--output', help='Name of the output file, w/o extension.', default='combined')
    parser.add_argument('-c', '--classname', help='Name of the class.', default='Filling type')

    parser.add_argument('-v', '--validation', help='Calculates the accuracy.', action='store_true')

    args = parser.parse_args()

    print('strategy: '+str(args.strategy))

    df_truth = pd.read_csv('ground_truth.csv')

    df_list = []
    # read csv files as pandas df
    for f in args.files:
        print('reading '+f)
        df = pd.read_csv(f, index_col=False)
        # clear column names
        df = df[df.columns.drop(list(df.filter(regex='Unnamed')))]
        df = df.rename(columns=lambda x: re.sub(r'^.+prob[_ ]?(\d)',args.classname+r' prob\1', x))
        # put into list
        df_list.append(df)

    # pdb.set_trace()

    # apply the specified strategy
    df_combined = pd.DataFrame(columns=['Object', 'Sequence', args.classname])
    true_pred = 0.0

    for index, row in df_list[0].iterrows():
        rowlist = []
        for df in df_list[:]:
            r = df.loc[(df['Object'] == row['Object']) & (df['Sequence'] == row['Sequence'])]
            rowlist.append(r)

        # calculate the combined label and append
        if args.strategy == 'average':
            label = combine_average(rowlist)
        elif args.strategy == 'max':
            label = combine_max(rowlist)
        elif args.strategy == 'weighted':
            label = combine_weighted_average(rowlist)
        elif args.strategy == 'first':
            label = choose_index(0)
        elif args.strategy == 'second':
            label = choose_index(1)

        new_row = {'Object':row['Object'], 'Sequence':row['Sequence'], args.classname:int(label)}
        df_combined = df_combined.append(new_row, ignore_index=True)

        if args.validation:
            true_val = df_truth.loc[(df['Object'] == row['Object']) & (df['Sequence'] == row['Sequence'])][args.classname].iloc[0]
            if int(label) == true_val:
                true_pred += 1.0

    df_combined.to_csv(args.output+'.csv')

    if args.validation:
        print(true_pred/len(df_truth['Object']))
