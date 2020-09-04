import pandas as pd
import argparse


if __name__ == "__main__":
    # Take the file paths from arguments and fill the form
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--ftype', nargs='+', help='Filling type csv file.', required=True)
    parser.add_argument('-l', '--flevel', nargs='+', help='Filling level csv file.', required=True)
    parser.add_argument('-c', '--capacity', nargs='+', help='Filling capacity csv file.', required=True)
    parser.add_argument('-o', '--output', help='Name of the output file', default='./submission.csv')

    args = parser.parse_args()

    # get the empty submission form
    df_sub = pd.read_csv('Submission_form.csv')

    # read files
    df_ftype = pd.read_csv(args.ftype)
    df_flevel = pd.read_csv(args.flevel)
    df_capacity = pd.read_csv(args.capacity)

    # add columns
    df_sub['Filling type'] = df_ftype['Filling type']
    df_sub['Filling level [%]'] = df_flevel['Filling level [%]']
    df_sub['Container capacity [mL]'] = df_capacity['Container capacity [mL]']

    df_sub.to_csv(args.output, index=False)
