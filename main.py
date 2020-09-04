import pandas as pd
import argparse

def combine_ftype():
    Content_2_index = {
        0: "Empty",
        1: "Pasta",
        2: "Rice",
        3: "Water"
    }
    vggish = pd.read_csv('./filling_type/vggish/predictions/200903163404/ftype_test_agg_vggish.csv')

    ftype_randomforest = pd.read_csv(
        './filling_type/CORSMAL-audio-only-filling-type-analysis/results/ftype-randomforest-final.csv',
    )
    ftype_randomforest.sort_values(['Object', 'Sequence']).reset_index(drop=True)

    random_forest_preds = ftype_randomforest[[
        'Filling type prob0', 'Filling type prob1', 'Filling type prob2', 'Filling type prob3'
    ]]
    vggish_preds = vggish[['ftype_prob_0', 'ftype_prob_1', 'ftype_prob_2', 'ftype_prob_3']]

    ftype_combined = (random_forest_preds.values + vggish_preds.values) / 2

    return pd.Series([Content_2_index[cls] for cls in ftype_combined.argmax(axis=1)])

def combine_flvl():
    filling_2_value = {0: 0, 1: 50, 2: 90}
    flvl_vggish = pd.read_csv('filling_level/vggish/predictions/200903162117/flvl_test_agg_vggish.csv')
    print(flvl_vggish)
    flvl_vggish = flvl_vggish[['flvl_prob_0', 'flvl_prob_1', 'flvl_prob_2']]
    flvl_combined = flvl_vggish.values
    print(flvl_vggish)
    return pd.Series([int(filling_2_value[cls]) for cls in flvl_combined.argmax(axis=1)])

def capacity():
    a = pd.read_csv('./capacity/results/estimation_combined.csv')
    return a['Container capacity [mL]']


def estimate_fmass(submission):
    Content_2_density = {
        "Empty": 0.0,  # "Empty"
        "Pasta": 0.41,  # "Pasta"
        "Rice": 0.85,  # "Rice"
        "Water": 1.00  # "Water"
    }
    fmass_col = []
    for cont, seq, capacity, c_mass, ftype, flvl, fmass in submission.values:
        fmass = Content_2_density[ftype] * flvl / 100 * capacity
        fmass_col.append(fmass)

    return pd.Series(fmass_col)


if __name__ == "__main__":
    submission = pd.read_csv('./Submission_form.csv')
    submission['Sequence'] = submission['Sequence'].apply(lambda x: f'{x:04d}')

    submission['Filling type'] = combine_ftype()
    submission['Filling level [%]'] = combine_flvl()
    submission['Container capacity [mL]'] = capacity()

    submission.to_csv('./submission_before_final_form.csv', index=False)
    submission['Filling mass [g]'] = estimate_fmass(submission)
    submission.to_csv('./submission.csv', index=False)
