import os
import argparse
import pandas as pd


def combine_ftype(on_private):
    # Content_2_index = {
    #     0: "Empty",
    #     1: "Pasta",
    #     2: "Rice",
    #     3: "Water"
    # }
    if on_private:
        vggish_path = './filling_type/vggish/predictions/200903163404/ftype_private_test_agg_vggish.csv'
        rf_path = './filling_type/CORSMAL-pyAudioAnalysis/ftype-randomforest-final_result_private_test.csv'
    else:
        vggish_path = './filling_type/vggish/predictions/200903163404/ftype_public_test_agg_vggish.csv'
        rf_path = './filling_type/CORSMAL-pyAudioAnalysis/ftype-randomforest-final_result_public_test.csv'

    vggish = pd.read_csv(vggish_path)

    ftype_randomforest = pd.read_csv(rf_path)
    ftype_randomforest = ftype_randomforest.sort_values(['Object', 'Sequence']).reset_index(drop=True)

    random_forest_preds = ftype_randomforest[[
        'Filling type prob0', 'Filling type prob1', 'Filling type prob2', 'Filling type prob3'
    ]]
    vggish_preds = vggish[['ftype_prob_0', 'ftype_prob_1', 'ftype_prob_2', 'ftype_prob_3']]

    ftype_combined = (random_forest_preds.values + vggish_preds.values) / 2

    # return pd.Series([Content_2_index[cls] for cls in ftype_combined.argmax(axis=1)])
    return pd.Series([cls for cls in ftype_combined.argmax(axis=1)])


def combine_flvl(on_private):
    # filling_2_value = {0: 0, 1: 50, 2: 90}
    cols_with_probs_1 = ['flvl_prob_0', 'flvl_prob_1', 'flvl_prob_2']

    if on_private:
        vggish_path = './filling_level/vggish/predictions/200903162117/flvl_private_test_agg_vggish.csv'
        r21d_path = './filling_level/r21d_rgb/predictions/200903214601/flvl_private_test_agg_r21d_rgb.csv'
        rf_path = './filling_level/CORSMAL-pyAudioAnalysis/flevel-randomforest-final_result_private_test.csv'
    else:
        vggish_path = './filling_level/vggish/predictions/200903162117/flvl_public_test_agg_vggish.csv'
        r21d_path = './filling_level/r21d_rgb/predictions/200903214601/flvl_public_test_agg_r21d_rgb.csv'
        rf_path = './filling_level/CORSMAL-pyAudioAnalysis/flevel-randomforest-final_result_public_test.csv'

    flvl_vggish = pd.read_csv(vggish_path)
    flvl_r21d = pd.read_csv(r21d_path)

    flvl_vggish = flvl_vggish[cols_with_probs_1]
    flvl_r21d = flvl_r21d[cols_with_probs_1]

    # flvl_combined = (flvl_vggish.values + flvl_r21d.values) / 2
    # flvl_combined = flvl_vggish.values

    # we also observed that adding pyAudioAnalysis' random forest predictions, improves valid performance
    cols_with_probs_2 = ['Filling level [%] prob0', 'Filling level [%] prob1', 'Filling level [%] prob2']
    flvl_rf = pd.read_csv(rf_path)
    flvl_rf = flvl_rf.sort_values(['Object', 'Sequence']).reset_index(drop=True)
    flvl_rf = flvl_rf[cols_with_probs_2]
    flvl_combined = (flvl_vggish.values + flvl_r21d.values + flvl_rf.values) / 3

    # return pd.Series([int(filling_2_value[cls]) for cls in flvl_combined.argmax(axis=1)])
    return pd.Series([int(cls) for cls in flvl_combined.argmax(axis=1)])


def capacity(on_private):
    if on_private:
        cap_path = './capacity/results/estimation_combination_private_test.csv'
        # cap_path = './capacity/results/estimation_combination_with_0_private_test.csv'
        # cap_path = './capacity/results/estimation_combination_with_1_private_test.csv'
    else:
        cap_path = './capacity/results/estimation_combination_public_test.csv'
        # cap_path = './capacity/results/estimation_combination_with_0_public_test.csv'
        # cap_path = './capacity/results/estimation_combination_with_1_public_test.csv'

    a = pd.read_csv(cap_path)
    return a['capacity[mL]']


# def estimate_fmass(submission):
#     Content_2_density = {
#         "Empty": 0.0,  # "Empty"
#         0: 0.0,  # "Empty"
#         "Pasta": 0.41,  # "Pasta"
#         1: 0.41,  # "Pasta"
#         "Rice": 0.85,  # "Rice"
#         2: 0.85,  # "Rice"
#         "Water": 1.00,  # "Water"
#         3: 1.00,  # "Water"
#     }
#     fmass_col = []
#     for cont, seq, capacity, c_mass, ftype, flvl, fmass in submission.values:
#         fmass = Content_2_density[ftype] * flvl / 100 * capacity
#         fmass_col.append(fmass)

#     return pd.Series(fmass_col)


def make_submission_form(data_path, on_private):
    columns = ['Container ID', 'Sequence', 'Filling type', 'Filling level', 'Container Capacity']
    submission = pd.DataFrame(columns=columns)

    if on_private:
        container_ids = ['13', '14', '15']
    else:
        container_ids = ['10', '11', '12']

    # creating columns for container id and sequence using filenames from audio folder – 0053_audio.wav -> 53
    object_list = []
    sequence_list = []
    for container_id in container_ids:
        path = os.path.join(data_path, container_id, 'audio')
        filenames = sorted(os.listdir(path))
        seq_ids = [int(fname.replace('_audio.wav', '')) for fname in filenames]
        sequence_list.extend(seq_ids)
        object_list.extend([container_id] * len(seq_ids))

    submission['Container ID'] = object_list
    submission['Sequence'] = sequence_list

    return submission


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict_on_private', dest='predict_on_private', action='store_true', default=False)
    parser.add_argument('--data_path', default='./dataset/')
    args = parser.parse_args()

    # Gather prediction for the public test set
    submission_public = make_submission_form(args.data_path, on_private=False)
    submission_public['Filling type'] = combine_ftype(on_private=False)
    submission_public['Filling level'] = combine_flvl(on_private=False)
    submission_public['Container Capacity'] = capacity(on_private=False)

    # submission_public['Filling mass'] = estimate_fmass(submission_public)
    submission_public.to_csv('./submission_public_test.csv', index=False)
    print('Formed predictions in ./submission_public_test.csv')

    # If specified, gather prediction for the public test set
    if args.predict_on_private:
        submission_private = make_submission_form(args.data_path, on_private=True)
        submission_private['Filling type'] = combine_ftype(on_private=True)
        submission_private['Filling level'] = combine_flvl(on_private=True)
        submission_private['Container Capacity'] = capacity(on_private=True)

        # submission_private['Filling mass'] = estimate_fmass(submission_private)
        submission_private.to_csv('./submission_private_test.csv', index=False)
        print('Formed predictions in ./submission_private_test.csv')
