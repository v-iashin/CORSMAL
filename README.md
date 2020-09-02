# CORSMAL

## Organization
Please follow this structure
`/task/approach/code`

For example:
- `/filling_level`
    - `/vggish`
        - `/your`
        - `/files`
        - `/and`
        - `/folders`
    - `/your_approach`
        - `/your`
        - `/files`
        - `/and`
        - `/folders`
    - ...
    - `README.md` (please document the approach TINY bit 😉: env, how to run, results on valid/train)
- `/capacity`
    - `/LODE`
    - ...
    - `README.md` (please document the approach TINY bit 😉: env, how to run, results on valid/train)
- `/filling_type`
    - `/your_approach`
        - `/your`
        - `/files`
        - `/and`
        - `/folders`
    - ...
    - `README.md` (please document the approach TINY bit 😉: env, how to run, results on valid/train)


## Output
Each approach should return predictions for test set `your_approach.csv`. Please use `Submission_form.csv` for your reference. Leave other field to be empty, i.e. when predicting `filling_type` a line in `your_approach.csv` should be of format: `container_id,sequence,prob0,prob1,prob2,prob3`.

## Submission
Please, use `main.py` which will aggregate the predictions from each task and form the final prediction on the test-set
