## How to run
To Run Prediction and create out files (This will produce the dev1.out, dev2.out, test1.out, test2.out, dev3.out and pred files)
>python run.py --predict
To Run Training and create model files (This will produce blstm1.pt and blstm2.pt files)
>python run.py --train


# Assumptions
- cuda torch is required for loading and training the models.
- I am reading the unzipped globe.6B.100d.txt file for Task 2 which should be present in the same directory.
- The prediction command produces these files:
1. dev1.out - prediction file on dev for Task 1
2. dev2.out - prediction file on dev for Task 2
3. test1.out - prediction file on test for Task 1
4. test2.out - prediction file on test for Task 2
5. pred - prediction file on test for bonus task
6. dev3.out - prediction file on dev for bonus task

Some extra gold1.out, gold2.out and gold3.out files are also produced which produce the perl standard files for dev data which can be ignored if not required.

## Dependencies
python version == 3.10.9
numpy==1.23.5
scikit-learn==1.2.0
pytorch==1.13.1
torch==1.13.1
