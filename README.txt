To run an experiment: python experiment.py louo (or random_split)

- experiment.py : configures the experiment,ie. cross val scheme and outputs results
- trainer.py : trains and evaluates the model
- models.py : model definition
- utils.py : hyperparameters and helper functions
- dataloader.py : dataset class and dataloaders definition

The code basically read a pickle file that defines the data for each fold of a certain cross-validation scheme.

LOUO: We have 7 experts (E) and 6 novices (N).
I splited them like this:
fold 1: E1,N1
fold 2: E2,N2,E7 # This one has one more expert because we had more experts than novices and there were less attempts here than the other folds.
fold 3: E3,N3
fold 4: E4,N4
fold 5: E5,N5
fold 6: E6,N6 

Also, E2 doesn't have video data. So in case we use video data in the future, we might leave out the force data from E2, so fold 2 will become: E7,N2 in that case.

Preprocessing folder:
-split_sheets.py : reads original data and seperates the attempts, as well as saves them in seperate csv files.
-preprocessing.py : replaces negative force values with zeros
-create_pkl_files.py : creates pickle files for cross val schemes