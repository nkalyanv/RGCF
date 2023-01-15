In the src folder:
The train_classifier.py script can be used to train a neural network for the filter of RGCF. (Command: python train_classifier.py)
The test_classifier.py script can be used to test the performance of the filter on unseen data using the trained NN. (Command: python test_classifier.py)
The results are stored in the Experiment_Results folder. 



In the prior work folder:
Using the Experiments.py (Command: python "genGraphs.py") script the prior work defenses can be tested against the four attacks:
1) All Ones
2) Inverse Attack 
3) Gradient Shift Attack 
4) Random Attack.
The number of Byzantine and normal workers can be set as well. This script generates numpy arrays of the loss and validation accuracy of the run.

To visualise these results, you can use the genGraphs.py script using the command "python genGraphs.py". This script generates the graphs shown in the paper from the corresponding numpy arrays. This script also obtains the results of the RGCF filter from the src path to construct the graphs.

