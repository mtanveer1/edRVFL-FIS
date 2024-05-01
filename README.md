# edRVFL-FIS: Ensemble Deep Random Vector Functional Link Neural Network Based on Fuzzy Inference System

Please cite the following paper if you are using this code.

Reference: M. Sajid, M. Tanveer, and P. N. Suganthan (2024). "Ensemble Deep Random Vector Functional Link Neural Network Based on Fuzzy Inference System”
- Revision submitted in IEEE Transactions on Fuzzy Systems.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

We have put a demo of the "edRVFL-FIS" model with the "cardiotocography_3clases" dataset
We have three variants of edRVFL-FIS as follows:
	1. edRVFL-FIS-K if you choose clus=1 in edRVFL_FIS_main.m file.
	2. edRVFL-FIS-C if you choose clus=2 in edRVFL_FIS_main.m file.
	3. edRVFL-FIS-R if you choose clus=3 in edRVFL_FIS_main.m file.
 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

The experimental procedures are executed on a computing system possessing MATLAB R2023a software, Intel(R) Xeon(R) Platinum 8260 CPU @ 2.30GHz, 2301 Mhz, 24 Core(s),
48 Logical Processor(s) with 256 GB RAM on a Windows-10 operating platform.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Demo Hyperparameter setting
---------------------------
option.activation = 1; %Sigmoid Activation function
option.scale = 1;
option.renormal=1;
option.normal_type=0; %(0 for batch normaliation and 1 for layer normalization)
option.L=7; %Number of hidden layers
option.N=810; %Number of hidden nodes
option.NumFuzzyRule=15; %Number of fuzzy layer nodes/centers/rules
option.C=1; %Regularization parameter

Note: For deatiled parameters setting, please refer "Ensemble Deep Random Vector Functional Link Neural Network Based on Fuzzy Inference System" paper.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Description of files:
---------------------
edRVFL_FIS_main.m: This is the main file to run selected models on datasets. In the path variable specificy the path to the folder containing the codes and datasets on which you wish to run the algorithm. 

MRVFL.m: Intermediate function to call training and testing functions.

edRVFL_FIS_train.m: Training function.

edRVFL_FIS_defuzzy_train.m: Defuzzification function used at the time of training.

edRVFL_FIS_test.m: Testing function.

edRVFL_FIS_defuzzy_test.m: Defuzzification function used at the time of testing.

l2_weights.m: Function to calculate the output layer weights.

sigmoid.m: Sigmoid activation function.

majorityVoting.m: Majority voting function.

Evaluate.m: Function to evaluate the accuracy.

cardiotocography_3clases.mat: cardiotocography_3clases dataset used to execute the code.
________________________________________________________________
Remarks:

	1. The codes have been cleaned for better readability and documented, then re-run and checked the codes only in a few datasets, so if you find any bugs/issues, please write to M. Sajid (phd2101241003@iiti.ac.in).
 
	2. For the detailed experimental setup, please follow our paper.  
________________________________________________________________
Some parts of the codes have been taken from:

	1. Shi, Qiushi, Rakesh Katuwal, Ponnuthurai N. Suganthan, and M. Tanveer. "Random vector functional link neural network based ensemble deep learning." Pattern Recognition 117 (2021): 107978.
 
	2. Feng, Shuang, and CL Philip Chen. "Fuzzy broad learning system: A novel neuro-fuzzy model for regression and classification." IEEE transactions on cybernetics 50, no. 2 (2018): 414-424.

01-May-2024


