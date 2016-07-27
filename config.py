from sklearn import ensemble, svm, tree, linear_model, neighbors, naive_bayes
from time import strftime

# t/2549->458, but somehow task 478 was downloaded?
test_task_ids = [9903, 3567]#[3524, 3903, 10101, 18]#[9968,14965,2075] 
task_ids = [9968,14965,2075,9970,28,9956,3903,3504,16,23,11,3972,10090,3992,9902,3891,3512,3492,219,3485,14966,3889,43,3536,3896,9957,9964,9971,9914,9952,3954,12,31,36,3973,34539,
			3529,3524,3567,9983,9976,20,3481,3979,6,3950,9977,37,32,49,18,3493,3518,3549,3917,9903,9960,9908,9946,3962,3918,3913,9904,4000,9954,14,34536,9909,3521,3494,14970,3019,
			45,21,9985,9980,9978,10101,10093,3999,7307,3968,3995,9950,9955,9967,3510,3534,14964,14971,2074,58,53,3022,34537,22,3527,3560,9979,9986,3,3902,3964,9905,9981,14969,3971,3899]

base_learners = [ensemble.RandomForestClassifier,
			svm.SVC,
			tree.DecisionTreeClassifier,
			naive_bayes.GaussianNB, # Other two NB are for discrete/binary data.
			ensemble.GradientBoostingClassifier,
			lambda : linear_model.LogisticRegression(multi_class='multinomial', solver='lbfgs'),
			neighbors.KNeighborsClassifier]

output_directory = "output"
document_name = "{}/results-{}.txt".format(output_directory, strftime("%Y-%m-%d_%H_%M_%S"))
logfile_name = "{}/log-{}.txt".format(output_directory, strftime("%Y-%m-%d_%H_%M_%S"))

excluded_tasks = { 2075: 'Classes with too few examples'}