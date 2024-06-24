
import project1 as p1
import utils
import numpy as np

#-------------------------------------------------------------------------------
# Data loading. There is no need to edit code in this section.
#-------------------------------------------------------------------------------

train_data = utils.load_data('reviews_train.tsv')
val_data = utils.load_data('reviews_val.tsv')
test_data = utils.load_data('reviews_test.tsv')

train_texts, train_labels = zip(*((sample['text'], sample['sentiment']) for sample in train_data))
val_texts, val_labels = zip(*((sample['text'], sample['sentiment']) for sample in val_data))
test_texts, test_labels = zip(*((sample['text'], sample['sentiment']) for sample in test_data))

#dictionary = p1.bag_of_words(train_texts)
dictionary = p1.bag_of_words(train_texts, True)   # Remove stopwords

# binarize = True
#train_bow_features = p1.extract_bow_feature_vectors(train_texts, dictionary)
#val_bow_features = p1.extract_bow_feature_vectors(val_texts, dictionary)
#test_bow_features = p1.extract_bow_feature_vectors(test_texts, dictionary)

# If using the word count. binarize=False
train_bow_features = p1.extract_bow_feature_vectors(train_texts, dictionary, False)
val_bow_features = p1.extract_bow_feature_vectors(val_texts, dictionary, False)
test_bow_features = p1.extract_bow_feature_vectors(test_texts, dictionary, False)

#-------------------------------------------------------------------------------
# Problem 5
#-------------------------------------------------------------------------------

#toy_features, toy_labels = toy_data = utils.load_toy_data('toy_data.tsv')
#
#T = 2500
#L = 0.2
#
#thetas_perceptron = p1.perceptron(toy_features, toy_labels, T)
#thetas_avg_perceptron = p1.average_perceptron(toy_features, toy_labels, T)
#thetas_pegasos = p1.pegasos(toy_features, toy_labels, T, L)
#
#def plot_toy_results(algo_name, thetas):
#    print('theta for', algo_name, 'is', ', '.join(map(str,list(thetas[0]))))
#    print('theta_0 for', algo_name, 'is', str(thetas[1]))
#    utils.plot_toy_data(algo_name, toy_features, toy_labels, thetas)

#plot_toy_results('Perceptron', thetas_perceptron)
#plot_toy_results('Average Perceptron', thetas_avg_perceptron)
#plot_toy_results('Pegasos', thetas_pegasos)

#-------------------------------------------------------------------------------
# Problem 7
#-------------------------------------------------------------------------------
#
# T = 10
# L = 0.01
#
# pct_train_accuracy, pct_val_accuracy = \
#    p1.classifier_accuracy(p1.perceptron, train_bow_features,val_bow_features,train_labels,val_labels,T=T)
# print("{:35} {:.4f}".format("Training accuracy for perceptron:", pct_train_accuracy))
# print("{:35} {:.4f}".format("Validation accuracy for perceptron:", pct_val_accuracy))
#
# avg_pct_train_accuracy, avg_pct_val_accuracy = \
#    p1.classifier_accuracy(p1.average_perceptron, train_bow_features,val_bow_features,train_labels,val_labels,T=T)
# print("{:43} {:.4f}".format("Training accuracy for average perceptron:", avg_pct_train_accuracy))
# print("{:43} {:.4f}".format("Validation accuracy for average perceptron:", avg_pct_val_accuracy))
#
# avg_peg_train_accuracy, avg_peg_val_accuracy = \
#     p1.classifier_accuracy(p1.pegasos, train_bow_features,val_bow_features,train_labels,val_labels,T=T,L=L)
# print("{:50} {:.4f}".format("Training accuracy for Pegasos:", avg_peg_train_accuracy))
# print("{:50} {:.4f}".format("Validation accuracy for Pegasos:", avg_peg_val_accuracy))

#-------------------------------------------------------------------------------
# Problem 8
#-------------------------------------------------------------------------------

# data = (train_bow_features, train_labels, val_bow_features, val_labels)
#
# # values of T and lambda to try
# Ts = [1, 5, 10, 15, 25, 50]
# Ls = [0.001, 0.01, 0.1, 1, 10]
#
# pct_tune_results = utils.tune_perceptron(Ts, *data)
# print('perceptron valid:', list(zip(Ts, pct_tune_results[1])))
# print('best = {:.4f}, T={:.4f}'.format(np.max(pct_tune_results[1]), Ts[np.argmax(pct_tune_results[1])]))
#
# avg_pct_tune_results = utils.tune_avg_perceptron(Ts, *data)
# print('avg perceptron valid:', list(zip(Ts, avg_pct_tune_results[1])))
# print('best = {:.4f}, T={:.4f}'.format(np.max(avg_pct_tune_results[1]), Ts[np.argmax(avg_pct_tune_results[1])]))
#
# # fix values for L and T while tuning Pegasos T and L, respective
# fix_L = 0.01
# peg_tune_results_T = utils.tune_pegasos_T(fix_L, Ts, *data)
# print('Pegasos valid: tune T', list(zip(Ts, peg_tune_results_T[1])))
# print('best = {:.4f}, T={:.4f}'.format(np.max(peg_tune_results_T[1]), Ts[np.argmax(peg_tune_results_T[1])]))
#
# fix_T = Ts[np.argmax(peg_tune_results_T[1])]
# peg_tune_results_L = utils.tune_pegasos_L(fix_T, Ls, *data)
# print('Pegasos valid: tune L', list(zip(Ls, peg_tune_results_L[1])))
# print('best = {:.4f}, L={:.4f}'.format(np.max(peg_tune_results_L[1]), Ls[np.argmax(peg_tune_results_L[1])]))
#
# utils.plot_tune_results('Perceptron', 'T', Ts, *pct_tune_results)
# utils.plot_tune_results('Avg Perceptron', 'T', Ts, *avg_pct_tune_results)
# utils.plot_tune_results('Pegasos', 'T', Ts, *peg_tune_results_T)
# utils.plot_tune_results('Pegasos', 'L', Ls, *peg_tune_results_L)

#-------------------------------------------------------------------------------
# Use the best method (perceptron, average perceptron or Pegasos) along with
# the optimal hyperparameters according to validation accuracies to test
# against the test dataset. The test data has been provided as
# test_bow_features and test_labels.
#-------------------------------------------------------------------------------

#Your code here
T = 25
L = 0.01

avg_peg_train_accuracy, avg_peg_test_accuracy = \
    p1.classifier_accuracy(p1.pegasos, train_bow_features,test_bow_features,train_labels,test_labels,T=T,L=L)
#print("{:50} {:.4f}".format("Training accuracy for Pegasos:", avg_peg_train_accuracy))
print("{:50} {:.4f}".format("Test accuracy for Pegasos:", avg_peg_test_accuracy))

#-------------------------------------------------------------------------------
# Assign to best_theta, the weights (and not the bias!) learned by your most
# accurate algorithm with the optimal choice of hyperparameters.
#-------------------------------------------------------------------------------

T = 25
L = 0.01

thetas_pegasos = p1.pegasos(train_bow_features, train_labels, T, L)

best_theta = thetas_pegasos[0] # Your code here
wordlist  = [word for (idx, word) in sorted(zip(dictionary.values(), dictionary.keys()))]
sorted_word_features = utils.most_explanatory_word(best_theta, wordlist)
print("Most Explanatory Word Features")
print(sorted_word_features[:10])
print(sorted_word_features[-10:])








## MY rough pad

#print(p1.perceptron([[1, 0], [1, -1], [2, 3]], [1, -1, 1], 1))
#print(p1.perceptron([[1, 1], [2, -1]], [-1, 1], 1))
'''
print(p1.average_perceptron([[ 0.41490348, -0.08250658, -0.03321058, -0.36229215,  0.31420183,  0.35811894,
  0.11128741, -0.47206405, -0.47793648,  0.12747828],
[ 0.18086483,  0.23198512,  0.28824286,  0.42539597,  0.01649095, -0.11470911,
 -0.27615933, -0.12317445, -0.03104139, -0.46502016],
[ 0.2751927,  -0.0400112,  -0.16468255, -0.36054498,  0.49687732,  0.20157144,
  0.45315508, -0.37120103, -0.45574504, -0.10128742],
 [ 0.45819052,  0.27268992,  0.39709244, -0.21002639,  0.23601954,  0.32412751,
 -0.24145376,  0.06488368, -0.04824896,  0.13859551],
[-0.19707492,  0.43457661, -0.09024343,  0.41237591,  0.30605877, -0.18159691,
 -0.4435708,  -0.27509223, -0.02927586,  0.24689214]], [-1, 1, 1, 1, -1], 5))
'''

'''
print(p1.pegasos_single_step_update([ 0.40367132, -0.21349862,  0.03312223,  0.40361444,  0.27599914, -0.49593083,
    0.43230463,  0.42157706, -0.3396615,  -0.05265132], -1, 0.7025525539656643, 0.7095553439183783,
    [-0.01855253,  0.17689572,  0.16005859,  0.32972929, -0.35467273, -0.09131889,
        0.21232983, -0.10609079,  0.08492366,  0.42350706], -1.0398587927587393))
'''


'''
print(p1.pegasos([[ 0.1837462, 0.29989789, -0.35889786, -0.30780561, -0.44230703, -0.03043835, 0.21370063,  0.33344998, -0.40850817, -0.13105809],
 [ 0.08254096,  0.06012654,  0.19821234,  0.40958367,  0.07155838, -0.49830717,
   0.09098162,  0.19062183, -0.27312663,  0.39060785],
 [-0.20112519, -0.00593087,  0.05738862,  0.16811148, -0.10466314, -0.21348009,
   0.45806193, -0.27659307,  0.2901038,  -0.29736505],
 [-0.14703536, -0.45573697, -0.47563745, -0.08546162, -0.08562345,  0.07636098,
  -0.42087389, -0.16322197, -0.02759763,  0.0297091 ],
 [-0.18082261,  0.28644149, -0.47549449, -0.3049562,   0.13967768,  0.34904474,
   0.20627692,  0.28407868,  0.21849356, -0.01642202]], [-1, -1, -1,  1, -1], 10, 0.1456692551041303))
'''
'''
print(p1.classify([[ 0.1837462, 0.29989789, -0.35889786, -0.30780561, -0.44230703, -0.03043835, 0.21370063,  0.33344998, -0.40850817, -0.13105809],
 [ 0.08254096,  0.06012654,  0.19821234,  0.40958367,  0.07155838, -0.49830717, 0.09098162,  0.19062183, -0.27312663,  0.39060785],
 [-0.20112519, -0.00593087,  0.05738862,  0.16811148, -0.10466314, -0.21348009, 0.45806193, -0.27659307,  0.2901038,  -0.29736505],
 [-0.18082261,  0.28644149, -0.47549449, -0.3049562,   0.13967768,  0.34904474, 0.20627692,  0.28407868,  0.21849356, -0.01642202]], 
 [-0.14703536, -0.45573697, -0.47563745, -0.08546162, -0.08562345,  0.07636098, -0.42087389, -0.16322197, -0.02759763,  0.0297091 ], 0))
'''

'''
cutoff = len(toy_features)
train_toy_features = toy_features[0:150,]
val_toy_features = toy_features[150:,]

train_toy_labels = toy_labels[0:150]
val_toy_labels = toy_labels[150:]

T = 10
L = 0.01

pct_train_accuracy, pct_val_accuracy = p1.classifier_accuracy(p1.perceptron, train_toy_features,val_toy_features,train_toy_labels,val_toy_labels,T=T)
print("{:35} {:.4f}".format("Training accuracy for perceptron:", pct_train_accuracy))
print("{:35} {:.4f}".format("Validation accuracy for perceptron:", pct_val_accuracy))

avg_pct_train_accuracy, avg_pct_val_accuracy = \
    p1.classifier_accuracy(p1.average_perceptron, train_toy_features,val_toy_features,train_toy_labels,val_toy_labels,T=T)
print("{:43} {:.4f}".format("Training accuracy for average perceptron:", avg_pct_train_accuracy))
print("{:43} {:.4f}".format("Validation accuracy for average perceptron:", avg_pct_val_accuracy))

avg_peg_train_accuracy, avg_peg_val_accuracy = \
    p1.classifier_accuracy(p1.pegasos, train_toy_features,val_toy_features,train_toy_labels,val_toy_labels,T=T,L=L)
print("{:50} {:.4f}".format("Training accuracy for Pegasos:", avg_peg_train_accuracy))
print("{:50} {:.4f}".format("Validation accuracy for Pegasos:", avg_peg_val_accuracy))
'''
# th, th0 = p1.perceptron([[1, 0], [1, -1], [2, 3]],[1, -1, 1], 1)
# val_cls = p1.classify([[1, 1], [2, -1]], th, th0)
# val_acc = p1.accuracy(val_cls, [-1, 1])
#
# print(th, th0, val_cls, val_acc)
#p1.classifier_accuracy(p1.pegasos, train_bow_features,val_bow_features,train_labels,val_labels,T=T,L=L)
# th, th0 = p1.pegasos(train_bow_features,train_labels, 10, 0.01)
# val_cls = p1.classify(val_bow_features, th, th0)
# val_acc = p1.accuracy(val_cls, val_labels)
# print(th, th0, val_cls, val_acc)

#pct_train_accuracy, pct_val_accuracy = p1.classifier_accuracy(p1.perceptron, [[1, 0], [1, -1], [2, 3]],[[1, 1], [2, -1]],[1, -1, 1],[-1, 1],T=1)

#peg_train_accuracy, peg_val_accuracy = p1.classifier_accuracy(p1.pegasos, [[1, 0], [1, -1], [2, 3]],[[1, 1], [2, -1]],[1, -1, 1],[-1, 1],T=1, L=0.2)
#print(pct_train_accuracy, pct_val_accuracy)
#print(peg_train_accuracy, peg_val_accuracy)
