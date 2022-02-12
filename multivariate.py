# -*- coding: utf-8 -*-
"""Multivariate.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/12Wy3GYJMJLwuTfB5V_8TvmHdf_826vnk
"""

import pandas as pd

from google.colab import files
uploaded = files.upload()

sms_spam = pd.read_csv('SMSSpamCollection', sep='\t',
header=None, names=['Label', 'SMS'])

print(sms_spam.shape)
sms_spam.head()

sms_spam['Label'].value_counts(normalize=True)

data_randomized = sms_spam.sample(frac=1, random_state=1)

# Calculate index for split
training_test_index = round(len(data_randomized) * 0.8)

# Split into training and test sets
training_set = data_randomized[:training_test_index].reset_index(drop=True)
test_set = data_randomized[training_test_index:].reset_index(drop=True)

print(training_set.shape)
print(test_set.shape)

training_set['Label'].value_counts(normalize=True)

test_set['Label'].value_counts(normalize=True)

training_set.head(3)
# After cleaning
training_set['SMS'] = training_set['SMS'].str.replace(
   '\W', ' ') # Removes punctuation
training_set['SMS'] = training_set['SMS'].str.lower()
training_set.head(3)

training_set['SMS'] = training_set['SMS'].str.split()
#data_randomized['SMS'] = data_randomized['SMS'].str.split()
vocabulary = []
for sms in training_set['SMS']:
   for word in sms:
      vocabulary.append(word)

vocabulary = list(set(vocabulary))

len(vocabulary)

word_counts_per_sms = {unique_word: [0] * len(training_set['SMS']) for unique_word in vocabulary}

for index, sms in enumerate(training_set['SMS']):
   for word in sms:
      print(word_counts_per_sms[word][index])
      if word_counts_per_sms[word][index]!=1:
        word_counts_per_sms[word][index] = 1

word_counts = pd.DataFrame(word_counts_per_sms)
word_counts.head()

training_set_clean = pd.concat([training_set, word_counts], axis=1)
training_set_clean.head()

spam_messages = training_set_clean[training_set_clean['Label'] == 'spam']
ham_messages = training_set_clean[training_set_clean['Label'] == 'ham']

# P(Spam) and P(Ham)
p_spam = len(spam_messages) / len(training_set_clean)
p_ham = len(ham_messages) / len(training_set_clean)

# N_Spam
n_words_per_spam_message = spam_messages['SMS'].apply(len)
n_spam = n_words_per_spam_message.sum()

# N_Ham
n_words_per_ham_message = ham_messages['SMS'].apply(len)
n_ham = n_words_per_ham_message.sum()

# N_Vocabulary
n_vocabulary = len(vocabulary)

# Laplace smoothing
alpha = 1

parameters_spam = {unique_word:0 for unique_word in vocabulary}
parameters_ham = {unique_word:0 for unique_word in vocabulary}

# Calculate parameters
for word in vocabulary:
   n_word_given_spam = spam_messages[word].sum() # spam_messages already defined
   p_word_given_spam = (n_word_given_spam + alpha) / (n_spam + alpha*n_vocabulary)
   parameters_spam[word] = p_word_given_spam

   n_word_given_ham = ham_messages[word].sum() # ham_messages already defined
   p_word_given_ham = (n_word_given_ham + alpha) / (n_ham + alpha*n_vocabulary)
   parameters_ham[word] = p_word_given_ham

'''def cal_total_prob(word):
  w_sum = training_set_clean[word].sum()
  print(w_sum)
  numR = len(training_set_clean.axes[0])
  print(numR)
  prob = w_sum/numR
  print(prob)
  return prob'''

import re

def classify(message):
   '''
   message: a string
   '''

   message = re.sub('\W', ' ', message)
   message = message.lower().split()

   p_spam_given_message = p_spam
   p_ham_given_message = p_ham
   X=1
   '''for word in message:
      X = X * cal_total_prob(word)'''
   for word in message:
      if word in parameters_spam:
         p_spam_given_message *= parameters_spam[word]
        

      if word in parameters_ham: 
         p_ham_given_message *= parameters_ham[word]

   print('P(Spam|message):', p_spam_given_message)
   print('P(Ham|message):', p_ham_given_message)
   p_sam_with_total_c = p_spam_given_message
   p_ham_with_total_c = p_ham_given_message
   #p_sam_with_total_c = p_spam_given_message/X
   #p_ham_with_total_c = p_ham_given_message/X
   if p_ham_with_total_c > p_sam_with_total_c:
      return 'ham'
   elif p_ham_with_total_c < p_sam_with_total_c:
      return 'spam'
   else:
      return 'ham'

classify("Sounds good, Tom, then see u there")

test_set['predicted'] = test_set['SMS'].apply(classify)
test_set.head()

correct = 0
total = test_set.shape[0]

for row in test_set.iterrows():
   row = row[1]
   if row['Label'] == row['predicted']:
      correct += 1

print('Correct:', correct)
print('Incorrect:', total - correct)
print('Accuracy:', correct/total)