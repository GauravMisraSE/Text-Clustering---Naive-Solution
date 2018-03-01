# Text-Clustering---Naive-Solution
 Implement the Bisecting K-Means algorithm.  Deal with text data (news records) in document-term sparse matrix format.  Design a proximity function for text data.  Think about the Curse of Dimensionality.  Think about best metrics for evaluating clustering solutions.

## Detailed Description:
For the purposes of this assignment, you will implement the bisecting k-Means clustering algorithm. You may not use libraries for this portion of your assignment. Additionally, you will gain experience with internal cluster evaluation metrics.
Input data (provided as training data) consists of 8580 text records in sparse format. No labels are provided.
For evaluation purposes (leaderboard ranking), we will use the Normalized Mutual Information Score (NMI), which is an external index metric for evaluating clustering solutions. Essentially, your task is to assign each of the instances in the input data to K clusters identified from 1 to K.

For the leaderboard evaluation, set K to 7. The leaderboard will report the NMI on 50% samples from the dataset.
The train.dat file is a simple CSR sparse matrix containing the features associated with different feature ids in the input file. It differs from previous train.dat files in that it does not contain labels as the first element in each row.

## Some things to note:
 The public leaderboard shows results for 50% of randomly chosen test instances only. This is a standard practice in data mining challenges to avoid gaming of the system. The private leaderboard will be released after the deadline and evaluates all the entries in the data set.
 Each day, you can submit a prediction file up to 5 times.
 The final ranking will always be based on the last submission.
 format.dat shows an example file containing 8580 rows with random cluster assignments from 1 to 7.
 There are no test.dat files in this assignment.
