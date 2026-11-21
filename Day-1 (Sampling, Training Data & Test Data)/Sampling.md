     1. First, understand sampling

Imagine you want to understand the average monthly spending of students in India.

There might be millions of students. Asking every student about their spending would be impractical.

So, you select 5,000 students and calculate their average spending.

This is called sampling.

There are two important terms:
- Population: The entire group you want to study.
- Sample: A smaller subset selected from that population.


<img width="1009" height="584" alt="image" src="https://github.com/user-attachments/assets/25cf6016-0d61-4fd1-82b2-0553278bcf60" />



For example:

Term	Example

Population	All IIT Madras students

Sample	500 selected students

Parameter	Actual average spending of all students

Statistic	Average spending calculated from 500 students

The important idea: We use sample statistics to estimate population parameters.

Why does sampling matter in Data Science?

Because your dataset is almost never the entire population.

Your model learns from the data you provide. If your sample doesn't represent the real-world population, your model can learn the wrong patterns.

For example:

Suppose you build a loan-default prediction model using data containing:
- 80% salaried individuals

- 15% self-employed individuals

- 5% business owners
But your actual customer population contains 40% salaried individuals, 35% self-employed individuals and 25% business owners.

Your training data doesn't adequately represent the population where the model will be used.

This is a sampling and representation problem.

----

    2. Types of sampling every Data Scientist should know










1. Simple Random Sampling

Every individual has an equal chance of being selected.

Example: Randomly selecting 500 students from a college database.













2. Stratified Sampling

Divide the population into groups (strata), then sample from each group.

Example: Selecting students proportionally from different departments.













3. Systematic Sampling

Select every kth observation after a starting point.

Example: Selecting every 10th customer from a customer list.













4. Cluster Sampling

Divide the population into clusters and randomly select entire clusters.

Example: Randomly selecting 10 schools and surveying all students in those schools.



Interview insight: Stratified sampling and cluster sampling are not the same.
- Stratified: Take some observations from every group.
- Cluster: Select some groups and study their members.
