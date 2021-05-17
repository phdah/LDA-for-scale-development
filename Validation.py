# Load libraries
import pandas as pd
import numpy as np
from scipy.special import betaln
from matplotlib import pyplot as plt
import scipy.stats as st
from scipy import special

# Import data
wd = 'C:/Users/'
file_standard = 'Sustainability_standard.xlsx'
file_proposed = 'Sustainability_text_analysis.xlsx'

# Standard approach
df_1_org = pd.read_excel(wd + file_standard)

# Proposed approach
df_2_org = pd.read_excel(wd + file_proposed)


# Factor analysis - PCA

def pca(df, missing):
    df = df.replace(missing, np.nan)
    print('\n**PCA analysis**')
    print('\nMissing values % df')
    print(df.isnull().sum() / len(df))
    # Change to np array, drop missing values
    df_array = np.array(df.dropna())

    print("\nObservations left:", len(df_array))
    # Get covariance matrix
    cov_data = np.corrcoef(df_array.T)
    # cov_data = np.cov(df_array.T)

    # PCA
    no_of_components = len(cov_data)
    # Calculate eigenvalues and vectors
    # Vectors are column vise
    np.random.seed(101)
    pca_eigen_values, pca_eigen_vectors = np.linalg.eig(cov_data)
    # pca1_eigen_vectors = pca1_eigen_vectors.T

    # Sort the eigenvalues
    pca_sorted_components = np.argsort(pca_eigen_values)[::-1]

    # Calculate proportion of variance explained
    pca_projection_matrix = pca_eigen_vectors[pca_sorted_components[:no_of_components]]
    pca_explained_variance = pca_eigen_values[pca_sorted_components]
    pca_explained_variance_ratio = pca_explained_variance / pca_eigen_values.sum()

    # Keep all items with |loading| > 0.3 in component one
    print('\nEigenvalues:\n',
          np.row_stack((pca_sorted_components, pca_explained_variance.round(1), pca_explained_variance_ratio.round(2))))
    print('\nEigenvectors:\n', np.row_stack((pca_sorted_components, pca_projection_matrix.round(3))))


pca(df_1_org, 98)
pca(df_2_org, 98)


# Internal consistency - Cronbach's alpha

def CronAlpha(df, missing):
    df = df.replace(missing, np.nan)
    df = df.dropna()

    k = df.shape[1]
    c = sum(df.var())
    v = df.sum(axis=1).var()
    # Cronbach's alpha > 0.7 is sufficient
    a = (k / (k - 1)) * (1 - c / v)
    return a


a1 = CronAlpha(df_1_org, 98)
a2 = CronAlpha(df_2_org, 98)
print('\nCronbachs alpha\nStandard approach:', round(a1, 3), ', Text analysis:', round(a2, 3))


# Bayes Factor for non-informative response

# Log binom function
def chooseln(N, k):
    return special.gammaln(N + 1) - special.gammaln(N - k + 1) - special.gammaln(k + 1)


# Bayes Factor function
def BF_beta_binom(df1, df2, missing, prior):
    """
    theta~Beta(a,pha, beta)
    y~Bin(n, theta)
    Compute the marginal likelihood, analytically, for a beta-binomial model
    logarithm to prevent underflow
    Output Bayes Factor
    """
    alpha, beta = prior
    n1 = df1.count().sum()
    df1 = df1.replace(missing, np.nan)
    s1 = df1.isna().sum().sum()
    print('\nProportion of missing values in Standard approach: ', round(s1 / n1, 3), " ( n=", n1, ")")

    n2 = df2.count().sum()
    df2 = df2.replace(missing, np.nan)
    s2 = df2.isna().sum().sum()
    print('Proportion of missing values in Text analysis: ', round(s2 / n2, 3), " ( n=", n2, ")")

    BF_12 = np.exp(chooseln(n1, s1) + betaln(alpha + s1, beta + n1 - s1) - chooseln(n2, s2) - betaln(alpha + s2, beta + n2 - s2))

    return BF_12


BF = BF_beta_binom(df_1_org, df_2_org, 98, prior)

# BF_12: >100 Extreme evidence for M1, <1/100 Extreme evidence for M2
print('\nBayes Factor: ', BF.round(2))

# Test multiple priors
def BF_multiPriors(df1, df2, missing, mean, a):

    BF = []
    for a1 in a:

        b = (a1/mean) - a1
        prior = ((a1, b))
        PriorPlot(prior)
        print("Mean: ", mean, "\nPrior parameters: ", prior)

        BF.append(BF_beta_binom(df1, df2, missing, prior).round(1))
    return BF

a = (2,5,10,20,30,70,100,150)


BF = BF_multiPriors(df_1_org, df_2_org, 98, 0.45, a)
BF
