# Load libraries
import pandas as pd
import numpy as np
from scipy.special import betaln
from scipy.stats import beta
from matplotlib import pyplot as plt
import scipy.stats as st
from scipy import special

# import pyreadstat

# Import data
wd = 'C:/Users/philip.sjoberg/Desktop/Master thesis/Data/'
file = 'Data_validation_test.xlsx'

# Read data
df_org = pd.read_excel(wd + file)

# Standard approach
df_1_org = df_org.iloc[:100, : 4]

# Proposed approach
df_2_org = df_org.iloc[:100, 4:]


# Factor analysis - PCA

def pca(df, missing):
    df = df.replace(missing, np.nan)
    print('\n**PCA analysis**')
    print('\nMissing values % df')
    print(df.isnull().sum() / len(df))
    # Change to np array, drop missing values
    df_array = np.array(df.dropna())

    # Get covariance matrix
    cov_data = np.corrcoef(df_array.T)
    # print('\nCorrelation table df 1')
    # print(cov_data)

    # PCA
    no_of_components = len(cov_data)
    # Calculate eigenvalues and vectors
    # Vectors are column vise
    pca_eigen_values, pca_eigen_vectors = np.linalg.eig(cov_data)
    # pca1_eigen_vectors = pca1_eigen_vectors.T

    # Sort the eigenvalues
    pca_sorted_components = np.argsort(pca_eigen_values)[::-1]

    # Calculate proportion of variance explained
    pca_projection_matrix = pca_eigen_vectors[pca_sorted_components[:no_of_components]]
    pca_explained_variance = pca_eigen_values[pca_sorted_components]
    pca_explained_variance_ratio = pca_explained_variance / pca_eigen_values.sum()

    # Keep all items with |loading| > 0.4 in component one
    print('\nEigenvalues:\n',
          np.row_stack((pca_sorted_components, pca_explained_variance, pca_explained_variance_ratio)))
    print('\nEigenvectors:\n', np.row_stack((pca_sorted_components, pca_projection_matrix)))


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
print('\nCronbachs alpha: DF 1:', round(a1, 3), ', DF 2:', round(a2, 3))


# Bayes Factor for non-response

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
    print('\nProportion of missing values in data one: ', round(s1 / n1, 3))
    p_y1 = np.exp(chooseln(n1, s1) + betaln(alpha + s1, beta + n1 - s1) - betaln(alpha, beta))

    n2 = df2.count().sum()
    df2 = df2.replace(missing, np.nan)
    s2 = df2.isna().sum().sum()
    print('Proportion of missing values in data two: ', round(s2 / n2, 3))
    p_y2 = np.exp(chooseln(n2, s2) + betaln(alpha + s2, beta + n2 - s2) - betaln(alpha, beta))

    BF_12 = p_y1 / p_y2

    return BF_12



def postPlot(prior):
    from scipy.stats import beta
    a, b = prior
    a = a
    b = b
    distri = beta(a, b)
    x = np.linspace(0, 1, 300)
    x_pdf = distri.pdf(x)
    plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.plot(x, x_pdf, label=fr"$\alpha$ = {a:d}, $\beta$ = {b:d}")
    plt.yticks([])
    plt.xlabel("$\\theta$")
    plt.legend()


priors = ((1, 1), (30, 30), (50, 50), (2, 4))
for theta in priors:
    postPlot(theta)

'''
The priors have a big impact on the Bayes Factor.
For a non-informative, the BF is very significant,
for more realistic priors, the BF is not significant...
'''

prior = (2, 4)  # (alpha1 , beta1) , (alpha2 , beta2)

BF = BF_beta_binom(df_1_org, df_2_org, 98, prior)

# BF_12: >100 Extreme evidence for M1, <1/100 Extreme evidence for M2
print('\nBayes Factor: ', BF)
print('BF > 10  :', BF > 10)
print('BF < 1/10:', BF < 1 / 10)




# Test of proportions (frequentist approach)

# H_0: p1 = p2
# H_1: p1 > p2

def Z_scoreProp(df1, df2, missing, a):
    n1 = df1.count().sum()
    df1 = df1.replace(missing, np.nan)
    s1 = df1.isna().sum().sum()
    p1 = s1 / n1

    n2 = df2.count().sum()
    df2 = df2.replace(missing, np.nan)
    s2 = df2.isna().sum().sum()
    p2 = s2 / n2

    p = (s1 + s2) / (n1 + n2)

    # z-score for comparing two proportions
    z = (p1 - p2) / np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))

    st.norm.ppf(.05)
    p_z = 1 - st.norm.cdf(z)
    print('\nHypothesis test:')
    print('Z-score: ', z)
    print('P-value: ', p_z)
    return p_z


# We compare first input >/< second input
hypothesis_test = Z_scoreProp(df_2_org, df_1_org, 98, 0.05)
