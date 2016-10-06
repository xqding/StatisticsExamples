import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.neighbors import KernelDensity

## This excise is repeating the example 4.5 in the Book "Monte Carlo Statistical Method (by Robert, Christian, Casella, George)"
## sample from student's t distribution T(nu, mu, sigma^2) using Dickey's decomposition
## Y ~ Gamma(nu/2, 2*sigma**2/nu)
## X|y  ~ Normal(mu, 1/y)
## Then X ~ T(nu, mu, sigma^2)

## parameters
nu = 4.6
mu = 0.0
sigma = 1.0

## true underlying density
X = np.linspace(-10,10,200)
original_den = stats.t.pdf(X, nu, loc = mu, scale = sigma)

#### calcuate absolute error (AE) for naive method and Rao-Blackwellization method
## simulate (X,Y) from the joint distribution
N = 1000
data = []
X_sample = []
Y_sample = []
for i in range(N):
    y = np.random.gamma(nu/2.0, 2.0*sigma**2/nu)
    x = np.random.normal(mu, np.sqrt(1/y))
    X_sample.append(x)
    Y_sample.append(y)
    data.append((x,y))
X_sample = np.reshape(X_sample, (-1,1))

naive_AE = []
RaoBlackwellized_AE = []
for k in range(50,N+1,50):
    print k
    ## density estimation using kernel method
    kde = KernelDensity(kernel = "gaussian", bandwidth = 0.5).fit(X_sample[0:k])
    log_den = kde.score_samples(np.reshape(X, (-1,1)))
    naive_den = np.exp(log_den)
    naive_AE.append(np.sum(np.abs(naive_den - original_den)))
    
    ## density estimation using Rao-Blackwellization
    RaoBlackwellized_den = []
    for x in X:
        cumsum = 0.0
        for y in Y_sample[0:k]:
            cumsum += stats.norm.pdf(x, loc = mu, scale = np.sqrt(1.0/y))
        RaoBlackwellized_den.append(cumsum / k)
    RaoBlackwellized_den = np.array(RaoBlackwellized_den)
    RaoBlackwellized_AE.append(np.sum(np.abs(RaoBlackwellized_den - original_den)))
    
plt.clf()
plt.plot(range(50,N+1,50), naive_AE, label = "naive", color = "red", linewidth = 3)
plt.plot(range(50,N+1,50), RaoBlackwellized_AE, label = "Rao-Blackwellization", color = "blue", linewidth = 3)
plt.xlabel("num of samples", fontsize = 20)
plt.ylabel("absolute error", fontsize = 20)
plt.title("Absolute Error for Density Estimation", fontsize = 20)
plt.legend()
plt.savefig("densityEstimationAbsoluteError.png")
#plt.show()


# #### plot a sample 
# ## simulate (X,Y) from the joint distribution
# N = 500
# data = []
# X_sample = []
# Y_sample = []
# for i in range(N):
#     y = np.random.gamma(nu/2.0, 2.0*sigma**2/nu)
#     x = np.random.normal(mu, np.sqrt(1/y))
#     X_sample.append(x)
#     Y_sample.append(y)
#     data.append((x,y))
# X_sample = np.reshape(X_sample, (-1,1))

# ## density estimation using kernel method
# kde = KernelDensity(kernel = "gaussian", bandwidth = 0.5).fit(X_sample)
# log_den = kde.score_samples(np.reshape(X, (-1,1)))
# naive_den = np.exp(log_den)

# ## density estimation using Rao-Blackwellization
# RaoBlackwellized_den = []
# for x in X:
#     print x
#     cumsum = 0.0
#     for y in Y_sample:
#         cumsum += stats.norm.pdf(x, loc = mu, scale = np.sqrt(1.0/y))
#     RaoBlackwellized_den.append(cumsum / N)

# naive_AE = np.sum(np.abs(naive_den - original_den))
# RaoBlackwellized_AE = np.sum(np.abs(RaoBlackwellized_den - original_den))
# print "naive: {0:.4f}, Rao-Blackwellization: {1:.4f}".format(naive_AE, RaoBlackwellized_AE)
# plt.clf()
# plt.fill(X, original_den, label = "original", fc='black', alpha=0.2)
# plt.plot(X, naive_den, label = "naive:{0:.3f}".format(naive_AE), color = "red", linewidth = 2)
# plt.plot(X, RaoBlackwellized_den, label = "Rao-Blackwell:{0:.3f}".format(RaoBlackwellized_AE), color = "blue", linewidth = 2)
# plt.title("Estimated Density with 500 Samples", fontsize = 20)
# plt.xlim((-12,12))
# plt.legend(fontsize = 13)
# plt.savefig("densityEstimationExamples.png")
