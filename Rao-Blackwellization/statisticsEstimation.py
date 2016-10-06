import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

## This excise is repeating the example 4.5 in the Book "Monte Carlo Statistical Method (by Robert, Christian, Casella, George)"
## sample from student's t distribution T(nu, mu, sigma^2) using Dickey's decomposition
## Y ~ Gamma(nu/2, 2*sigma**2/nu)
## X|y  ~ Normal(mu, 1/y)
## Then X ~ T(nu, mu, sigma^2)

## parameters
nu = 4.6
mu = 0.0
sigma = 1.0

### repeat estimations of exp(-x^2) (True value: 0.537332)to get the variance of the estimation
trueValue = 0.537332
R = 50 # num of repeats
naive = []
RaoBlackwellization = []
native = []
r = 1
while r <= 50:
    print r
    ## simulate (X,Y) from the joint distribution
    N = 2000
    data = []
    for i in range(N):
        y = np.random.gamma(nu/2.0, 2.0*sigma**2/nu)
        x = np.random.normal(mu, np.sqrt(1/y))
        data.append((x,y))

    ## naive estimate exp(-x^2)
    naiveEstimate = []
    naiveCumsum = 0.0
    for i in range(N):
        naiveCumsum += np.exp(-data[i][0]**2)
        naiveEstimate.append(naiveCumsum / float(i+1))
    naive.append(naiveEstimate);
    
    ## Rao-Blackwellization estimate exp(-x^2)    
    RaoBlackwellizedEstimate = []
    RaoBlackwellizedCumsum = 0.0
    for i in range(N):
        RaoBlackwellizedCumsum += 1.0 / np.sqrt(2.0 / data[i][1] + 1)
        RaoBlackwellizedEstimate.append(RaoBlackwellizedCumsum / float(i+1))
    RaoBlackwellization.append(RaoBlackwellizedEstimate)
    r+=1

    ## simulate X from the native T distribution
    nativeX = stats.t.rvs(nu, loc = mu, scale = sigma, size = N)
    nativeEstimate = []
    nativeCumsum = 0
    for i in range(N):
        nativeCumsum += np.exp(-nativeX[i]**2)
        nativeEstimate.append(nativeCumsum / float(i+1))
    native.append(nativeEstimate);


naive = np.array(naive)
naiveMean = naive.mean(0)
naiveStd = np.sqrt(((naive - trueValue)**2).mean(0))

RaoBlackwellization = np.array(RaoBlackwellization)
RaoBlackwellizationMean = RaoBlackwellization.mean(0)
RaoBlackwellizationStd = np.sqrt(((RaoBlackwellization - trueValue)**2).mean(0))

native = np.array(native)
nativeMean = native.mean(0)
nativeStd = np.sqrt(((native - trueValue)**2).mean(0))

plt.clf()
plt.plot(naiveStd, label = "naive", color = "red", linewidth = 1.5)
plt.plot(nativeStd, label = "native", color = "black", linewidth = 1.5)
plt.plot(RaoBlackwellizationStd, label = "Rao-Blackwellization", color = "blue", linewidth = 1.5)
plt.ylim(-0.05)
plt.legend(fontsize = 20)
plt.title("Estimation of exp(-X^2) for Student's t distribution", fontsize = 20)
plt.xlabel("num of draws", fontsize = 20)
plt.ylabel("RMSE", fontsize = 20)
plt.savefig("staticsEstimationRMSE.png")

## plot some sample estimations
plt.clf()
plt.plot([trueValue for i in range(N)], label = "True Value", color = "cyan")
plt.plot(naive[0,:], label = "naive", color = "red")
plt.plot(naive[10,:], color = "red")
plt.plot(native[0,:], label = "native", color = "black")
plt.plot(native[10,:], color = "black")
plt.plot(RaoBlackwellization[0,:], label = "Rao-Blackwellization", color = "blue")
plt.plot(RaoBlackwellization[10,:], color = "blue")
plt.ylim((0.45, 0.65))
plt.legend(fontsize = 20)
plt.title("Examples for Estimation of exp(-X^2) of Student's t distribution", fontsize = 15)
plt.xlabel("num of draws")
plt.ylabel("estimated value")
plt.savefig("staticsEstimationexamples.png")
