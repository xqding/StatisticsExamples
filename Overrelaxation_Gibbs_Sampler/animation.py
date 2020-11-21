import numpy as np
import numpy.random as nr
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

Mu = np.array([0,0])
Sigma = np.array([[1,0.7],[0.7,1]])
Lambda = np.linalg.inv(Sigma)

rho = Sigma[0,1]
lim = 3*rho
x1, x2 = np.mgrid[-lim:lim:.01, -lim:lim:.01]
pos = np.empty(x1.shape + (2,))
pos[:, :, 0] = x1; pos[:, :, 1] = x2
rv = multivariate_normal(Mu, Sigma)


fig = plt.figure()
ax = fig.add_subplot(111)
plt.ion()
plt.contourf(x1, x2, rv.pdf(pos))
plt.colorbar()
fontsize = 20
plt.xlabel("$x_1$", fontsize = fontsize)
plt.ylabel("$x_2$", fontsize = fontsize)

x1 = 0
x2 = 0
N = 1
markersize = 5
plt.plot([x1],[x2], "ok", markersize = markersize)
print("Num of points:", N, "Initial sample: " "(x1, x_2) = ", (x1, x2))
def onclick(event):
    global N, x1, x2
    if N % 2 == 1:
        cond_mu1 = Mu[0] - Lambda[1,0]*(x2 - Mu[1])/Lambda[0,0]
        cond_sigma1 = 1/Lambda[0,0]
        x1 = nr.normal(cond_mu1, cond_sigma1)
        print("Num of points:", N, ", update x1:", "(x1, x_2) = ", (x1, x2))
        plt.plot([x1], [x2], "ok", markersize = markersize)
        plt.draw()
    if N % 2 == 0:        
        cond_mu2 = Mu[1] - Lambda[0,1]*(x1 - Mu[0])/Lambda[1,1]
        cond_sigma2 = 1/Lambda[1,1]
        x2 = nr.normal(cond_mu2, cond_sigma2)
        print("Num of points:", N, ", update x2: ", "(x1, x_2) = ", (x1, x2))
        plt.plot([x1], [x2], "ok", markersize = markersize)
        plt.draw()
    N += 1
    
cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()


# N = 40 # number of samples
# x1_samples = [0]
# x2_samples = [0]
# x2 = 0.5
# for i in range(N):
#     cond_mu1 = Mu[0] - Lambda[1,0]*(x2 - Mu[1])/Lambda[0,0]
#     cond_sigma1 = 1/Lambda[0,0]
#     x1 = nr.normal(cond_mu1, cond_sigma1)
#     x1_samples.append(x1)
    
#     cond_mu2 = Mu[1] - Lambda[0,1]*(x1 - Mu[0])/Lambda[1,1]
#     cond_sigma2 = 1/Lambda[1,1]
#     x2 = nr.normal(cond_mu2, cond_sigma2)
#     x2_samples.append(x2)


# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(np.random.rand(10))

# def onclick(event):
#     print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
#           (event.button, event.x, event.y, event.xdata, event.ydata))

# cid = fig.canvas.mpl_connect('button_press_event', onclick)
# plt.show()
