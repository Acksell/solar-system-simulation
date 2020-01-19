from matplotlib import pyplot as plt

errors = [
 [1.73042097e-10, 5.83455136e-10, 1.79790316e-08, 2.92655127e-07],
 [1.73127827e-10, 5.83644064e-10, 1.79940539e-08, 2.98756131e-07]]

timesteps = [10,100,1000,10000]
plt.plot(timesteps,errors[0])
plt.plot(timesteps,errors[1])
plt.legend(["Leapfrog", "RungeKutta"])
plt.yscale("log")
plt.xscale("log")
plt.ylabel("$\Delta qx_{rel}$")
plt.show()