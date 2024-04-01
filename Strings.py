import random
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, lognorm

#%% bowl_game
def bowl_game(n_initial):
    bowl = []
    loops = []
    for i in range(n_initial):
        bowl.append([1,1])
    while len(bowl) != 0:
        end1 = random.randint(0, 2*len(bowl) - 1)
        end2 = random.randint(0, 2*len(bowl) - 1)
        while end1 == end2:
            end2 = random.randint(0, 2*len(bowl) - 1)
        
        #swap so end1 is the smaller number and end2 is the larger number 
        t = min(end1, end2)
        end2 = max(end1, end2)
        end1 = t
    
        if ((end1 % 2 == 0) and (end1 == (end2 - 1))):
            loop_length = bowl[int(end1 / 2)][0]
            loops.append([loop_length, loop_length])
            del(bowl[int(end1 / 2)])
        else:
            new_length = bowl[int(math.floor(end1 / 2))][end1 % 2] + bowl[int(math.floor(end2 / 2))][end2 % 2]
            del(bowl[int(math.floor(end2/ 2))])
            del(bowl[int(math.floor(end1/ 2))])
            bowl.append([new_length, new_length])

    return loops

#%% Expected number of loops
n = 10

# finding the number of loops for lots of iteration
number_of_loops = []
for i in range(10000):
    loops = bowl_game(n)
    number_of_loops.append(len(loops))

# finding the expected number of loops using formula
expected_loops = 0
for i in range(1, n + 1):
    expected_loops += 1/(2*i-1)
    
print("Expected number of loops for n = "+ str(n)+":", sum(number_of_loops)/len(number_of_loops))
print("Expected number of loops for n = "+ str(n) +" using formula:", expected_loops)

#%%

# creating an array to store RMSE values in 
RMSE = np.zeros(4)

#%% Plot histogram

bins = np.arange(0, n+1, 1)
mid_bins = np.zeros(len(bins)-1)   
for i in range(len(mid_bins)):
    mid_bins[i] = (bins[i + 1] - bins[i])/2 + bins[i]     

plt.figure()
sns.histplot(number_of_loops, bins=mid_bins, stat='density').set(title = 'Results when $n = 100$', xlabel = "Number of loops in the bowl")
plt.xlim(0, 10)
plt.show()

#%% applying a normal distribution
fig = sns.histplot(number_of_loops,bins=mid_bins, stat='density')
fig.set_title('$n=100$ with normal curve of best fit')
x2=np.linspace(0.5,max(number_of_loops)) 
MU = np.mean(number_of_loops)
SIG = np.sqrt(np.var(number_of_loops,ddof=1)) 
y2 = norm.pdf(x2,MU,SIG)
fig.plot(x2,y2,'r') 
fig.grid('on') 
plt.xlim(0,10)
plt.xlabel("Number of loops in the bowl")
plt.show(fig) 

#%% applying a log normal distribution
fig = sns.histplot(number_of_loops,bins=mid_bins, stat='density')
fig.set_title('$n=100$ with lognormal curve of best fit')
x2 = np.linspace(min(number_of_loops),max(number_of_loops)) 
y2 = lognorm.pdf(x2,  np.std(np.log(number_of_loops)), loc = 0, scale = np.exp(np.mean(np.log(number_of_loops))))
fig.plot(x2,y2,'r') 
fig.grid('on') 
plt.xlim(0,10)
plt.xlabel("Number of loops in the bowl")
plt.show(fig) 

#%% Histogram data and removing zeros so able to take logs
hist_out, bin_edges = np.histogram(number_of_loops, bins=mid_bins)
new_mid_bins = np.zeros(len(bin_edges) - 1)
for i in range(len(new_mid_bins)):
    new_mid_bins[i] = (bin_edges[i + 1] - bin_edges[i])/2 + bin_edges[i]  

hist_out_noZeros_idx = np.argwhere(hist_out>0)
hist_out_noZeros = hist_out[hist_out_noZeros_idx]
hist_out_noZeros_log = np.log(hist_out_noZeros)
mid_bins_noZeros = new_mid_bins[hist_out_noZeros_idx]

#%% applying 2nd order polynomial

order = 2 # can change to get different order polynomials
model_poly_2 = np.polyfit(mid_bins_noZeros[:-1,0], hist_out_noZeros[:-1,0], order)
bestfitk_poly_2 = np.polyval(model_poly_2, mid_bins_noZeros)
plt.plot(mid_bins_noZeros, hist_out_noZeros, "x")
plt.plot(mid_bins_noZeros,bestfitk_poly_2)
plt.title('$n=100$ with 4nd order polynomial from polyfit') 
plt.grid('on') 
plt.xlabel("Final number of loops")
plt.ylabel("Density")
plt.show() 

# finding RMSE for this case
MSE = np.square(np.subtract(hist_out_noZeros, bestfitk_poly_2)).mean()
RMSE[0]= np.sqrt(MSE)

#%% Curve Fitting - log - degree 2

# finding polynomial of degree 2
poly_deg = 2
polynomial_fit_coeff = np.polyfit(mid_bins_noZeros[:-1,0], hist_out_noZeros_log[:-1,0], poly_deg)
y_polynomial_2 = np.polyval(polynomial_fit_coeff, mid_bins_noZeros)

e_y = np.exp(y_polynomial_2)

plt.plot(mid_bins_noZeros, hist_out_noZeros_log, 'x')
plt.plot(mid_bins_noZeros, y_polynomial_2)
plt.title("$n=100$ with log ployfit best fit degree 2")
plt.xlabel("Final number of loops")
plt.ylabel("log(Density)")
plt.show()

plt.plot(mid_bins_noZeros, hist_out_noZeros, 'x')
plt.plot(mid_bins_noZeros, e_y)
plt.title("$n=100 with exponential of log ployfit of degree 2" )
plt.xlabel("Final number of loops")
plt.ylabel("Density")
plt.grid('on') 
plt.show()

# finding RMSE for this case
MSE = np.square(np.subtract(hist_out_noZeros, e_y)).mean()
RMSE[1]= np.sqrt(MSE)

#%% Curve Fitting - log - degree 3

# finding polynomial of degree 3
poly_deg = 3
polynomial_fit_coeff = np.polyfit(mid_bins_noZeros[:-1,0], hist_out_noZeros_log[:-1,0], poly_deg)
y_polynomial_3 = np.polyval(polynomial_fit_coeff, mid_bins_noZeros)

e_y = np.exp(y_polynomial_3)

plt.plot(mid_bins_noZeros, hist_out_noZeros_log, 'x')
plt.plot(mid_bins_noZeros, y_polynomial_3)
plt.title("$n=100$ with log ployfit best fit degree 3")
plt.xlabel("Final number of loops")
plt.ylabel("log(Density)")

plt.show()

plt.plot(mid_bins_noZeros, hist_out_noZeros, 'x')
plt.plot(mid_bins_noZeros, e_y)
plt.title("$n=100$ with exponential of log ployfit of degree 3" )
plt.xlabel("Final number of loops")
plt.ylabel("Density")
plt.grid('on') 
plt.show()

# finding RMSE for this case
MSE = np.square(np.subtract(hist_out_noZeros, e_y)).mean()
RMSE[2]= np.sqrt(MSE)

#%% Expected number of self-loops

n = 10

number_self_loops_current = 0
number_self_loops = []

# calculating the number of self-loops each time the game is played
for i in range(10000):
    number_self_loops_current = 0
    bowl = bowl_game(n)
    for j in range(len(bowl)):
        if bowl[j][0] == 1:
            number_self_loops_current += 1
    number_self_loops.append(number_self_loops_current)

print("Expected number self of loops for n = "+ str(n)+":", sum(number_self_loops)/len(number_self_loops))
print("Expected number of self-loops using formula is", n/(2*n - 1))