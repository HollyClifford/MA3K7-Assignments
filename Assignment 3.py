import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

#%%

def hat_picking(i_max, min_num, max_num):
    ans_vect = np.zeros([i_max])
    hat_array = np.arange(min_num, (max_num + 1))
    for i in range(0, i_max):
        
        if (i % 1000) == 0:
            print(f'i = {i} of {i_max}')
            
        hat_list = list(hat_array)

        while len(hat_list)>1:
            hat_list_numel = len(hat_list)
            rand_select_num1_idx = random.randint(0, (hat_list_numel - 1))
            rand_select_num1 = hat_list[rand_select_num1_idx]
            del hat_list[rand_select_num1_idx]

            hat_list_numel = len(hat_list)
            rand_select_num2_idx = random.randint(0, (hat_list_numel - 1))
            rand_select_num2 = hat_list[rand_select_num2_idx]
            del hat_list[rand_select_num2_idx]

            difference_val = abs(rand_select_num2 - rand_select_num1)
            hat_list.append(difference_val)

        ans_vect[i] = hat_list[0]
    return ans_vect

num_it = 10000
bin_max = 9
ans_vect = hat_picking(num_it, 5, bin_max)

#%%

plt.figure()
sns.histplot(ans_vect, bins=9, stat='density').set(title = 'Results when $min = 5$ and $max = 9$', xlabel = "Final number in the hat")
plt.show()


#%%

# data previously run to make the results quicker - has 100000 iterations
# ans_vect = np.load("array2024_100000.npy")

#%%
final_vals_array = ans_vect


bin_width = 1

bin_min = 0

bin_max_ad = bin_max + bin_width/2
bin_min_ad = bin_min - bin_width/2

bin_array = np.arange(bin_min_ad, bin_max_ad, bin_width)

hist_out, bin_edges = np.histogram(ans_vect, bins=bin_array)
bin_mids = 0.5 * ( bin_edges[0:-1] + bin_edges[1:] )


hist_out_scaled = hist_out / sum(hist_out)

hist_out_noZeros_idx = np.argwhere(hist_out>0)

hist_out_noZeros = hist_out[hist_out_noZeros_idx]

hist_out_noZeros_scaled = hist_out_noZeros / sum(hist_out)

bin_mids_ho_noZeros = bin_mids[hist_out_noZeros_idx]

#%%

# creating an array to store RMSE values in 
RMSE = np.zeros(3)

#%%


# finding polynomial of degree 1
poly_deg = 1
polynomial_fit_coeff_2 = np.polyfit(bin_mids_ho_noZeros[:,0], hist_out_noZeros_scaled[:,0], poly_deg)
y_polynomial_2 = np.polyval(polynomial_fit_coeff_2, bin_mids_ho_noZeros[:,0])

plt.plot(bin_mids_ho_noZeros, hist_out_noZeros_scaled, marker = "x", linestyle = "")
plt.plot(bin_mids_ho_noZeros, y_polynomial_2)
plt.title("$n = 2025$ with polynomial best fit degree 1") 
plt.xlabel("Final Number")
plt.ylabel("Density")
plt.plot()
plt.grid('on')
plt.show()

# finding RMSE for this case
MSE = np.square(np.subtract(bin_mids_ho_noZeros[:,0],y_polynomial_2)).mean()
RMSE[0]= np.sqrt(MSE)

# finding polynomial of degree 2
poly_deg = 2
polynomial_fit_coeff_2 = np.polyfit(bin_mids_ho_noZeros[:,0], hist_out_noZeros_scaled[:,0], poly_deg)
y_polynomial_2 = np.polyval(polynomial_fit_coeff_2, bin_mids_ho_noZeros[:,0])

plt.plot(bin_mids_ho_noZeros, hist_out_noZeros_scaled, marker = "x", linestyle = "")
plt.plot(bin_mids_ho_noZeros, y_polynomial_2)
plt.title("$n = 3024$ with polynomial best fit degree 2") 
plt.xlabel("Final Number")
plt.ylabel("Density")
plt.plot()
plt.grid('on')
plt.show()

# finding RMSE for this case
MSE = np.square(np.subtract(hist_out_noZeros_scaled[:,0],y_polynomial_2)).mean()
RMSE[1]= np.sqrt(MSE)

#%%

# making exponential best fit 
hist_out_noZeros_scaled_log = np.log(hist_out_noZeros_scaled)

poly_deg = 1 #degree of the polynomial fit
polynomial_fit_coeff = np.polyfit(bin_mids_ho_noZeros[:,0], hist_out_noZeros_scaled_log[:,0], poly_deg)

y_log_fit = np.polyval(polynomial_fit_coeff, bin_mids_ho_noZeros[:,0])

y_fit = np.exp(y_log_fit)


# Log graph w. best fit
plt.figure();
plt.title("$n = 3024$ with log ployfit best fit degree 2") 
plt.plot(bin_mids_ho_noZeros, hist_out_noZeros_scaled_log, 'x')
plt.xlim([0, 2000])
plt.plot(bin_mids_ho_noZeros, y_log_fit)
plt.xlabel("Final Number")
plt.ylabel("log(Density)")
plt.grid('on')


# Normal graph w. best fit
plt.figure();
plt.plot(bin_mids_ho_noZeros, hist_out_noZeros_scaled, 'x')
plt.xlim([0, 2000])
plt.ylim([0, 1.2 * max(hist_out_noZeros_scaled)])
plt.xlabel("Final Number")
plt.title("$n = 2024$ with exponential of log poltfit degree 1") 
plt.ylabel("Density")
plt.grid('on')
plt.plot(bin_mids_ho_noZeros, y_fit)


# finding RMSE for this case
MSE = np.square(np.subtract(hist_out_noZeros_scaled[:,0],y_fit)).mean()
RMSE[2]= np.sqrt(MSE)


#%% 

print(RMSE)

