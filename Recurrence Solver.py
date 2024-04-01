import numpy as np
import matplotlib.pyplot as plt

#%% P(n,k) using recurrence relation and cases

# n is number of strings in bowl
# k is number of loops after n iterations

def P(n,k):
    # P(0,0) = 1
    if n == 0 and k == 0:
        return 1
    # if less than 1 for either and not in above case then not possible. (cannot have negative strings)
    if n < 1 or k < 1:
        return 0
    # if number of loops greater than number of strings not possible.
    if k > n:
        return 0
    # P(1,1) = 1
    if n == 1 and k == 1:
        return 1
    # if not in any of the above cases, return results of recurrence relation
    return P(n-1,k-1)/(2*n-1) + 2*P(n-1,k)*(n-1)/(2*n-1)

#%% Testing of P(n,k)

n_max = 5
k_max = n_max

# matrix with results of calculations in
y_mat = np.ones([n_max+1, k_max+1])*2

# fill matrix with results of P(n,k)
for n in range(n_max+1):
    print(f"n = {n}")
    for k in range(n+1):
        y_mat[n, k] = P(n,k)


# find maximum of each row to normalise the results
y_mat_row_max = np.max(y_mat, axis=1) 
y_mat2 = np.zeros([n_max+1, k_max+1])
for i in range(0, n_max+1):
    y_mat2[i, :] = y_mat[i,:]/y_mat_row_max[i]

#%%

# plot to show probabilities as an image
plt.plot()
plt.imshow(y_mat)
plt.xlabel("$k$ - no. loops in bowl at end")
plt.ylabel("$n$ - no. strings in bowl initially")
plt.title("Probability of getting k loops from n strings, P(n,k)")
plt.show()

# plot to normalise the results
plt.plot()
plt.imshow(y_mat2)
plt.xlabel("$k$ - no. loops in bowl at end")
plt.ylabel("$n$ - no. strings in bowl initially")
plt.title("Probability of getting k loops from n strings, P(n,k) - normalised")
plt.show()