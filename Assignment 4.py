import random

#%% 
def number_path(n):
    current_step = 1
    steps_taken = [1]
    number_flips_1 = 0
    number_flips_2 = 0
    while current_step < n:
        # flip a coin with values 1 and 2 on it
        coin_flip_value = random.randint(1, 2)
        
        # move the number of steps 
        current_step += coin_flip_value
        steps_taken.append(current_step)
        
        # add one to number of flips of the value
        if coin_flip_value == 1:
            number_flips_1 += 1
        else: 
            number_flips_2 += 1
    return steps_taken, number_flips_1, number_flips_2

#%%

n = 25
steps_taken, number_flips_1, number_flips_2 = number_path(n)
        
#%%

n = 25
no_lands_on_n = 0
for i in range(100000):
    steps_taken, number_flips_1, number_flips_2 = number_path(n)
    if steps_taken[-1]==n:
        no_lands_on_n += 1
        