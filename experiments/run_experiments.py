import os

kappas=[1., 10.,100., 'inf']
nums_particles=[200, 400, 800, 1600]

for kappa in kappas:
    for num_particles in nums_particles:
            print('"-'*30)
            print("Now Running: k = " + str(kappa)+", n = "+str(num_particles))
            os.system("python3 Ackley_quantitative_nd.py"\
                      +" -k " + str(kappa)\
                      +" -n " + str(num_particles)\
                      +" -r 100 -s 1000 -p -c 50")
            print("Done Running: k = " + str(kappa)+", n = "+str(num_particles))
