import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
"""usage: pass the two means and standard deviations of the two distributions (normal). This finds the overlapping area to return the degree of match"""

#Get point of intersect
def solve(m1,m2,std1,std2):
  a = 1/(2*std1**2) - 1/(2*std2**2)
  b = m2/(std2**2) - m1/(std1**2)
  c = m1**2 /(2*std1**2) - m2**2 / (2*std2**2) - np.log(std2/std1)
  return np.roots([a,b,c])


def getMatchDegree(m1,m2,std1,std2):
    if(std1==0 or std2==0):
        if(m1 == m2):
            return 1.0 #kind of like Dirac delta case
        else:
            return 0.0
    result = solve(m1,m2,std1,std2)
    r1 = result[0]
    r2 = result[1]
    area = 0.0
    if(r1 == r2): #only one root
        if(m2 > m1):
            area = norm.cdf(r1,m2,std2) + 1 - norm.cdf(r1,m1,std1)
        else:
            area = norm.cdf(r1,m1,std1) + 1 - norm.cdf(r1,m2,std2)
    else:
        if(r1 <= m1 and m1 <= r2 and r1 <= m2 and m2 <= r2):
            if(norm.pdf(m1,m1,std1) < norm.pdf(m2,m2,std2)):
                area = norm.cdf(r1,m2,std2) + 1 - norm.cdf(r2,m2,std2) + norm.cdf(r2,m1,std1) - norm.cdf(r1,m1,std1)
            else:
                area = norm.cdf(r1,m1,std1) + 1 - norm.cdf(r2,m1,std1) + norm.cdf(r2,m2,std2) - norm.cdf(r1,m2,std2)
        elif(m1 <= r1 or m1 >= r2):
            area = norm.cdf(r1,m2,std2) + norm.cdf(r2,m1,std1) - norm.cdf(r1,m1,std1) + 1- norm.cdf(r2,m2,std2)
        elif(m2 <= r1 or m2 >= r2):
            area = norm.cdf(r1,m1,std1) + norm.cdf(r2,m2,std2) - norm.cdf(r1,m2,std2) + 1- norm.cdf(r2,m1,std1)



    # integrate
    return area
