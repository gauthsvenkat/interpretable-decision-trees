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
        if(std1 == 0 and std2 == 0):
            return 0.0
        elif(std1 == 0):
            return norm.pdf(m1,m2,std2)
        else:
            return norm.pdf(m2,m1,std1)
    elif(std1 == std2):
        root = (m1 + m2)/2
        if(norm.pdf(root - 1,m1,std1) < norm.pdf(root-1,m2,std2)):
            return 2 * norm.cdf(root,m1,std1)
        else:
            return 2 * norm.cdf(root,m2,std2)
    else:
        result = solve(m1,m2,std1,std2)
        r1 = result[0]
        r2 = result[1]
        area = 0.0
        if(norm.pdf(r1-1,m1,std1) < norm.pdf(r1-1,m2,std2)):
            area += norm.cdf(r1,m1,std1)
        else:
            area += norm.cdf(r1,m2,std2)
        if(norm.pdf((r1+r2)/2,m1,std1) < norm.pdf((r1+r2)/2,m2,std2)):
            area += norm.cdf(r2,m1,std1) - norm.cdf(r1,m1,std1)
        else:
            area += norm.cdf(r2,m2,std2) - norm.cdf(r1,m2,std2)
        if(norm.pdf(r2+1,m1,std1) < norm.pdf(r2+1,m2,std2)):
            area += 1 - norm.cdf(r2,m1,std1)
        else:
            area += 1 - norm.cdf(r2,m2,std2)
        return area
