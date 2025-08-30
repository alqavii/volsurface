import dis
from math import sqrt
import pandas as pd
import numpy as np
import datetime
from scipy.optimize import brentq
import math

from sympy import ln



class ZeroRatesEngine():
    @staticmethod
    def calcZeroRates(yields: dict) -> list:
        discountRates = {}
        discountRates[0.5] = 100/(100+100*yields[0.5]/2)

        #oneYearDiscount = (100-(100*yields[1.0]/2)*sixMonthDiscount)/(100+(100*yields[1.0]/2))
        #
        #twoYearDiscount = (((-(sqrt(oneYearDiscount)*100*yields[2.0]/2))+
        #                    (sqrt((100*yields[2.0])-4*(100*(1+yields[2.0]/2))*
        #                          (-(100-(sixMonthDiscount*100*yields[2.0]/2)-
        #                             (oneYearDiscount*100*yields[2.0]/2))))))/(2*(100*(1+yields[2.0]/2))))**2
        

        for tenor, rate in yields.items():
            if tenor == 0.5:
                pass

            prev = max(discountRates.keys())
            c = 100*yields[tenor]/2
            forwardRate = brentq(lambda x: c*(sum(a for a in discountRates.values())) + c*discountRates[prev]*(sum(math.exp(x*-a/2) for a in range(1, (tenor-prev)*2))) + (100+c)*math.exp(x*-((tenor-prev))) , 
            -0.1,1.0)

            discountRates.update({
                prev + k*0.5: discountRates[prev]*math.exp(-forwardRate[0]*(k*0.5))
                for k in range(1, int((tenor-prev)*2)+1)
            })

        zeroRates = [(-1/tenor)*ln(df) for tenor,df in discountRates.items()]

        return zeroRates
    

