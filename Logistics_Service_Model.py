import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import os


class LogisticsServiceModel:
    def __init__(self, L_s,theta,f=1,L_e=10, phi=0.05, alpha=0.5, beta=0.7, gamma=0.5, c=0.5):
        self.L_e = L_e  # E-tailer's logistics service level    
        self.L_s = L_s  # Seller's logistics service level
        self.phi = phi  # Commission rate
        self.theta = theta      # Market potential (from normal distribution)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.c = c      # Variable cost of logistics for E-tailer
        self.f = f      # Seller's third-party logistics cost
        self.max_capacity = 5 # E-tailer's maximum service capacity

    def M1(self):
        return (1 - self.phi) * (self.theta * (2 + self.alpha * (1 + self.phi)) + 2 * self.c +
                                 self.L_e * (2 * self.beta - self.alpha * self.gamma * (1 + self.phi)) +
                                 self.L_s * (self.alpha * self.beta * (1 + self.phi) - 2 * self.gamma)) + self.alpha * self.f * (1 + self.phi)

    def M2(self):
        return (1 - self.phi) * (self.theta * (2 + self.alpha) + self.alpha * self.c +
                                 self.L_e * (self.alpha * self.beta - 2 * self.gamma) +
                                 self.L_s * (2 * self.beta - self.alpha * self.gamma)) + 2 * self.f

    def N1(self):
        return self.theta * (2 + self.alpha * (1 - self.phi) - self.alpha**2 * self.phi) + \
               self.L_e * (self.beta * (2 - self.alpha**2 * self.phi) - self.alpha * self.gamma * (1 - self.phi)) + \
               self.L_s * (self.alpha * self.beta * (1 - self.phi) + self.gamma * (self.alpha**2 * self.phi - 2)) - \
               self.c * (2 - self.alpha**2) + self.alpha * self.f

    def N2(self):
        return (1 - self.phi) * (self.theta * (2 + self.alpha) + self.alpha * self.c +
                                 self.L_e * (self.alpha * self.beta - 2 * self.gamma) +
                                 self.L_s * (2 * self.beta - self.alpha * self.gamma)) - self.f * (2-self.alpha**2 * (1 + self.phi))
    
    # Optimal prices for sharing/no sharing

    def p1_sharing(self,ww):
        top = (
            -self.L_e * self.alpha * self.beta * self.phi**2 +
            self.L_e * self.alpha * self.beta +
            self.L_e * self.alpha * self.gamma * self.phi**2 -
            self.L_e * self.alpha * self.gamma -
            2 * self.L_e * self.beta * self.phi +
            2 * self.L_e * self.beta +
            2 * self.L_e * self.gamma * self.phi -
            2 * self.L_e * self.gamma -
            self.theta * self.alpha * self.phi**2 +
            self.theta * self.alpha -
            2 * self.theta * self.phi +
            2 * self.theta +
            2 * self.alpha * self.c * self.phi -
            2 * self.alpha * self.c -
            self.alpha * self.phi * self.calc_w(ww) +
            3 * self.alpha * self.calc_w(ww) -
            2 * self.c * self.phi +
            2 * self.c
        )
        bottom = self.alpha**2 * self.phi**2 - self.alpha**2 - 4 * self.phi + 4

        return top/bottom
    
    def p2_sharing(self,ww):
        top = (
            -self.L_e * self.alpha * self.beta * self.phi +
            self.L_e * self.alpha * self.beta +
            self.L_e * self.alpha * self.gamma * self.phi -
            self.L_e * self.alpha * self.gamma -
            2 * self.L_e * self.beta * self.phi +
            2 * self.L_e * self.beta +
            2 * self.L_e * self.gamma * self.phi -
            2 * self.L_e * self.gamma -
            self.theta * self.alpha * self.phi +
            self.theta * self.alpha -
            2 * self.theta * self.phi +
            2 * self.theta +
            self.alpha**2 * self.c * self.phi -
            self.alpha**2 * self.c -
            self.alpha**2 * self.phi * self.calc_w(ww) +
            self.alpha**2 * self.calc_w(ww) -
            self.alpha * self.c * self.phi +
            self.alpha * self.c +
            2 * self.calc_w(ww)
        )

        bottom = self.alpha**2 * self.phi**2 - self.alpha**2 - 4 * self.phi + 4
        return top/bottom  
    
    def p1_nosharing(self):
        return self.M1()/((1-self.phi)*(4-self.alpha**2*(1+self.phi)))
    
    def p2_nosharing(self):
        return self.M2()/((1-self.phi)*(4-self.alpha**2*(1+self.phi)))  
    
    # Demand for sharing and no sharing
    def D_sharing_etailer(self,ww):
        return self.theta - self.p1_sharing(ww) +self.alpha * self.p2_sharing(ww) + self.beta * self.L_e - self.gamma * self.L_e
    
    def D_nosharing_etailer(self):
         return self.theta - self.p1_nosharing() +self.alpha * self.p2_nosharing() + self.beta * self.L_e - self.gamma * self.L_s        
    
    def D_sharing_seller(self,ww):
        return self.theta - self.p2_sharing(ww) +self.alpha * self.p1_sharing(ww) + self.beta * self.L_e - self.gamma * self.L_e
    
    def D_nosharing_seller(self):
         return self.theta - self.p2_nosharing() +self.alpha * self.p1_nosharing() + self.beta * self.L_s - self.gamma * self.L_e       
    
    def calc_excess_demand(self, ww): # Unfulfilled capacity

        """Calculate excess demand for TPLP."""
        demand_sharing_etailer = self.D_sharing_etailer(ww)
        demand_sharing_seller = self.D_sharing_seller(ww)
        total_demand = demand_sharing_etailer + demand_sharing_seller
        
        # Check if the total demand exceeds the maximum capacity
        if total_demand > self.max_capacity:
            excess = total_demand - self.max_capacity
        else:
            excess = 0
            
        return excess

    # Profit functions sharing and no sharing

    def profit_nosharing_etailer(self):
        M1 = self.M1()
        M2 = self.M2()
        N1 = self.N1()
        N2 = self.N2()
        bottom = (1 - self.phi) * (4 - self.alpha**2 * (1 + self.phi))**2
        return max(((N1 * (M1 - self.c * (1 - self.phi) * (4 - self.alpha**2 * (1 + self.phi))) +
                self.phi * M2 * N2 / (1 - self.phi)) / bottom),0)
    
    def profit_nosharing_seller(self):
        N2 = self.N2()
        bottom = (1 - self.phi) * (4 - self.alpha**2 * (1 + self.phi))**2
        return max(((N2**2) / bottom),0)
    
    def profit_nosharing_tplp(self):
        a,b,c = 0.01,-0.12,0.1
        cost_per_unit = a * self.L_s**2 + b * self.L_s + c # Convex cost function of service level
        total_volume = self.D_nosharing_seller()
        return max((self.f-cost_per_unit)*total_volume,0)


    def profit_sharing_etailer(self,ww):
        return max((self.p1_sharing(ww)-self.c)*(self.theta-self.p1_sharing(ww)+self.alpha*self.p2_sharing(ww)+(self.beta-self.gamma)*self.L_e) + \
        (self.phi*self.p2_sharing(ww)+self.calc_w(ww)-self.c)*(self.theta-self.p2_sharing(ww)+self.alpha*self.p1_sharing(ww)+(self.beta-self.gamma)*self.L_e),0)
    

    def profit_sharing_seller(self,ww):
        if ww == True:

            return max(((1-self.phi)*self.p2_sharing(ww)-self.calc_w(ww))*(self.theta-self.p2_sharing(ww)+self.alpha*self.p1_sharing(ww)+(self.beta-self.gamma)*self.L_e),0)
        else: # If w=wstarwstar, add 1e-2 to profits to incentivize RL to pick sharing mode instead
            return max(((1-self.phi)*self.p2_sharing(ww)-self.calc_w(ww))*(self.theta-self.p2_sharing(ww)+self.alpha*self.p1_sharing(ww)+(self.beta-self.gamma)*self.L_e)+1e-2,0)
    
    def profit_sharing_tplp(self, ww):
        a,b,c = 0.01,-0.12,0.1
        excess_demand = self.calc_excess_demand(ww)
        cost_per_unit = a * self.L_s**2 + b * self.L_s + c  # Convex cost function of service level
        return max((self.f - cost_per_unit) * excess_demand,0)

    def calc_w(self,ww):
        if ww == True:
            result = (
                ((self.phi - 1) * (
                    8 * self.c + 8 * self.theta + 8 * self.L_e * self.beta - 8 * self.L_e * self.gamma -
                    8 * self.alpha * self.c - 8 * self.phi * self.theta + 2 * self.alpha**2 * self.c -
                    3 * self.alpha**3 * self.c + self.alpha**4 * self.c + self.alpha**3 * self.theta -
                    2 * self.alpha**2 * self.phi * self.theta + self.alpha**3 * self.c * self.phi**2 -
                    self.alpha**4 * self.c * self.phi**2 + 2 * self.alpha**2 * self.phi**2 * self.theta -
                    self.alpha**3 * self.phi**2 * self.theta - 8 * self.L_e * self.beta * self.phi +
                    8 * self.L_e * self.gamma * self.phi - 4 * self.alpha * self.c * self.phi +
                    4 * self.alpha * self.phi * self.theta + self.L_e * self.alpha**3 * self.beta -
                    self.L_e * self.alpha**3 * self.gamma + 2 * self.alpha**2 * self.c * self.phi +
                    2 * self.alpha**3 * self.c * self.phi - 2 * self.L_e * self.alpha**2 * self.beta * self.phi +
                    2 * self.L_e * self.alpha**2 * self.gamma * self.phi +
                    2 * self.L_e * self.alpha**2 * self.beta * self.phi**2 -
                    self.L_e * self.alpha**3 * self.beta * self.phi**2 -
                    2 * self.L_e * self.alpha**2 * self.gamma * self.phi**2 +
                    self.L_e * self.alpha**3 * self.gamma * self.phi**2 +
                    4 * self.L_e * self.alpha * self.beta * self.phi -
                    4 * self.L_e * self.alpha * self.gamma * self.phi
                )) / (
                    2 * (self.alpha**3 * self.phi**2 - 2 * self.alpha**3 * self.phi +
                        self.alpha**3 - self.alpha**2 * self.phi**2 + 2 * self.alpha**2 * self.phi -
                        self.alpha**2 - 4 * self.alpha * self.phi + 8 * self.alpha +
                        4 * self.phi - 8)
                )
            )

            return result
        
        elif ww == False:
            result = (
                (2 * self.L_s * self.beta - 2 * self.L_e * self.beta - 2 * self.f + self.alpha**2 * self.c +
                self.alpha**2 * self.f + self.L_e * self.alpha * self.gamma - self.L_s * self.alpha * self.gamma +
                2 * self.L_e * self.beta * self.phi - 2 * self.L_s * self.beta * self.phi - self.alpha**2 * self.c * self.phi +
                self.alpha**2 * self.f * self.phi - self.L_e * self.alpha * self.gamma * self.phi +
                self.L_s * self.alpha * self.gamma * self.phi)
                / (2 * (self.alpha**2 - 1))
            )

            return result



