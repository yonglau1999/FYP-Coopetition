import numpy as np
from Logistics_Service_Model import LogisticsServiceModel

def stackelberg_game(L_s, theta, f):
    """
    Function to compute the optimal actions for the seller and TPLP.
    
    Parameters:
        f (float): Price factor set by the TPLP (0 <= f <= 10).
        L_s (float): Service level set by the TPLP.
        theta (float): Market potential.

    Returns:
        dict: A dictionary containing the seller's action and TPLP's action.
    """
    # Initialize the logistics service model
    model = LogisticsServiceModel(L_s, theta, f)

    profit_et_no_sharing = model.profit_nosharing_etailer() 
    profit_et_sharing = model.profit_sharing_etailer(True)
    profit_seller_no_sharing = model.profit_nosharing_seller()
    profit_seller_sharing = model.profit_sharing_seller(True)
    profit_tplp_no_sharing = model.profit_nosharing_tplp()


    # Calculate profit differences directly in the loop
    profit_diff_et = profit_et_sharing - profit_et_no_sharing
    profit_diff_seller = profit_seller_sharing - profit_seller_no_sharing

    if profit_diff_et <0:
        return 0,0
    else:
        return 1,1  
    