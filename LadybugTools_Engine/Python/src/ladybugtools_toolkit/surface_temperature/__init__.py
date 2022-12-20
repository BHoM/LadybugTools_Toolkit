import numpy as np
import pandas as pd


def params_rad(
    sh, RAD, alb, em, sigma, Tskyb, hc, TairK, lambd, ep, Tint, Cv, Tsurf0, ETo, kc
):

    # Evaluate each part of the equation
    # a0= -RAD*(1-alb)
    a0 = -(RAD * 0.8 * sh * (1 - alb) + RAD * 0.2)
    c = (em) * sigma
    a1 = c * -((Tskyb) ** 4)
    a2 = -hc * TairK
    b1 = hc
    b2 = lambd / ep
    a3 = -b2 * Tint
    b3 = Cv * ep / 3600
    a4 = -b3 * (Tsurf0)
    a5 = ETo * kc

    A = a0 + a1 + a2 + a3 + a4 + a5
    B = b1 + b2 + b3
    C = c
    return A, B, C


# ROUGH METHOD

# get shaded proportion for each hour of day (including sy view overnight)
#
