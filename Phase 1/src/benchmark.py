from pathlib import Path
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import time 

# Import all methods 
from option import Option
from binomial import crr_price
from monteCarlo import mc_naive, mc_antithetic, mc_control, mc_control_antithetic
from monteCarloLSM import LSMoptionValue
from pde import pde_crank_nicolson
from BSpricer import BSModel

DATA = Path(__file__).parent.parent / "data"
REPORTS = Path(__file__).parent.parent / "reports"

_bs = BSModel()

PRICERS = {
    "BS":            lambda opt, style: _bs.price(opt) if style != 'American' else None,
    "crr":           lambda opt, style: crr_price(opt, period=1000, american=(style == 'American')),
    "MC Naive":      lambda opt, _: mc_naive(opt, n_paths=100000)[0],
    "MC Anti":       lambda opt, _: mc_antithetic(opt, n_paths=100000)[0],
    "MC Contr":      lambda opt, _: mc_control(opt, n_paths=100000)[0],
    "MC Anti Contr": lambda opt, _: mc_control_antithetic(opt, n_paths=100000)[0],
    "lsm":           lambda opt, style: LSMoptionValue(opt, n_steps=50, n_paths=20000) if style == 'American' else None,
    "pde":           lambda opt, style: pde_crank_nicolson(opt, style=style.lower(), n_steps=200, n_space=200),
    "BS,pde":        lambda opt,style: _bs.price(opt) if style != 'American' else pde_crank_nicolson(opt, style=style.lower(), n_steps=200, n_space=200),
    "BS,anti":       lambda opt, style: _bs.price(opt) if style != 'American' else mc_naive(opt, n_paths=100000)[0]
}

def calcPriceWithMethod(cleanData: pd.DataFrame, method: str):
    pricer = PRICERS[method]
    time_start = time.time()
    for row in cleanData.itertuples():
        option = Option(S=row[2], K=row[3], T=row[4], r=row[5]/100, sigma=row[6]/100, kind=row[1].lower())
        pricer(option, row[7])
    return time.time() - time_start


if __name__ == "__main__" : 
    
    ####### Extraction of data from csv file and calculation of option prices using different methods #######
    start_time = time.time()
    data = pd.read_excel(DATA / "options_benchmark.xlsx", sheet_name=0, header=2,index_col=0)
    end_time = time.time()
    print(f"Time taken to read data: {end_time - start_time:.2f} seconds")

    start_time = time.time()
    cleanData = data[['Kind','Spot  S', 'Strike  K', 'T  (years)', 'Rate  r (%)', 'Vol  σ (%)','Style']]
    end_time = time.time()
    print(f"Time taken to process data: {end_time - start_time:.2f} seconds")

    methods = ["crr","BS", "MC Naive", "MC Anti", "MC Contr", "MC Anti Contr", "lsm", "pde", "BS,pde", "BS,anti"]
    times = []
    for method in methods :
        if method == "BS" :
            times.append(calcPriceWithMethod(cleanData,method)) 
            print(f"Time taken to price data with {method} method ", times[-1], "for 1100 options not 2000")
        else : 
            times.append(calcPriceWithMethod(cleanData,method)) 
            print(f"Time taken to price data with {method} method ", times[-1] )
    
    # Graph of the time taken
    plt.figure(figsize=(12, 5))
    plt.bar(methods, times, color="steelblue", edgecolor="white")
    plt.xlabel("Method")
    plt.ylabel("Total time (s)")
    plt.title("Time to price ~2000 options per method")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(REPORTS / "Time Taken to price.png")
    plt.show()