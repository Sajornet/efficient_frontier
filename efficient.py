import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
import scipy.optimize as optimize

yf.pdr_override() #Fixing data read Yahoo
assets = ["IVVPESOISHRS.MX", "NAFTRACISHRS.MX", "CETETRCISHRS.MX"]
start_date = "2015-08-06"
end_date = "2018-08-06"

df=pd.DataFrame()
for asset in assets:
    df_asset = pdr.get_data_yahoo(asset, start=start_date, end=end_date)["Adj Close"]
    df_asset = df_asset.to_frame(name=asset)
    df = pd.concat([df_asset, df], axis=1, sort=False)
df = df.dropna()
df.head(5)

df = df.pct_change().dropna()
#Creamos matriz de pesos para cada activo ("Equally weighted")
weights = []
n_assets = len(assets)

#Equally weighted
for i in range(n_assets):
    weights.append(1/n_assets)

w = np.array(weights)

r = np.array(np.mean(df))
C = np.cov(df.transpose())

#Validamos
print("Rendimiento esperado:", r)
print("Pesos activos:", w)
print("Matriz VarCov:", C)

def mu(w,r):
    '''Portfolio performance annualized'''
    return sum(w * r * 252) 


def sigma(w, C):
    '''STD annualized'''
    return np.dot(w,np.dot(C,w.T)) ** (1/2) * 252 ** (1/2)


def sharpe(w):
    '''Sharpe ratio with rf of 2%'''
    rf = .02
    return (mu(w,r) - rf) / sigma(w,C)


def neg_sharpe(w):
    '''Sharpe ratio negativo'''
    return -sharpe(w)


def random_ports(n):
    '''Portafolio aleatorios'''
    means, stds = [],[]
    for i in range(n):
        rand_w = np.random.rand(len(assets))
        rand_w = rand_w / sum(rand_w)
        means.append(mu(rand_w, r))
        stds.append(sigma(rand_w,C))
    
    return means, stds

print("Sharpe port equal w:", round(sharpe(w),2))

def apply_sum_constraint(inputs):
    total = 1 - np.sum(inputs)
    return total

my_constraints = ({'type': 'eq', "fun": apply_sum_constraint })


result = optimize.minimize(neg_sharpe, 
                      w, 
                      method='SLSQP', 
                      bounds=((0, 1.0), (0, 1.0), (0, 1.0)),
                      options={'disp': True},
                      constraints=my_constraints)
print(result)
optimal_w = result["x"]

#Chart
n_portfolios = 10000
means, stds = random_ports(n_portfolios)

best_mu = mu(optimal_w, r)
best_sigma = sigma(optimal_w, C)
best_sharpe = sharpe(optimal_w)
plt.plot(stds, means, 'o', markersize=1)
plt.plot(best_sigma, best_mu, 'x',  markersize=10)
plt.xlabel('std')
plt.ylabel('mean')
plt.title('Mean and standard deviation of returns of randomly generated portfolios')
