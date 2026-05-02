
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import cvxpy as cp

#from IPython.display import display


# load data for Elspotprices file
def LoadData_pr():

  import os
  import pandas as pd

  # Load electricity prices
  price_path = os.path.join(os.getcwd(), 'ElspotpricesEA.csv')
  df_prices = pd.read_csv(price_path)

  # Convert to datetime
  df_prices["HourDK"] = pd.to_datetime(df_prices["HourDK"])

  # Convert prices to DKK/kWh
  df_prices['SpotPriceDKK'] = df_prices['SpotPriceDKK'] / 1000

  # Filter only DK2 prices
  df_prices = df_prices.loc[df_prices['PriceArea'] == "DK2"]

  # Keep only the local time and price columns
  df_prices = df_prices[['HourDK', 'SpotPriceDKK']]

  # Reset the index
  df_prices = df_prices.reset_index(drop=True)

  return df_prices


# Function to load electricity and prosumer data
def LoadData():

  import os
  import pandas as pd

  ### Load electricity prices ###
  price_path = os.path.join(os.getcwd(), 'ElspotpricesEA.csv')
  df_prices = pd.read_csv(price_path)

  # Convert to datetime
  df_prices["HourDK"] = pd.to_datetime(df_prices["HourDK"])

  # Convert prices to DKK/kWh
  df_prices['SpotPriceDKK'] = df_prices['SpotPriceDKK'] / 1000

  # Filter only DK2 prices
  df_prices = df_prices.loc[df_prices['PriceArea'] == "DK2"]

  # Keep only the local time and price columns
  df_prices = df_prices[['HourDK', 'SpotPriceDKK']]

  # Keep only 2022 and 2023
  df_prices = df_prices.loc[df_prices["HourDK"].dt.year.isin([2022, 2023])]

  # Reset the index
  df_prices = df_prices.reset_index(drop=True)

  ###  Load prosumer data ###
  file_P = os.path.join(os.getcwd(), 'ProsumerHourly.csv')
  df_pro = pd.read_csv(file_P)
  df_pro["TimeDK"] = pd.to_datetime(df_pro["TimeDK"])
  df_pro = df_pro.reset_index(drop=True)
  df_pro.rename(columns={'Consumption': 'Load'}, inplace=True)
  df_pro.rename(columns={'TimeDK': 'HourDK'}, inplace=True)

  return df_prices, df_pro


# Function to calculate the hourly buy and sell prices
def PricesDK(df_prices):

  # Set the Sell price equal to the spot price
  df_prices["Sell"] = df_prices["SpotPriceDKK"]

  # Define the fixed Tax and TSO columns
  df_prices["Tax"] = 0.8
  df_prices["TSO"] = 0.1

  ### Add the DSO tariffs

  # The Low period has the same price during both summer/winter periods
  df_prices.loc[df_prices["HourDK"].dt.hour.isin([0, 1, 2, 3, 4, 5]),
                "DSO"] = 0.15

  # Peak period in Winter
  df_prices.loc[(df_prices["HourDK"].dt.month.isin([1, 2, 3, 10, 11, 12]))
                & (df_prices["HourDK"].dt.hour.isin([17, 18, 19, 20])),
                "DSO"] = 1.35

  # Peak period in Summer
  df_prices.loc[(df_prices["HourDK"].dt.month.isin([4, 5, 6, 7, 8, 9]))
                & (df_prices["HourDK"].dt.hour.isin([17, 18, 19, 20])),
                "DSO"] = 0.6

  # High period in Winter
  df_prices.loc[(df_prices["HourDK"].dt.month.isin([1, 2, 3, 10, 11, 12]))
                & (df_prices["HourDK"].dt.hour.
                   isin([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 21, 22, 23])),
                "DSO"] = 0.45

  # High period in Summer
  df_prices.loc[(df_prices["HourDK"].dt.month.isin([4, 5, 6, 7, 8, 9]))
                & (df_prices["HourDK"].dt.hour.
                   isin([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 21, 22, 23])),
                "DSO"] = 0.23

  # Calculate VAT
  df_prices["VAT"] = 0.25 * (df_prices["Tax"] + df_prices["TSO"] +
                             df_prices["DSO"] + df_prices["SpotPriceDKK"])

  # Calculate Buy price
  df_prices["Buy"] = df_prices["Tax"] + df_prices["TSO"] + df_prices[
      "DSO"] + df_prices["SpotPriceDKK"] + df_prices["VAT"]

  return df_prices


# Function to calculate the annual cost and profit of energy based on an hourly netting
def HrNet(df_pro):
  # Calculate the energy consumption for days Load is greater then generation
  df_pro['Imports'] = ((df_pro['Load'] - df_pro['PV'])).where(df_pro['Load']
                                                              > df_pro['PV'],
                                                              other='0')
  df_pro['Imports'] = df_pro['Imports'].astype(float)

  # Calculate Cost
  df_pro['Cost'] = df_pro['Imports'] * df_pro['Buy']

  # Calculate the energy sent to the grid
  df_pro['Exports'] = ((df_pro['PV'] - df_pro['Load'])).where(df_pro['PV']
                                                              > df_pro['Load'],
                                                              other='0')
  df_pro['Exports'] = df_pro['Exports'].astype(float)

  # Calculate Profit
  df_pro['Profit'] = df_pro['Exports'] * df_pro['Sell']

  # Return data as yearly aggregation
  df_hrnet = df_pro.groupby('Year').agg({
      'Cost': 'sum',
      'Profit': 'sum'
  }).reset_index()

  return df_hrnet


# Function to calculate the anual cost and profit of energy assuming there is no netting
def NoNet(df_pro):

  import pandas as pd

  # Calculate the profits and costs of the PV and load assuming no netting occurs
  df_nn = pd.DataFrame()
  df_nn['Year'] = df_pro['Year']
  df_nn['Cost'] = df_pro['Buy'] * df_pro['Load']
  df_nn['Profit'] = df_pro['Sell'] * df_pro['PV']

  # Aggregate the data to a yearly basis
  df_cost = df_nn.groupby('Year').agg({'Cost': 'sum'}).reset_index()
  df_profit = df_nn.groupby('Year').agg({'Profit': 'sum'}).reset_index()

  df_nonet = pd.merge(df_cost[['Year', 'Cost']],
                      df_profit[['Year', 'Profit']],
                      on='Year')

  return df_nonet


def Optimizer(params, prices):

    n = len(prices)

    p_c = cp.Variable(n)
    p_d = cp.Variable(n)
    X = cp.Variable(n)
    profit = cp.sum(p_d @ prices - p_c @ prices)

    constraints = [
        p_c >= 0, p_d >= 0, p_c <= params['Pmax'], p_d <= params['Pmax']
    ]
    constraints += [X >= params['Cmin'], X <= params['Cmax']]
    constraints += [
        X[0] == params['C_0'] + p_c[0] * params['n_c'] - p_d[0] / params['n_d']
    ]

    constraints += [
        X[1:] == X[:-1] + p_c[1:] * params['n_c'] - p_d[1:] / params['n_d']
    ]

    constraints += [X[n - 1] == params['C_n']]

    problem = cp.Problem(cp.Maximize(profit), constraints)
    problem.solve(solver=cp.ECOS)

    return profit.value, p_c.value, p_d.value, X.value


def ProsumerOptimizer(params, l_b, l_s, p_PV, p_L):

  import cvxpy as cp

  n = len(l_b)
  p_c = cp.Variable(n)
  p_d = cp.Variable(n)
  p_b = cp.Variable(n)
  p_s = cp.Variable(n)
  X = cp.Variable(n)
  cost = cp.sum(p_b @ l_b - p_s @ l_s)

  constraints = [
      p_c >= 0, p_d >= 0, p_c <= params['Pmax'], p_d <= params['Pmax'], p_s
      >= 0, p_b >= 0
  ]
  constraints += [X >= 0, X <= params['Cmax']]
  constraints += [
      X[0] == params['C_0'] + p_c[0] * params['n_c'] - p_d[0] / params['n_d']
  ]
  constraints += [p_PV + p_b + p_d == p_L + p_s + p_c]

  constraints += [
      X[1:] == X[:-1] + p_c[1:] * params['n_c'] - p_d[1:] / params['n_d']
  ]

  constraints += [X[n - 1] == params['C_n']]

  problem = cp.Problem(cp.Minimize(cost), constraints)
  problem.solve(solver=cp.ECOS)

  return cost.value, p_c.value, p_d.value, p_b.value, p_s.value, X.value
