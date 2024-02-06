import pandas as pd
from pandas import read_excel
import pandasql as ps
import requests as rq
import numpy as np
from numpy import inf
import sys
import world_bank_data as wb
import country_converter as cc
import re
from datetime import datetime
import plotly.express as px
from chart_studio import plotly as py
import plotly.offline as py
import plotly.graph_objects as go
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import mpld3
from scipy.spatial.distance import squareform, pdist

pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
#autopep8 --in-place --aggressive --aggressive filename.py

sourceCases="https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
sourceDeaths="https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"
urlDataUS = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/"

## getting some data


def getPopulationDensity(data):
    d = data.copy()
    # https://population.un.org/wpp/Download/Standard/Population/
    pop_DENSITY = pd.read_csv('population-density.csv',
                   sep = '|',
                   engine = 'python')
    pop_DENSITY.columns = ["Country", "PopulationDensity"]
    pop_DENSITY = pd.DataFrame(pop_DENSITY)
    result = pd.merge(d, pop_DENSITY, on='Country', how='left')
    return result
    
def getChinaProvinces():
    ## locally saved text file from https://en.wikipedia.org/wiki/Provinces_of_China

    file = open('ch-provinces.tsv', 'r', encoding="utf8") 
    lines = []
    for i, text in enumerate(file.readlines()):
        if i % 3 == 0:
            line = ''
        line += text.strip()
        if i % 3 == 2:
            lines = lines + [line.split("|||")]
    df = pd.DataFrame.from_records(lines).iloc[:, [1, 2, 4, 5, 6, 7]]
    df.columns = ['ISO', 'Province_Orig', 'Capital', 'Population', 'Density', 'Area']
    df.Population = [int(re.sub(r',|\[8\]', '', p)) for p in df.Population]
    df['Province'] = [ \
        re.sub("Province.*|Municipality.*|Autonomous.*|Zhuang.*|Special.*|Hui|Uyghur", "", s).strip() \
        for s in df['Province_Orig']]
    return df.sort_values('Province')

def getWorldPopulation(data):
    #http://api.worldbank.org/v2/country/all/indicator/SP.POP.TOTL?date=2019&format=jsonstat&per_page=20000
    d = data.copy()
    ## this data is coming from a python package
    # 2020 year data returns null, checking 2019 year data
    pop_GLO = wb.get_series('SP.POP.TOTL', date='2019', id_or_value='id', simplify_index=True)
    
    countries = d['Country'].unique()
    IOS3_codes = cc.convert(list(countries), to='ISO3')
    ISO3_map = dict(zip(countries, IOS3_codes))
    d.insert(4, 'Population', [pop_GLO[c] if c in pop_GLO else 0 for c in [ISO3_map[country] for country in d.Country]])

    
    ## how to get seperate chinese province population data
    pop_CHINA = getChinaProvinces().set_index('Province')['Population']
    ind = (d.Country == 'China') & (d.State != '<all>')
    d.loc[ind, 'Population'] = [pop_CHINA[p] if p in pop_CHINA else 0 for p in d.loc[ind, 'State']]

    return d
 
def getRealTimeData(source, columnName): 
    agg_dict = { columnName:sum, 'Lat':np.median, 'Long':np.median }
    realTimeData = pd.read_csv(source) \
             .rename(columns={ 'Country/Region':'Country', 'Province/State':'State' }) \
             .melt(id_vars=['Country', 'State', 'Lat', 'Long'], var_name='Date', value_name=columnName) \
             .astype({'Date':'datetime64[ns]', columnName:'Int64'}, errors='ignore')
    ## dataset includes each chinese provinc1e separetely. extract all:
    data_CHINA = realTimeData[realTimeData.Country == 'China']
    realTimeData = realTimeData.groupby(['Country', 'Date']).agg(agg_dict).reset_index()
 
    realTimeData['State'] = '<all>'
    return pd.concat([realTimeData, data_CHINA])

def getRealTimeUSData(fileName, columnName, addPopulation=False): 
    id_vars=['Country', 'State', 'Lat', 'Long']
    agg_dict = { columnName:sum, 'Lat':np.median, 'Long':np.median }
    if addPopulation:
        id_vars.append('Population')
        agg_dict['Population'] = sum 
    data = pd.read_csv(urlDataUS + fileName).iloc[:, 6:] \
             .drop('Combined_Key', axis=1) \
             .rename(columns={ 'Country_Region':'Country', 'Province_State':'State', 'Long_':'Long' }) \
             .melt(id_vars=id_vars, var_name='Date', value_name=columnName) \
             .astype({'Date':'datetime64[ns]', columnName:'Int64'}, errors='ignore') \
             .groupby(['Country', 'State', 'Date']).agg(agg_dict).reset_index()
    return data

def allData():
    dataGLOB = getRealTimeData(sourceCases, "CumulativeCases") \
        .merge(getRealTimeData(sourceDeaths, "CumulativeDeaths"))
    dataGLOB = getWorldPopulation(dataGLOB)
    dataGLOB = getPopulationDensity(dataGLOB)
    dataUS = getRealTimeUSData("time_series_covid19_confirmed_US.csv", "CumulativeCases") \
        .merge(getRealTimeUSData("time_series_covid19_deaths_US.csv", "CumulativeDeaths", addPopulation=True))
    data = pd.concat([dataGLOB, dataUS])
    return data

def monthlyDeaths(data1):
    fig = px.line(data1, x=data1.MonthlyPartition, y=data1.MonthlyDeaths, color=data1.Country, title="Death Numbers Monthly")
    fig.show()

def monthlyCases(data1):
    fig2 = px.line(data1, x=data1.MonthlyPartition, y=data1.MonthlyCases, color=data1.Country, title="Death Percentage Per New Case Monthly")
    fig2.show()

#implementing clustering functions
def kMeans4All(data1):
    data = data1
    query2 = "select Country, sum(MonthlyCases) as TotalCases, sum(MonthlyDeaths) as TotalDeaths, Population, PopulationDensity from data group by Country, Population, PopulationDensity"
    data2 = ps.sqldf(query2)
    data2['MortalityRate'] = (data2.TotalDeaths * 100 / data2.TotalCases)
    data2['InfectionRate'] = (data2.TotalCases.fillna(0) * 100 / data2.Population.fillna(0))
    pd.set_option('display.max_rows', data2.shape[0]+1)

    kmeans = KMeans(n_clusters=3, random_state=0)
    
    data2 = pd.DataFrame(data2).fillna(0)
    data2 = data2[~data2.isin([np.nan, np.inf, -np.inf]).any(1)]

    #print(data2)
    kmeans = KMeans(n_clusters=3, random_state=0)
    data2['Cluster'] = kmeans.fit_predict(data2[['MortalityRate', 'InfectionRate']])
    centroids = kmeans.cluster_centers_
    cen_x = [i[0] for i in centroids] 
    cen_y = [i[1] for i in centroids]
    ## add to df
    data2['cen_x'] = data2.Cluster.map({0:cen_x[0], 1:cen_x[1], 2:cen_x[2]})
    data2['cen_y'] = data2.Cluster.map({0:cen_y[0], 1:cen_y[1], 2:cen_y[2]})
    # define and map colors
    colors = ['#DF2020', '#81DF20', '#2095DF']
    data2['c'] = data2.Cluster.map({0:colors[0], 1:colors[1], 2:colors[2]})
    data2.to_csv("results_kmeans", sep=';', columns=['Country', 'Population', 'MortalityRate', 'InfectionRate', 'Cluster'], encoding='utf-8')
    x = data2.MortalityRate
    y = data2.InfectionRate
    c = data2.c
    s = data2.Population/1E6

    fig, ax = plt.subplots()
    ax.grid(color='gray', linestyle='solid')
    scatter = ax.scatter(x, y, c=c, s=s, alpha=0.6)
    labels = ['{0}'.format(c) for c in data2.Country]
    ax.set_title("Cluster Analysis\n", size=20)
    ax.set_xlabel('Mortality Rate (Percent)')
    ax.set_ylabel('Infection Rate (Percent)')

    
    
    tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=labels)
    mpld3.plugins.connect(fig, tooltip)

    mpld3.show()

def kMeans3D(data1):
    data = data1
    query2 = "select Country, sum(MonthlyCases) as TotalCases, sum(MonthlyDeaths) as TotalDeaths, Population, PopulationDensity from data group by Country, Population, PopulationDensity"
    data2 = ps.sqldf(query2)
    data2['MortalityRate'] = (data2.TotalDeaths * 100 / data2.TotalCases)
    data2['InfectionRate'] = (data2.TotalCases.fillna(0) * 100 / data2.Population.fillna(0))
    pd.set_option('display.max_rows', data2.shape[0]+1)

    kmeans = KMeans(n_clusters=4, random_state=0)
    
    data2 = pd.DataFrame(data2).fillna(0)
    data2 = data2[~data2.isin([np.nan, np.inf, -np.inf]).any(1)]
    data2['Cluster'] = kmeans.fit_predict(data2[['MortalityRate', 'InfectionRate','PopulationDensity']])
    colors = ['#DF2020', '#81DF20', '#2095DF','#FF87AD']
    data2['c'] = data2.Cluster.map({0:colors[0], 1:colors[1], 2:colors[2], 3:colors[3]})
    #data2.to_csv("results_kmeans_3d", sep=';', columns=['Country', 'MortalityRate', 'InfectionRate', 'PopulationDensity', 'Cluster'], encoding='utf-8')

    fig = px.scatter_3d(data2, x='MortalityRate', y='InfectionRate', z='PopulationDensity', symbol='Country',
              color='c')
    fig.show()

def main():
    
    data = allData()[['Country', 'State', 'Date', 'Lat', 'Long', 'Population', 'PopulationDensity', 'CumulativeCases', 'CumulativeDeaths']]
    data['MonthlyPartition'] = pd.to_datetime(data['Date']).dt.strftime('%Y-%m')
    query = "SELECT Country, max(CumulativeCases) as MonthlyCases, max(CumulativeDeaths) as MonthlyDeaths, MonthlyPartition, Population, PopulationDensity FROM data where State='<all>' group by Country, MonthlyPartition, Population, PopulationDensity order by Country asc"
    data1 = ps.sqldf(query)
    #pd.set_option('display.max_rows', data1.shape[0]+1)

    #print(data1)

    #monthly cases per month
    #monthlyDeaths(data1)

    #percentage rate of monthly deaths
    #monthlyCases(data1)

    #kMeans4All(data1)
    kMeans3D(data1)

if __name__ == "__main__":
    main()
