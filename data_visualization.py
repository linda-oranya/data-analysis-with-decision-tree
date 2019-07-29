#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
import seaborn as sns


df=pd.read_csv("gapminder-FiveYearData.csv")
df.head(5)


df2=df.pivot_table('lifeExp','continent','year')


fig=sns.heatmap(df2).get_figure()


fig.savefig('heatmapfigure')





