#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import numpy as np


# In[3]:


data = pd.Series([0.25, 0.5, 0.75, 1.0])
data


# In[4]:


data.values


# In[5]:


data.index


# In[6]:


data[1]


# In[7]:


data[1:3]


# In[8]:


data = pd.Series([0.25, 0.5, 0.75, 1.0],index=['a', 'b', 'c', 'd'])
data


# In[9]:


data['b']


# In[11]:


data = pd.Series([0.25, 0.5, 0.75, 1.0],index=[2, 5, 3, 7])
data


# In[22]:


population_dict = {'California': 38332521,'Texas': 26448193,'New York': 19651127,'Florida': 19552860,'Illinois': 12882135}
population = pd.Series(population_dict)
population


# In[17]:


population['Texas']


# In[23]:


population['California':'Florida']


# In[24]:


pd.Series({2:'a', 1:'b', 3:'c'}, index=[3, 2])


# In[25]:


area_dict = {'California': 423967, 'Texas': 695662, 'New York': 141297,'Florida': 170312, 'Illinois': 149995}
area = pd.Series(area_dict)
area


# In[26]:


states = pd.DataFrame({'population': population,'area': area})
states


# In[27]:


states.index


# In[28]:


states.columns


# In[30]:


data = [{'a': i, 'b': 2 * i}for i in range(3)]
pd.DataFrame(data)


# In[31]:


pd.DataFrame([{'a': 1, 'b': 2}, {'b': 3, 'c': 4}])


# In[32]:


pd.DataFrame({'population': population,'area': area})


# In[44]:


pd.DataFrame(np.random.rand(3,2),columns=['foo','bar'],index=['a','b','c'])


# In[45]:


A = np.zeros(3, dtype=[('A', 'i8'), ('B', 'f8')])
A


# In[46]:


pd.DataFrame(A)


# In[48]:


ind = pd.Index([2, 3, 5, 7, 11])
ind


# In[49]:


print(ind.size, ind.shape, ind.ndim, ind.dtype)


# In[50]:


ind[1] = 0 #immutable in pandas


# In[51]:


indA = pd.Index([1, 3, 5, 7, 9])
indB = pd.Index([2, 3, 5, 7, 11])


# In[52]:


indA & indB # intersection


# In[53]:


indA | indB # union


# In[54]:


indA ^ indB # symmetric difference


# In[55]:


indA.intersection(indB)


# In[56]:


data = pd.Series([0.25, 0.5, 0.75, 1.0],index=['a', 'b', 'c', 'd'])
data


# In[57]:


list(data.items())


# In[58]:


data['e'] = 1.25
data


# In[60]:


# slicing by explicit index
data['a':'c']


# In[61]:


# slicing by implicit integer index
data[0:2]


# In[62]:


# masking
data[(data > 0.3) & (data < 0.8)]


# In[63]:


# fancy indexing
data[['a', 'e']]


# In[64]:


data[0:2]


# In[65]:


data = pd.Series(['a', 'b', 'c'], index=[1, 3, 5])
data


# In[66]:


# explicit index when indexing
data[1]


# In[67]:


# implicit index when slicing
data[1:3]


# In[68]:


data.loc[1]


# In[69]:


data.loc[1:3]


# In[70]:


data.iloc[1]


# In[71]:


data.iloc[0]


# In[72]:


data.iloc[1:3]


# In[73]:


area = pd.Series({'California': 423967, 'Texas': 695662,'New York': 141297, 'Florida': 170312,'Illinois': 149995})
pop = pd.Series({'California': 38332521, 'Texas': 26448193,'New York': 19651127, 'Florida': 19552860,'Illinois': 12882135})
data = pd.DataFrame({'area':area, 'pop':pop})
data


# In[74]:


data['area']


# In[75]:


data.area


# In[76]:


data['density'] = data['pop'] / data['area']
data


# In[77]:


data.values


# In[78]:


data.T


# In[79]:


data.iloc[:3, :2]


# In[80]:


data.loc[:'Illinois', :'pop']


# In[82]:


data.loc[data.density > 100, ['pop', 'density']]


# In[84]:


data.iloc[0, 2] = 90
data


# In[85]:


data[data.density > 100]


# In[6]:


import pandas as pd
import numpy as np
rng = np.random.RandomState(42)
ser = pd.Series(rng.randint(0, 10, 4))
df = pd.DataFrame(ser)
df


# In[8]:


df = pd.DataFrame(rng.randint(0, 10, (3, 4)),columns=['A', 'B', 'C', 'D'])
df


# In[9]:


np.sin(df * np.pi / 4)


# In[11]:


area = pd.Series({'Alaska': 1723337, 'Texas': 695662,'California': 423967}, name='area')
population = pd.Series({'California': 38332521, 'Texas': 26448193,'New York': 19651127}, name='population')


# In[14]:


population / area


# In[15]:


area.index | population.index


# In[16]:


area.index & population.index


# In[20]:


A = pd.Series([2, 4, 6], index=[0, 1, 2])
B = pd.Series([1, 3, 5], index=[1, 2, 3])
c = A + B
d = pd.DataFrame(c)
d


# In[21]:


A.add(B, fill_value=0)


# In[23]:


A = pd.DataFrame(rng.randint(0, 20, (2, 2)),columns=list('AB'))
A


# In[27]:


B = pd.DataFrame(rng.randint(0, 10, (3, 3)),columns=list('BAC'))
B


# In[28]:


A + B


# In[29]:


fill = A.stack().mean()
A.add(B, fill_value=fill)


# In[30]:


A = rng.randint(10, size=(3, 4))
A


# In[31]:


A-A[0]


# In[35]:


df = pd.DataFrame(A, columns=list('QRST'))
df - df.iloc[0]


# In[36]:


df.subtract(df['R'], axis=0)


# In[37]:


halfrow = df.iloc[0, ::2]
halfrow


# In[38]:


vals1 = np.array([1, None, 3, 4])
vals1


# In[42]:


for dtype in ['object', 'int']:
    print("dtype =", dtype)
    get_ipython().run_line_magic('timeit', 'np.arange(1E6, dtype=dtype).sum()')
    print()


# In[43]:


vals2 = np.array([1, np.nan, 3, 4])
vals2.dtype


# In[44]:


1 + np.nan


# In[45]:


0*np.nan


# In[46]:


vals2.sum(), vals2.min(), vals2.max()


# In[47]:


np.nansum(vals2), np.nanmin(vals2), np.nanmax(vals2)


# In[48]:


pd.Series([1, np.nan, 2, None])


# In[49]:


x = pd.Series(range(2), dtype=int)
x


# In[50]:


x[0] = None
x


# In[51]:


data = pd.Series([1, np.nan, 'hello', None])
data.isnull()   #detecting null


# In[52]:


data[data.notnull()]


# In[53]:


data.dropna()


# In[55]:


df = pd.DataFrame([[1, np.nan, 2],[2, 3, 5],[np.nan, 4, 6]])
df


# In[56]:


df.dropna()   # dropna() will drop all rows in which any null value is present


# In[59]:


df.dropna(axis='columns')


# In[60]:


df[3] = np.nan
df


# In[61]:


df.dropna(axis='columns', how='all')# in which coloumns/rows having all null values


# In[66]:


df.dropna(axis='rows', thresh=3) #Here the first and last row have been dropped,
                                 #because they contain only two nonnull values.


# In[67]:


data = pd.Series([1, np.nan, 2, None, 3], index=list('abcde'))
data


# In[72]:


data.fillna(0) #fill 0 in place of nan


# In[73]:


# forward-fill
data.fillna(method='ffill')


# In[74]:


# back-fill
data.fillna(method='bfill')


# In[75]:


df


# In[76]:


df.fillna(method='ffill', axis=1)


# In[83]:


df.fillna(method='bfill', axis=1)


# In[86]:


index = [('California', 2000), ('California', 2010),('New York', 2000), ('New York', 2010),('Texas', 2000), ('Texas', 2010)]
populations = [33871648, 37253956,18976457, 19378102,20851820, 25145561]
pop = pd.Series(populations, index=index)
pop


# In[87]:


pop[('California', 2010):('Texas', 2000)]


# In[89]:


pop[[i for i in pop.index if i[1] == 2010]]


# In[90]:


index = pd.MultiIndex.from_tuples(index)
index


# In[91]:


pop = pop.reindex(index)
pop


# In[92]:


pop[:, 2010]


# In[93]:


pop_df = pop.unstack()
pop_df


# In[94]:


pop_df.stack()


# In[95]:


pop_df = pd.DataFrame({'total': pop,'under18': [9267089, 9284094,4687374, 4318033,5906301, 6879014]})
pop_df


# In[98]:


f_u18 = pop_df['under18'] / pop_df['total']
f_u18.unstack()


# In[99]:


df = pd.DataFrame(np.random.rand(4, 2),   #Methods of MultiIndex Creation
index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]],
columns=['data1', 'data2'])
df


# In[100]:


data = {('California', 2000): 33871648,
 ('California', 2010): 37253956,
 ('Texas', 2000): 20851820,
 ('Texas', 2010): 25145561,
 ('New York', 2000): 18976457,
 ('New York', 2010): 19378102}
pd.Series(data)


# In[101]:


pd.MultiIndex.from_arrays([['a', 'a', 'b', 'b'], [1, 2, 1, 2]])


# In[102]:


pd.MultiIndex.from_tuples([('a', 1), ('a', 2), ('b', 1), ('b', 2)])


# In[103]:


pd.MultiIndex.from_product([['a', 'b'], [1, 2]])


# In[113]:


pop.index.names = ['state', 'year']
pop


# In[ ]:




