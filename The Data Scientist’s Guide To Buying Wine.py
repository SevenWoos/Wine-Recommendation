#!/usr/bin/env python
# coding: utf-8

# ## The Data Scientist’s Guide To Buying Wine
# Isolating the chemicals that make wine great.
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ### Importing the data and preprocessing

# In[2]:


# Importing the red wine data set and then delimiting
wine_data_red = pd.read_csv("winequality-red.csv", delimiter = ';')
wine_data_red


# In[3]:


# Importing the white wine data
wine_data_white = pd.read_csv("winequality-white.csv", delimiter = ';')
wine_data_white


# In[4]:


# Let us add the type 'column' to identify whether the wine is red or white
wine_data_red['type'] = ['red']*len(wine_data_red)
wine_data_white['type'] = ['white']*len(wine_data_white)


# In[5]:


wine_data_red.head()


# In[6]:


wine_data_white.head()


# In[7]:


# Combining both datasets
wine_data_df = wine_data_red.append(wine_data_white, ignore_index=True)


# In[8]:


wine_data_df


# In[9]:


# A random sample
wine_data_df.sample(10)


# ### What exactly do these chemicals imply for consumers?
# 
# Fixed Acidity: Gives wine a tart flavor. If there’s too little, wine tastes “flat”.
# 
# Citric Acid: Often used as a flavor additive. Adds a fresh, tart taste.
# 
# Volatile Acidity: In contrast to citric acid, this is gaseous acidity that can smell like vinegar; its presence is less likely to be intentional.
# 
# Residual Sugar: Correlates with sweetness. This is the sugar left over when grapes finish fermenting. “Dry wines” tend to have lower sugar.
# 
# Sulfur: Additive to prevent bacterial growth. In my research, it was contested whether or not there is a smell or taste associated with it.
# 
# Chlorides: The measure of salt.

# ### It looks like there’s both red and white bottles in this dataset. Is the distribution even?

# In[10]:


# We graph the distribution so we see how many red vs white bottles we have

# plt.rcParams["font.family"] = "avenir"
plt.suptitle("Distribution of Wine Quality and Color", fontsize=20)
plt.xlabel("xlabel", fontsize=18)
plt.ylabel("ylabel", fontsize=16)

ax = sns.countplot(
    x="quality", hue="type", data=wine_data_df, palette=["#da627d", "#f9dbbd"]
)
ax.set(xlabel="Wine Quality Rating", ylabel="Number of Bottles")

wine_data_df.groupby(["quality", "type"]).size()


# From this, we can see we have an imbalance in our data. For this demo, we’re just visualizing static data, so while the overrepresentation is worth noting, it’s not the end of the world.
# 
# But if we were trying to train a machine learning model with this data, it would be problematic. Why? Because certain categories are over or under-represented, so unless we do an intervention, it could lead to an issue with our model becoming biased.

# ### Is There Any Difference In Chemical Composition Between Different Ratings of Wine?
# If quality is just a rating of flavor, and flavor is a mixture of chemicals, wines with different ratings should have different chemical ratios, right?

# In[11]:


### Let’s test this theory on our red wine bottles!
wine_data_red.groupby(['quality']).mean().plot.bar(stacked=True, cmap="RdYlBu", figsize=(15, 5))

plt.suptitle(
    "Average Chemical Distribution Across Red Wine Of Different Qualities", fontsize=20
)
plt.xticks(size=18, rotation="horizontal")
plt.xlabel("Quality", fontsize=18)

plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.48), ncol=4, fontsize=15)


# At first glance, the chemicals follow a bell curve distribution, and an 8-star wine looks no different than a 4-star wine. That doesn’t sound intuitive.
# 
# That said, I think the data is scattered across too many wine-quality groupings. I’m thinking if we group them properly, we’ll find a more meaningful relationship.
# 
# We don’t care about the difference is between a 3 and 4-star bottle. We want to see what makes a bottle *phenomenal*.
# 
# We need to identify what causes the jump in quality between a 3 and an 8, not a 5 and a 6. So instead, let’s categorize a wine as “terrible”, “average”, and “phenomenal”.

# ### Let's break it down into 3 different wine categories, for each type

# In[12]:


# Beginning with red
category = []

for row in wine_data_red['quality']:
    if row >= 7:
        category.append('Phenomenal')
    elif row <= 4:
        category.append('Terrible')
    else:
        category.append('Average')

wine_data_red['category'] = category


# In[13]:


# Then remove the rating since we no longer need to differentiate based off of it
wine_data_red.drop(['quality'], axis=1)


# In[14]:


#Let's customize how the X axis is sorted. 
# Given they're categorical variables, I want them in a very specific order now.
wine_data_red['category'] = pd.Categorical(wine_data_red['category'], ['Terrible', 'Average', 'Phenomenal'])
wine_data_red.sort_values(by='category')


# In[15]:


# And last, let's group it by quality category
wine_data_red.groupby(['category']).mean().plot.bar(stacked=True, cmap="RdYlBu", figsize=(15, 5))

plt.suptitle('Average Chemical Distribution Across Red Wine Of Different Qualities', fontsize=20)
plt.xticks(size = 18, rotation = 'horizontal')
plt.xlabel('Quality', fontsize=18)

plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.48), ncol= 4, fontsize = 15)


# In[16]:


wine_data_red.groupby('category').mean()


# The average content of each chemical amongst each category of red wine.

# ### What trends do we see in red wines?
# By grouping our wines more intuitively, we uncover a few valuable relationships.

# 1. Low Salt: On average, the worst red wines have the greatest amount of chlorides. This makes sense given chlorides measured that “salty” character. “Phenomenal” wine has the least.
# 
# 
# 2. Acidity Matters: “phenomenal” wines have the least amount of volatile acidity, and the greatest amount of citric acid. Given what we know about wine acids, this makes sense: the citric acid was likely deliberately introduced because it gives the wine a pleasant taste, but the volatile acids were likely a product of poor fermentation.
# 
# 
# 3. Alcohol is King: The best red wines have the most alcohol.
# 
# Now that we have a profile for each category of wine, are there any additional relationships we can explore that could help explain why each snapshot looks the way it does?
# Let’s see if there are any other correlations between chemicals present in the wine.

# ### Let’s see if there are any other correlations between chemicals present in the wine.

# In[17]:


sns.set_theme(style='white')

corr = wine_data_red.corr()

# Return a copy of an array with the elements below the k-th diagonal zeroed.
mask = np.triu(np.ones_like(corr, dtype=bool))

f, ax = plt.subplots(figsize=(11, 9))

cmap = sns.diverging_palette(230, 20, as_cmap=True)

sns.heatmap(
    corr,
    mask=mask,
    cmap="icefire",
    annot=True,
    vmax=0.3,
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.5},
)

plt.title("Correlations Between Compounds In Red Wine", size=24)


# Interestingly, this graph confirms what we just deduced: for red wines, alcohol content has a relatively strong correlation with wine quality rating (.48), followed by volatile acidity (-.39).
# 
# 
# Some of the other correlations can be contextualized via basic chemistry . For example, alcohol is less dense than water, so it makes sense that more alcohol would be correlated with a lower wine density.
# 
# 
# Likewise, we would expect there to be a strong correlation between pH and Acidity, given pH measures whether a substance leans towards being an acid or a base.

# ### Could it really be that the best red wines are just the ones with the highest alcohol content?
# According to my European colleagues, “no”. When I presented this notebook at work, they noted this model does not take into account the “soul of the wine- the fruity notes”.
# 
# They have a great point, and in the future, I do hope we have a dataset that notes specific flavors.
# 
# 
# Until then, I am extremely confident that picking the wine with the highest alcohol content will increase the odds of us choosing a high quality bottle.

# In[19]:


# Because there's an imbalance of wines in the dataset, let's only grab 20 of each category so it doesn't skew the plot

plt.suptitle(
    "Average Chemical Distribution Across Red Wine Of Different Qualities", fontsize=20
)

sns.relplot(
    x="alcohol",
    y="residual sugar",
    hue="category",
    size="alcohol",
    sizes=(40, 400),
    alpha=0.5,
    palette=["#006400", "#da627d", "#ff7b00"],
    height=5,
    data=wine_data_red.groupby("category")
    .apply(lambda x: x.sample(22))
    .reset_index(drop=True),
)

plt.xlabel("Alcohol Content", size=18)
plt.ylabel("Residual Sugar Content", size=18)

plt.title("Alcohol vs. Sugar Content In Each Category of Wine", size=24)


# A scatterplot comparing the alcohol and sugar content of 22 wines from each category.

# ### But wait- we just analyzed red wine! Is there a difference in the correlations for a high quality white wine?

# In[21]:


sns.set_theme(style='white')

corr2 = wine_data_white.corr()

# Return a copy of an array with the elements below the k-th diagonal zeroed.
mask = np.triu(np.ones_like(corr2, dtype=bool))

f, ax = plt.subplots(figsize=(11, 9))

cmap = sns.diverging_palette(230, 20, as_cmap=True)

sns.heatmap(
    corr2,
    mask=mask,
    cmap="crest",
    annot=True,
    vmax=0.3,
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.5},
)

plt.title("Correlations Between Compounds In White Wine", size=24)


# Right off the bat, we see that even for white wines, alcohol content has the most positive correlation with quality. (0.44)

# Compared to the red wines, for which the presence of citric acid had a 20% correlation with quality, the white wines had a 0% correlation between citric acid and quality.

# I would hypothesize that, because citric acid is an additive that is deliberately added to red wine to enhance its flavor, it is unlikely a lower quality red bottle would have had it added.
# 
# 
# In contrast, even the most terrible quality white wine had higher levels of citric acid than the average bottle of red.
# 
# 
# I cannot be sure if these differences occur because citric acid naturally exists in higher concentrations in white wine, or if it is a more common practice to add it to white wine.
# 
# 
# Ultimately, we need to research a little more to account for why this difference in correlations exists.

# Additionally, the profiles of the two are definitely not the same.
# 
# 
# At the chemical level, great white wines had 2–4 times the amount of sulfur than their red counterparts.
# 
# 
# Given what we learned about sulfur as an additive to prevent spoilage, unless white wines have more naturally occurring sulfur, it is possible that to ferment the them necessitates extra protection as opposed to red (just as the red wine benefits from the addition of citric acid).
# 
# 
# Additionally, white wines had half the amount of chlorides the reds did, but nearly twice as much sugar. They were also slightly less acidic.

# ### Despite these differences, the alcohol content of highly-quality red and white wines was roughly the same.

# # So finally, what bottle should I buy?
# 

# Whether red or white, I think you’ll be happy with nearly any bottle with a high alcohol content. 
# 
# Citric acid content won’t make a difference when purchasing a white bottle, but it will often make or break your red.
# 
# 
# Although this demo was fun, it’s important to acknowledge that we took a mathematical approach to describing one of the most romanticized, poetic substances in the world. Whereas a sommelier would describe the day as “breezy”, we described it as having “windspeeds of 5 MPH”. In being so precise, it is possible to lose something in translation.
# 
# 
# That said, although we’ll be choosing our bottles more confidently, something tells me that my colleagues were right: sitting across your date, beaming “nice bottle, you can really taste the sulfur”, just doesn’t have the same ring to it.

# In[ ]:




