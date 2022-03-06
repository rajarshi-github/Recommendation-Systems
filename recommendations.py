import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re

from scipy.sparse import csr_matrix

from sklearn.neighbors import NearestNeighbors

from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cosine

# function to get zone
def getZone(pincode):
    # zones dictionary 
    zones = {'Delhi': 110000, 'Mumbai':400000, 'Bangalore': 560000}

    if pincode < 100000 or pincode > 570000:
        return None
    for k, v in zones.items():
        if abs(v - pincode) < 100000:
            return k
    return None


# In order to make any decision we have to give some weightage to an OrderStatus 
# so that eventually we can come up with an interaction score for the {Customer->Product} relationship
# We will use use that score in our model building in later part

# --------------------------
# Weights 
# --------------------------
# Delivered             = 50 
# Payment Incomplete    = 10
# Order Cancelled       = 1
# Item Returned         = 0
# --------------------------

# lets have a function for it
def getUserScore(x):
    if x == 'Delivered':
        return 50
    elif x == 'Payment Incomplete':
        return 10 
    elif x == 'Order Cancelled':
        return 1
    else:
        return 0 


def kmeansCustomerRecos(model, dfFinal, customerId, neighbors=5):
    
    distances, indices = model.kneighbors( dfFinal[dfFinal.index == customerId].values.reshape(1,-1), neighbors)
    nearestCustomers = []
    for i in range(0, len(distances.flatten())):
        if i == 0:
            print(f"Nearest Neighbours of Customer : {dfFinal.index[indices.flatten()[i]]} ")
        else:
            customer = dfFinal.index[indices.flatten()[i]]
            customerDistance = distances.flatten()[i]
            print('{0}: Customer={1}, with distance of {2}'.format(i, customer, customerDistance))

            nearestCustomers.append(customer)

    # we will try to find the predicted Vector as a mean of all the nearest neighbours 
    totalProducts = len(dfProducts)-1
    predictedVector = np.zeros((1, totalProducts))

    numNearestCustomers = len(nearestCustomers)
    for customer in nearestCustomers:
        predictedVector = predictedVector + dfFinal[(dfFinal.index == customer)].values

    predictedVector = predictedVector / numNearestCustomers

    # find the actual vector 
    actual = dfFinal[dfFinal.index == customerId].values.reshape(1,-1)

    # similarity of actual and predicted 
    distance = cosine(actual, predictedVector)
    similarity = 1 - distance

    print(f"Recommendations from model has a similarity of {similarity} with actual behavior for Customer: {customerId} ")
    #print(f"Recommendations for {CUSTOMERID} has a distance of {distance} from actual")    
    print("Recommendations: ")
    recos = {}
    for idx, weight in enumerate(list(predictedVector.flatten())):
        recos[idx]= weight
    top10reco = list(sorted(recos.items(), key=lambda item: item[1], reverse=True))[:10]
    top10recoIdx = [ item[0] for item in top10reco]
    for k , item in enumerate(dfFinal.columns):
        if k in top10recoIdx:
            print(item)
            
    print("\n\n")

def kmeansProductRecos(model, dfFinal, product, neighbors=10):
    
    distances, indices = model.kneighbors( dfFinal[dfFinal.index == product].values.reshape(1,-1), neighbors)
    nearestProducts = []
    for i in range(0, len(distances.flatten())):
        if i == 0:
            print(f"Nearest Neighbours of Product : {dfFinal.index[indices.flatten()[i]]} ")
        else:
            product = dfFinal.index[indices.flatten()[i]]
            productDistance = distances.flatten()[i]
            print('{0}: Product={1}, with distance of {2}'.format(i, product, productDistance))

            nearestProducts.append(product)
    print("\n\n")

print ("==============================")
print ("COLLABORATIVE FILTER MECHANISM")
print ("==============================")

#load the dataset into pandas dataframe
dfProducts = pd.read_csv('/Users/rpghosh/python-examples/reward360/data/products.csv', header=None)
dfTxn = pd.read_csv('/Users/rpghosh/python-examples/reward360/data/txn.csv')

# Add the column names to the Product dataframe
dfProducts.columns = ['Product', 'Price']

# Convert the column names into lower case 
# also remove any leading or lagging spaces, also whitespaces in between
# ----- Txn dataframe -----
# remove spaces in the product names
dfTxn['Product'] = dfTxn['Product'].apply ( lambda x: re.sub( '\s+', '', x) )

# conver to lower case
dfTxn['Product'] = dfTxn['Product'].apply ( lambda x: x.lower() )


# ----- Product dataframe -----
# remove spaces in the product names
dfProducts['Product'] = dfProducts['Product'].apply ( lambda x: re.sub( '\s+', '', x) )

# conver to lower case
dfProducts['Product'] = dfProducts['Product'].apply ( lambda x: x.lower() )

# all records have nan value for dfTxn.OrderDate
# so we can remove that column
# also the EmailAddress column seems to be of no use, since we 
# already have CustomerId

dfTxn.drop(columns=['CustomerEmailAddress', 'OrderDate'], inplace=True)

# looking at the price column it seems that some items are siginificantly higher priced than others 
# we will convert the price of product into 5 bins
numBins = 5
dfProducts['PriceCategory'] = pd.qcut(x=dfProducts['Price'], q = numBins, labels=[i+1 for i in range(numBins)] )


# new column dfTxn.["ProductPriceCategory"]
#dfTxn['ProductPriceCategory'] = 0
mergedDf = pd.merge(dfTxn, dfProducts, on='Product')
dfTxn = mergedDf

# add zone column
dfTxn['Zone'] = dfTxn['DeliveryPinCode'].apply(getZone)        


dfTxn['OrderStatusScore'] = dfTxn['OrderStatus'].apply(getUserScore)

# payment mode - lets make it into codes 
paymentCodes = {'net_banking':1, 'credit_card':2, 'upi':3, 'wallet':4, 'debit_card':5}
dfTxn['PaymentMode'] = dfTxn['OrderPaymentMode'].apply( lambda x: paymentCodes[x])

#  CustomerDeviceType - lets make it into codes 
dfTxn.CustomerDeviceType.unique()
deviceTypes = {'iOS':1, 'Windows':1, 'Linux':2, 'Android':3}
dfTxn['DeviceType'] = dfTxn['CustomerDeviceType'].apply(lambda x: deviceTypes[x])

# Gender - lets make it into codes 
genderTypes = {'Male': 1, 'Female':2}
dfTxn['Gender'] = dfTxn['CustomerGender'].apply(lambda x: genderTypes[x])

# Age - lets make it into 5 bins
dfTxn['Age'] = 0
dfTxn['Age'] = pd.qcut(x=dfTxn['CustomerAge'],q=5,labels=[1,2,3,4,5])

# lets consider only Delhi and Mumbai zone data
dfTrain = dfTxn[(dfTxn.Zone.isin(['Delhi','Mumbai']))]

# consider all these columns
dfTrain = dfTrain[['CustomerId','Gender', 'Age', 'PaymentMode', 'DeviceType','OrderStatusScore', 'PriceCategory' , 'Product']]

# ---------------------------------------------------------------------------------------------
# find the net interaction score of a transaction
print("Interaction based recommendation without considering Age")
# we will multiply TWO scores to get the net InteractionScore 
# 1. OrderStatusScore - which is based on type of transaction (delivered, cancelled, incomplete, returned)
# 2. PriceCategory - price category ranging from 1 to 5 (given this datset price of an item might 
#                                                       affect the type of transaction )
# ---------------------------------------------------------------------------------------------
dfTrain['InteractionScore'] = dfTrain['OrderStatusScore'].astype(np.float32) * dfTrain['PriceCategory'].astype(np.float32)


# in a new dataframe we get the score of relationship between a customerId - product
dfPriceScore = dfTrain.groupby(['CustomerId','Product'])['InteractionScore'].sum().reset_index()

# we need to create a dataframe where each customer can be represented as a vector
# and the field of the vector will be each product
# and values will be the InteractionScore for each customer-product relationship
dfFinal = dfPriceScore.pivot_table(index='CustomerId', columns='Product', values='InteractionScore').fillna(0)

print(f"Final vector for Customers")
print(f"Number of Customers : {dfFinal.shape[0]}")
print(f"Number of Products  : {dfFinal.shape[1]}")

customerMatrix = csr_matrix(dfFinal.values)

# Initialize a NearestNeighbors model
model = NearestNeighbors(metric='cosine', algorithm='brute')

print("Model hyper-parameters")
print(str(model.get_params()))

model.fit(customerMatrix)

# Let test with an example
print ("Selecting 20 random customers ... ")

allCustomers = dfFinal.index.to_list()
for _ in range(20):
    idx = np.random.choice(10000)
    customer = allCustomers[idx]
    kmeansCustomerRecos(model, dfFinal, customer, 5)
    



# ---------------------------------------------------------------------------------------------
# find the age based  interaction score of a transaction
print("========================")
print("Age based recommendation")
print("========================")
# Age of the customer will play a major role about the kind of product transacted
# ---------------------------------------------------------------------------------------------

# in a new dataframe we get the score of relationship between a customerId - product
dfAgeScore = dfTrain.groupby(['CustomerId','Product'])['Age'].size().reset_index()

# we need to create a dataframe where each customer can be represented as a vector
# and the field of the vector will be each product
# and values will be the InteractionScore for each customer-product relationship
dfFinal = dfAgeScore.pivot_table(index='CustomerId', columns='Product', values='Age').fillna(0)

print(f"Final vector for Customers")
print(f"Number of Customers : {dfFinal.shape[0]}")
print(f"Number of Products  : {dfFinal.shape[1]}")

customerMatrix = csr_matrix(dfFinal.values)

# Initialize a NearestNeighbors model
model = NearestNeighbors(metric='cosine', algorithm='brute')

print("Model hyper-parameters")
print(str(model.get_params()))

model.fit(customerMatrix)

# Let test with an example
#kmeansCustomerRecos(model, dfFinal, '012ILK3ZTP', 5)
print ("Selecting 20 random customers ... ")

allCustomers = dfFinal.index.to_list()
for _ in range(20):
    idx = np.random.choice(10000)
    customer = allCustomers[idx]
    kmeansCustomerRecos(model, dfFinal, customer, 5)
    

print ("==============================")
print ("CONTENT BASED FILTER MECHANISM")
print ("==============================")
# this recommendation mechanism will give the top product recommendations 
# to any incoming new customer
# I would like to use this mechanism to address the cold start problem 

# using only the Bangalore circle data
dfTrain = dfTxn[dfTxn.Zone.isin(['Bangalore'])]
print(f"Number of record : {dfTrain.shape}")

# consider all these columns
dfTrain = dfTrain[['CustomerId', 'Gender', 'Age', 'PaymentMode', 'DeviceType', 'OrderStatusScore', 'PriceCategory' , 'Product']]

# ---------------------------------------------------------------------------------------------
# find the net interaction score of a transaction
print("Interaction based recommendation without considering Age")
# we will multiply TWO scores to get the net InteractionScore 
# 1. OrderStatusScore - which is based on type of transaction (delivered, cancelled, incomplete, returned)
# 2. PriceCategory - price category ranging from 1 to 5 (given this datset price of an item might 
#                                                       affect the type of transaction )
# ---------------------------------------------------------------------------------------------
dfTrain['InteractionScore'] = dfTrain['OrderStatusScore'].astype(np.float32) * dfTrain['PriceCategory'].astype(np.float32)


# in a new dataframe we get the score of relationship between a customerId - product
dfPriceScore = dfTrain.groupby(['Product','CustomerId'])['InteractionScore'].sum().reset_index()

# we need to create a dataframe where each customer can be represented as a vector
# and the field of the vector will be each product
# and values will be the InteractionScore for each customer-product relationship
dfFinal = dfPriceScore.pivot_table(index='Product', columns='CustomerId', values='InteractionScore').fillna(0)

print(f"Final vector for Products")
print(f"Number of Products : {dfFinal.shape[0]}")
print(f"Number of Customers  : {dfFinal.shape[1]}")

productMatrix = csr_matrix(dfFinal.values)

# Initialize a NearestNeighbors model
model = NearestNeighbors(metric='cosine', algorithm='brute')

print("Model hyper-parameters")
print(str(model.get_params()))

model.fit(productMatrix)

# Let test with an example
print ("Selecting 3 random products ... ")

# list of all products
allProducts = dfProducts['Product'].to_list()

for _ in range(3):
    idx = np.random.choice(len(dfProducts)-1)
    product = allProducts[idx]
    kmeansProductRecos(model, dfFinal, product, 10)
    
# ---------------------------------------------------------------------------------------------
# find the net age based score of a transaction
print("Age based recommendation ")
# ---------------------------------------------------------------------------------------------

# in a new dataframe we get the score of relationship between a customerId - product
dfPriceScore = dfTrain.groupby(['Product','CustomerId'])['Age'].size().reset_index()

# we need to create a dataframe where each customer can be represented as a vector
# and the field of the vector will be each product
# and values will be the InteractionScore for each customer-product relationship
dfFinal = dfPriceScore.pivot_table(index='Product', columns='CustomerId', values='Age').fillna(0)

print(f"Final vector for Products")
print(f"Number of Products : {dfFinal.shape[0]}")
print(f"Number of Customers  : {dfFinal.shape[1]}")

productMatrix = csr_matrix(dfFinal.values)

# Initialize a NearestNeighbors model
model = NearestNeighbors(metric='cosine', algorithm='brute')

print("Model hyper-parameters")
print(str(model.get_params()))

model.fit(productMatrix)

# Let test with an example
print ("Selecting 3 random products ... ")

# list of all products
allProducts = dfProducts['Product'].to_list()

for _ in range(3):
    idx = np.random.choice(len(dfProducts)-1)
    product = allProducts[idx]
    kmeansProductRecos(model, dfFinal, product, 10)
    
