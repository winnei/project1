# linear regression for multioutput regression
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
# create datasets
X, y = make_regression(n_samples=10, n_features=3, n_informative=5, n_targets=2, random_state=1)
X.shape
X
y.shape
y
# define model
model = LinearRegression()
# fit model
model.fit(X, y)
# make a prediction
data_in = [[-2.02220122, 0.31563495, 0.82797464, -0.30620401, 0.16003707, -1.44411381, 0.87616892, -0.50446586, 0.23009474, 0.76201118]]
yhat = model.predict(data_in)
# summarize prediction
print(yhat[0])

from data_utils import load_data
#Display data
data_reg = load_data('./data/regression_data.pkl') #length 4 dictionary: obs, actions, info, dones
data_posbc = load_data('./data/bc_with_gtpos_data.pkl') #length 2 dictionary
data_imgbc = load_data('./data/bc_data.pkl') #length 2 dictionary


print(type(data_imgbc))
print(len(data_imgbc))

for key in data_reg.keys():
    print(key)
obs=data_reg['obs'] #array, 500 colored images
actions=data_reg['actions'] #array, 500 ones and zeros
info=data_reg['info'] #list, contains agent position, obstacle positions, goal position
dones=data_reg['dones'] #a boolean, False

np.shape(actions)
actions[0:30]

np.shape(info)
info[1]
print(type(info))
agent_pos=info[1]['agent_pos']
print(type(agent_pos))

print(type(dones))
print(dones)
y_train=[]
for i in info:
    y_train.append(i['agent_pos'])
y_train[0:10]
np.shape(x_train)
np.shape(y_train)
x_train[0]
y_train[0:2]



X = np.array([[[1, 1],[2,0]], [[1, 2],[2,1]], [[2, 2],[2,8]],[[2, 3],[3,0]]])
#X = np.array([[1, 1], [1, 2], [2, 2],[2, 3]])
X.shape
X
# y = 1 * x_0 + 2 * x_1 + 3
y_0 = np.dot(X, np.array([1, 2,1])) + 3
y_1 = np.dot(X, np.array([3, 4,1])) - 3
Y=[]
for i in range(0,len(X)):
    yi=[y_0[i],y_1[i]]
    Y.append(yi)
Y=np.array(Y)
Y
Y[0]
print(np.shape(Y))

reg = LinearRegression().fit(X, Y)
