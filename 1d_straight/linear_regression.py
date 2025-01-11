import numpy as np

def featurization(X):
    X_feat = X.copy()
    X_feat = abs(X_feat)
    X_feat.insert(0, 'x0', np.ones(len(X_feat.index))) # X_feat.index represents the row labels, len(X_feat.index) gives the number of rows in the DataFrame
    return X_feat


class LinearRegressor():
    def __init__(self, X_feat, y):
        self.X_feat = X_feat # feature matrix
        self.y = y # ground true of training data
        self.m = len(y) # sample length
        self.theta = np.random.uniform(-2, 2, X_feat.shape[1]) # intialize parameters of hypothesis
        #self.h = np.dot(self.theta, self.X_feat.T)  # hypothesis

    def hypothesis(self):
        return np.dot(self.theta, self.X_feat.T)

    def cost(self):
        h = self.hypothesis()
        return (1/(2*self.m)) * np.sum( (h - self.y)**2 )
    
    def jacobian(self):
        h = self.hypothesis()
        return (1/self.m) * np.dot((h - self.y), self.X_feat)
    
    def gradient_descent(self, num_iters, lr): # lr = learning rate
        J_hist = np.zeros(num_iters)
        for i in range(num_iters):
            # updata theta (hypothesis parameters)
            Jac = self.jacobian()
            self.theta -= lr * Jac
            # record history of cost
            J_hist[i] = self.cost()    
        return self.theta, J_hist

