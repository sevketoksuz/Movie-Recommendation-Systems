import numpy as np

class UserDefinedNMF:
    
    def __init__(self, n_factors=15, n_epochs=30, 
                reg_pu=0.12, reg_qi=0.12, learning_rate=0.01,
                #,reg_bu=0.04, reg_bi=0.02, lr_bu=0.01, lr_bi=0.01 
                user_id_map=None, item_id_map=None):
        self.n_factors = n_factors
        self.n_epochs = n_epochs 
        self.reg_pu = reg_pu #used
        self.reg_qi = reg_qi #used
        self.learning_rate = 0.01
        self.user_id_map = user_id_map
        self.item_id_map = item_id_map

        #self.reg_bu = reg_bu
        #self.reg_bi = reg_bi 
        #self.lr_bu = lr_bu
        #self.lr_bi = lr_bi
    
    def init_factors(self, num_factors, size):
        return np.abs(np.random.normal(0, .1, (size, num_factors)))
    
    def fit(self, trainset):
        self.user_factors = self.init_factors(self.n_factors, trainset.n_users)
        self.item_factors = self.init_factors(self.n_factors, trainset.n_items)

        for epoch in range(self.n_epochs):
            for u, i, r in trainset.all_ratings():
                error = r - np.dot(self.user_factors[u], self.item_factors[i])
                self.user_factors[u] += self.learning_rate * (error * self.item_factors[i] - self.reg_pu * self.user_factors[u])
                self.item_factors[i] += self.learning_rate * (error * self.user_factors[u] - self.reg_qi * self.item_factors[i])
                


            if epoch % 10 == 0 and epoch > 0:
                self.learning_rate *= 0.9

    def predict(self, u, i):
        est = np.dot(self.user_factors[u], self.item_factors[i]) 
        return est
    
    def test(self, testset):
        predictions = []

        for u, i, r in testset:
            try:
                u_mapped = self.user_id_map.get(u, None) if self.user_id_map else u
                i_mapped = self.item_id_map.get(i, None) if self.item_id_map else i

                if u_mapped is None or i_mapped is None:
                    print(f"Skipping prediction for uid: {u} and iid: {i} - ID out of bounds")
                    continue

                est = self.predict(u, i)
                predictions.append((u, i, r, est))
            except Exception as e:
                print(f"Error predicting for uid: {u} and iid: {i}: {e}")
        return predictions