import numpy as np

class UserDefinedNMF:
    
    def __init__(self, n_factors=15, n_epochs=30, 
                reg_pu=0.12, reg_qi=0.12, learning_rate=0.01,
                beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, 
                user_id_map=None, item_id_map=None):
        self.n_factors = n_factors
        self.n_epochs = n_epochs 
        self.reg_pu = reg_pu
        self.reg_qi = reg_qi
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.user_id_map = user_id_map
        self.item_id_map = item_id_map

        self.user_factors = None
        self.item_factors = None
        self.m_u = None
        self.v_u = None
        self.m_i = None
        self.v_i = None

    
    def init_factors(self, num_factors, size):
        return np.abs(np.random.normal(0, .1, (size, num_factors)))
    
    def init_adam(self, n_users, n_items):
        self.m_u = np.zeros((n_users, self.n_factors))
        self.v_u = np.zeros((n_users, self.n_factors))
        self.m_i = np.zeros((n_items, self.n_factors))
        self.v_i = np.zeros((n_items, self.n_factors))

    def update_with_adam(self, u, i, error, t):
        grad_u = error * self.item_factors[i] - self.reg_pu * self.user_factors[u]
        grad_i = error * self.user_factors[u] - self.reg_qi * self.item_factors[i]

        self.m_u[u] = self.beta1 * self.m_u[u] + (1 - self.beta1) * grad_u
        self.v_u[u] = self.beta2 * self.v_u[u] + (1 - self.beta2) * (grad_u**2)

        self.m_i[i] = self.beta1 * self.m_i[i] + (1 - self.beta1) * grad_i
        self.v_i[i] = self.beta2 * self.v_i[i] + (1 - self.beta2) * (grad_i**2)

        m_hat_u = self.m_u[u] / (1 - self.beta1**t + self.epsilon)
        v_hat_u = self.v_u[u] / (1 - self.beta2**t + self.epsilon)

        m_hat_i = self.m_i[i] / (1 - self.beta1**t + self.epsilon)
        v_hat_i = self.v_i[i] / (1 - self.beta2**t + self.epsilon)

        self.user_factors[u] += self.learning_rate * m_hat_u / (np.sqrt(v_hat_u) + self.epsilon)
        self.item_factors[i] += self.learning_rate * m_hat_i / (np.sqrt(v_hat_i) + self.epsilon)

    def fit(self, trainset):
        self.user_factors = self.init_factors(self.n_factors, trainset.n_users)
        self.item_factors = self.init_factors(self.n_factors, trainset.n_items)
        self.init_adam(trainset.n_users, trainset.n_items)

        for epoch in range(1, self.n_epochs+1):
            for u, i, r in trainset.all_ratings():
                error = r - np.dot(self.user_factors[u], self.item_factors[i])
                self.update_with_adam(u, i, error, epoch)                


            if epoch % 10 == 0:
                self.learning_rate *= 0.9

    def predict(self, u, i, blend_factor=0.80):
        raw_est = np.dot(self.user_factors[u], self.item_factors[i])
        activated_est = self.custom_activation(raw_est)
        est = (blend_factor * raw_est) + ((1 - blend_factor) * activated_est)


        est = max(min(est, 5.0), 0.5)
        return est

    
    def custom_activation(self, x, min_val=0.5, max_val=5.0):
        tanh_output = np.tanh(x)

        scaled_tanh = (tanh_output + 1) / 2

        scaled_x = scaled_tanh * (max_val - min_val) + min_val

        return scaled_x



    def test(self, testset):
        predictions = []

        for u, i, r in testset:
            try:
                u_mapped = self.user_id_map.get(u, None) if self.user_id_map else u
                i_mapped = self.item_id_map.get(i, None) if self.item_id_map else i

                if u_mapped is None or i_mapped is None:
                    print(f"Skipping prediction for uid: {u} and iid: {i} - ID out of bounds")
                    continue

                est = self.predict(u_mapped, i_mapped)
                predictions.append((u, i, r, est))
            except Exception as e:
                print(f"Error predicting for uid: {u} and iid: {i}: {e}")
        return predictions
    


    
