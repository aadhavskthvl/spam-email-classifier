# class LogisticRegressionClassifier:
#     def __init__(self, learning_rate=0.01, num_iterations=10000):
#         self.learning_rate = learning_rate
#         self.num_iterations = num_iterations
#         self.weights = None
#         self.bias = None

#     def sigmoid(self, z):
#         return 1 / (1 + np.exp(-z))

#     def train(self, data):
#         features = data[:, 1:]
#         labels = data[:, 0]

#         num_samples, num_features = features.shape

#         # Initialize weights and bias
#         self.weights = np.zeros(num_features)
#         self.bias = 0

#         # Gradient descent
#         for _ in range(self.num_iterations):
#             linear_model = np.dot(features, self.weights) + self.bias
#             y_predicted = self.sigmoid(linear_model)

#             dw = (1 / num_samples) * np.dot(features.T, (y_predicted - labels))
#             db = (1 / num_samples) * np.sum(y_predicted - labels)

#             self.weights -= self.learning_rate * dw
#             self.bias -= self.learning_rate * db

#     def predict(self, new_data):
#         linear_model = np.dot(new_data, self.weights) + self.bias
#         y_predicted = self.sigmoid(linear_model)
#         y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
#         return np.array(y_predicted_cls)

# class LogisticRegressionClassifier:
#     def __init__(self, learning_rate=0.01, num_iterations=10000):
#         self.learning_rate = learning_rate
#         self.num_iterations = num_iterations
#         self.weights = None
#         self.bias = None

#     def sigmoid(self, z):
#         return 1 / (1 + np.exp(-z))

#     def cross_entropy_loss(self, y_true, y_pred):
#         return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

#     def train(self, data):
#         features = data[:, 1:]
#         labels = data[:, 0]

#         num_samples, num_features = features.shape

#         # Initialize weights and bias
#         self.weights = np.zeros(num_features)
#         self.bias = 0

#         # Gradient descent
#         for _ in range(self.num_iterations):
#             linear_model = np.dot(features, self.weights) + self.bias
#             y_predicted = self.sigmoid(linear_model)

#             dw = (1 / num_samples) * np.dot(features.T, (y_predicted - labels))
#             db = (1 / num_samples) * np.sum(y_predicted - labels)

#             self.weights -= self.learning_rate * dw
#             self.bias -= self.learning_rate * db

#             # Calculate the cross-entropy loss
#             loss = self.cross_entropy_loss(labels, y_predicted)
#             print(f'Iteration: {_}, Loss: {loss}')

#     def predict(self, new_data):
#         linear_model = np.dot(new_data, self.weights) + self.bias
#         y_predicted = self.sigmoid(linear_model)
#         y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
#         return np.array(y_predicted_cls)


# def create_classifier2(data):
#     classifier = LogisticRegressionClassifier()
#     classifier.train(data)
#     return classifier

# # classifier2 = create_classifier2(np.loadtxt(open("data/training_spam.csv"), delimiter=",").astype(int))
# classifier2 = create_classifier2(np.concatenate((np.loadtxt(open("data/training_spam.csv"), delimiter=",").astype(int), training_data)))
