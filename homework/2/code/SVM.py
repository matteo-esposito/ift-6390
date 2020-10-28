import numpy as np


class SVM:
    def __init__(self, eta, C, niter, batch_size, verbose):
        self.eta = eta
        self.C = C
        self.niter = niter
        self.batch_size = batch_size
        self.verbose = verbose

    def make_one_versus_all_labels(self, y, m):
        """
        y : numpy array of shape (n,)
        m : int (num_classes)
        returns : numpy array of shape (n, m)
        """
        out = np.ones((len(y),m)) * -1 
        out[np.arange(len(y)),y] = 1
        return out

    def compute_loss(self, x, y):
        """
        x : numpy array of shape (minibatch size, num_features)
        y : numpy array of shape (minibatch size, num_classes)
        returns : float
        """
        reg_loss = (self.C/2) * np.sum(np.multiply(self.w,self.w))
        hinge_slope = 2 - np.multiply(np.matmul(x,self.w),y)
        hinge_loss = (1/x.shape[0]) * np.sum(np.maximum(0, hinge_slope)**2) 
        return hinge_loss + reg_loss


    def compute_gradient(self, x, y):
        """
        x : numpy array of shape (minibatch size, num_features)
        y : numpy array of shape (minibatch size, num_classes)
        returns : numpy array of shape (num_features, num_classes)
        """
        reg_grad = self.C * self.w
        
        hinge_val_slope = 2 - np.multiply(np.matmul(x,self.w),y)
        hinge_val = np.maximum(0,hinge_val_slope)
        #active = (hinge_val > 0).astype(int)
        
#         hinge_grad = np.zeros([x.shape[1],y.shape[1]])
#         for cls in range(y.shape[1]):
#             hinge_grad[:,cls] = np.mean(-1*x*(y[:,cls]*hinge_val[:,cls])[:,np.newaxis], axis=0)

        hinge_grad = np.matmul(x.transpose(), -1*y*hinge_val) / x.shape[0]
        #hinge_grad = 2*hinge_grad*np.maximum(0,hinge_slope)
        return 2 * hinge_grad + reg_grad

    # Batcher function
    def minibatch(self, iterable1, iterable2, size=1):
        l = len(iterable1)
        n = size
        for ndx in range(0, l, n):
            index2 = min(ndx + n, l)
            yield iterable1[ndx: index2], iterable2[ndx: index2]

    def infer(self, x):
        """
        x : numpy array of shape (num_examples_to_infer, num_features)
        returns : numpy array of shape (num_examples_to_infer, num_classes)
        """
        infered = np.ones([x.shape[0], self.m]) * (-1)
        distances = np.zeros([x.shape[0], self.m])
        for clss in range(self.m):
            #not sure whether i should use the norming by w
            distances[:,clss] = np.matmul(x,self.w[:,clss]) #/ np.sum(self.w[:,cls]*self.w[:,cls])
        
        infered[np.arange(x.shape[0]), np.argmax(distances, axis=1)] = 1
        return infered

    def compute_accuracy(self, y_inferred, y):
        """
        y_inferred : numpy array of shape (num_examples, num_classes)
        y : numpy array of shape (num_examples, num_classes)
        returns : float
        """
        matching = np.mean(np.multiply(y_inferred,y), axis=1)
        
        return np.sum(matching==1)/y.shape[0]

    def fit(self, x_train, y_train, x_test, y_test):
        """
        x_train : numpy array of shape (number of training examples, num_features)
        y_train : numpy array of shape (number of training examples, num_classes)
        x_test : numpy array of shape (number of training examples, nujm_features)
        y_test : numpy array of shape (number of training examples, num_classes)
        returns : float, float, float, float
        """
        self.num_features = x_train.shape[1]
        self.m = y_train.max() + 1
        y_train = self.make_one_versus_all_labels(y_train, self.m)
        y_test = self.make_one_versus_all_labels(y_test, self.m)
        self.w = np.zeros([self.num_features, self.m])

        train_losses = []
        train_accs = []
        test_losses = []
        test_accs = []

        for iteration in range(self.niter):
            # Train one pass through the training set
            for x, y in self.minibatch(x_train, y_train, size=self.batch_size):
                grad = self.compute_gradient(x, y)
                self.w -= self.eta * grad

            # Measure loss and accuracy on training set
            train_loss = self.compute_loss(x_train, y_train)
            y_inferred = self.infer(x_train)
            train_accuracy = self.compute_accuracy(y_inferred, y_train)

            # Measure loss and accuracy on test set
            test_loss = self.compute_loss(x_test, y_test)
            y_inferred = self.infer(x_test)
            test_accuracy = self.compute_accuracy(y_inferred, y_test)

            if self.verbose:
                print(f"Iteration {iteration} | Train loss {train_loss:.04f} | Train acc {train_accuracy:.04f} |"
                      f" Test loss {test_loss:.04f} | Test acc {test_accuracy:.04f}")

            # Record losses, accs
            train_losses.append(train_loss)
            train_accs.append(train_accuracy)
            test_losses.append(test_loss)
            test_accs.append(test_accuracy)

        return train_losses, train_accs, test_losses, test_accs