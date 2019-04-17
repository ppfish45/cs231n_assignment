import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.T.shape) # initialize the gradient as zero

  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0

  for i in range(num_train):
    scores = X[i].dot(W)
    std = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      det = scores[j] - std + 1
      if det > 0:
        loss += det
        dW[j] += X[i] / num_train
        dW[y[i]] -= X[i] / num_train

  loss /= num_train
  loss += reg * np.sum(W * W)

  dW = dW.T + 2 * reg * W

  #############################################################################
  # DONE:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  
  #############################################################################
  # DONE:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]

  scores = X.dot(W)
  correct_scores = np.array(scores[np.arange(num_train), y])

  loss_matrix = scores - correct_scores.reshape(-1, 1) + 1
  loss_matrix[np.arange(num_train), y] -= 1
  mask = (loss_matrix > 0).astype(int)
  loss = np.sum(np.maximum(0, loss_matrix)) / num_train + reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  

  #############################################################################
  # DONE:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  dW = np.zeros(W.shape)
  dW += X.T.dot(mask)
  trans = np.zeros([num_train, num_classes])
  trans[np.arange(num_train), y] = mask.sum(axis = 1)
  dW -= X.T.dot(trans)
  dW = dW / num_train + 2 * reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
