import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
    # Initialize the loss and gradient to zero.
  num_train=X.shape[0]
  loss = 0.0
  dW = np.zeros_like(W)
  
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
  scores=X.dot(W)
  scores=scores-np.amax(scores,axis=1)[:,np.newaxis]
  
  exp_scores=np.zeros_like(scores)
  for i in xrange(num_train):
    exp_row=np.exp(scores[i,:])
    exp_scores[i,:]=exp_row/np.sum(exp_row)
  loss-=np.log(exp_scores[[np.arange(num_train),y]])
  loss=np.sum(loss)/num_train
  loss += 0.5* reg * np.sum(W * W)
    
  for i in xrange(num_train):
    exp_row=np.exp(scores[i,:])
    tmp_row=exp_row/np.sum(exp_row)
    tmp_row[y[i]]-=1
    dW+=(X[i,:][:,np.newaxis]).dot(tmp_row[np.newaxis,:])
  dW/=num_train  
  dW += reg*W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train=X.shape[0]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores=X.dot(W)
  scores=scores-np.amax(scores,axis=1)[:,np.newaxis]
  exp_scores=np.exp(scores)
  exp_scores=np.divide(exp_scores.T,np.sum(exp_scores,axis=1)).T
  loss-=np.log(exp_scores[[np.arange(num_train),y]])
  loss=np.sum(loss)/num_train
  loss += 0.5* reg * np.sum(W * W)
  
  Bool_one=np.zeros(exp_scores.shape)
  Bool_one[[np.arange(num_train),y]]=1
  exp_scores-=Bool_one
  dW=(X.T).dot(exp_scores)
  dW/=num_train  
  dW += reg*W
    
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

