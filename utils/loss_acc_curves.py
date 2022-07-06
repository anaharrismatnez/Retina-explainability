import matplotlib.pyplot as plt
import pickle
import numpy as np



def PLOT_CURVES(train_acc, test_acc, train_loss, test_loss):
    plt.plot(train_acc, linewidth=0.8)
    plt.plot(test_acc, linewidth=0.8)
    plt.xlim(-0.5,25)
    plt.ylim(0,110)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['Train','Valid'])
    plt.title('Train vs Valid Accuracy')
    
    plt.show()

    plt.plot(train_loss, linewidth=1)
    plt.plot(test_loss, linewidth=1)
    plt.xlim(-0.5,25)
    plt.xlabel('epoch')
    plt.ylabel('losses')
    plt.legend(['Train','Valid'])
    plt.title('Train vs Valid Losses')
    
    plt.show() 

def PLOT_LR_LOSS (test_loss,train_loss,currLR):
    plt.plot(currLR,train_loss, linewidth=0.8)
    plt.ylim((0,1))
    plt.xlim((0.00009, 0.11))
    plt.xscale("log")
    plt.xlabel('epoch')
    plt.ylabel('losses')
    plt.legend(['Train','Valid'])
    plt.title('Train vs Valid Losses')
    
    plt.show() 
 
    plt.plot(currLR,test_loss, linewidth=0.8)
    plt.ylim(0,1)
    plt.xlim(0.00009, 0.11)
    plt.xscale("log")
    plt.xlabel('epoch')
    plt.ylabel('losses')
    plt.legend(['Train','Valid'])
    plt.title('Train vs Valid Losses')
    
    plt.show()

if __name__ == '__main__':


    train_loss = pickle.load(open(r'../../interpretability/reduced_VGG/GRAD_CAM/train_loss_25_0.1_reduced.pkl','rb'))
    train_acc = pickle.load(open(r'../../interpretability/reduced_VGG/GRAD_CAM/train_acc_25_0.1_reduced.pkl','rb'))
    test_loss = pickle.load(open(r'../../interpretability/reduced_VGG/GRAD_CAM/test_loss_25_0.1_reduced.pkl','rb'))
    test_acc = pickle.load(open(r'../../interpretability/reduced_VGG/GRAD_CAM/test_acc_25_0.1_reduced.pkl','rb'))
    PLOT_CURVES(train_acc, test_acc, train_loss, test_loss)  

    """ test_loss= pickle.load(open(r'../pkls/test_loss_25_reduced.pkl','rb'))
    train_loss= pickle.load(open(r'../pkls/train_loss_25_reduced.pkl','rb'))
    currLR= pickle.load(open(r'../pkls/StepLR_25_reduced.pkl','rb')) 
    

    PLOT_LR_LOSS (test_loss,train_loss,currLR)"""

