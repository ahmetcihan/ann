#include "mainwindow.h"

double ann::sigmoid_func(double val){
    return (1 / (1 + exp(-val)));     //sigmoid   - good
    //return tanh(val);                 //tanh      - works but slow and uses small learning rate
    //return val;                       //identity  - not properly worked
    //return atan(val);                   //atan      - not bad but slower
    //return (log(1+exp(val)));         //softplus  - good but slower
    /*********Leaky RELU**********/     //ReLU      -
    //if(val <= 0) return (0.01*val);
    //else return val;
    /*****************************/
    //return (val / (1 + exp(-val)));   //swish     - not bad but not good
    //return exp(-1*val*val);           //gaussien  - better than sigmoid
}
double ann::derivative_of_sigmoid_func(double val){
    return (sigmoid_func(val) * (1 - sigmoid_func(val)));
    //return (1 - tanh(val)*tanh(val));
    //return 1;
    //return (1 / (1 + val*val));
    //return (1 / (1 + exp(-val)));
    /***************RELU**********/
    //if(val < 0) return 0.01;
    //else return 1;
    /*****************************/
    //return (1 + exp(-val) + val*exp(-val))/((1 + exp(-val))*(1 + exp(-val)));
    //return -2*val*sigmoid_func(val);
}

