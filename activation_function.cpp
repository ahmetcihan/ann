#include "mainwindow.h"

double ann::sigmoid_func(double val){
    //return (1 / (1 + exp(-val)));     //sigmoid   - good
    //return tanh(val);                 //tanh      - not worked
    //return val;                       //identity  - not properly worked
    //return atan(val);                 //atan      - not bad but slower
    //return (log(1+exp(val)));         //softplus  - good but slower
    /***************RELU**********/     //ReLU      - very good training but test is bad
    //if(val < 0) return 0;
    //else return val;
    /*****************************/
    /*********Leaky RELU**********/     //ReLU      -
    if(val <= 0) return (0.01*val);
    else return val;
    /*****************************/
    //return (val / (1 + exp(-val)));   //swish     - not bad but not good
    //return exp(-1*val*val);           //gaussien  - not worked
}
double ann::derivative_of_sigmoid_func(double val){
    //return (sigmoid_func(val) * (1 - sigmoid_func(val)));                     //sigmoid   - good - u:0.8 is the best
    //return (1 - tanh(val)*tanh(val));                                         //tanh      - not worked
    //return 1;                                                                 //identity  - not properly worked
    //return (1 / (1 + val*val));                                               //atan      - not bad but slower
    //return (1 / (1 + exp(-val)));                                             //softplus  - good but slower
    /***************RELU**********/                                             //ReLU      - very good training but test is bad u:0.000132 is the best
    if(val < 0) return 0.01;
    else return 1;
    /*****************************/
    //return (1 + exp(-val) + val*exp(-val))/((1 + exp(-val))*(1 + exp(-val))); //swish     - not bad but not good
    //return -2*val*sigmoid_func(val);                                          //gaussien  - not worked
}

