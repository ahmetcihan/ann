#include "mainwindow.h"

double MainWindow::sigmoid_func(double val){
    //return (1 / (1 + exp(-val)));   //sigmoid   - good
    //return tanh(val);             //tanh      - not worked
    //return val;                   //identity  - not properly worked
    //return atan(val);             //atan      - not bad but slower
    //return (log(1+exp(val)));     //softplus  - good but slower
    /***************RELU**********/ //ReLU      - very good
    if(val < 0) return 0;
    if(val >= 0) return val;
    /*****************************/
}
double MainWindow::derivative_of_sigmoid_func(double val){
    //return (sigmoid_func(val) * (1 - sigmoid_func(val)));       //sigmoid   - good
    //return (1 - tanh(val)*tanh(val));                         //tanh      - not worked
    //return 1;                                                 //identity  - not properly worked
    //return (1 / (1 + val*val));                               //atan      - not bad but slower
    //return (1 / (1 + exp(-val)));                             //softplus  - good but slower
    /***************RELU**********/                             //ReLU      - very good
    if(val < 0) return 0;
    if(val >= 0) return 1;
    /*****************************/
}

