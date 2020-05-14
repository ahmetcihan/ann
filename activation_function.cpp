#include "mainwindow.h"

double MainWindow::sigmoid_func(double val){
    //return (1 / (1 + exp(-val))); //sigmoid
    //return tanh(val);     //not worked
    //return val;           //not worked
    //return atan(val);     //not worked
    //return (log(1+exp(val)));//not worked
    if(val < 0) return 0;
    if(val >= 0) return val;
}
double MainWindow::derivative_of_sigmoid_func(double val){
    //return (sigmoid_func(val) * (1 - sigmoid_func(val))); //sigmoid
    //return (1 - tanh(val)*tanh(val));   //not worked
    //return 1;                     //not worked
    //return (1 / (1 + val*val));   //not worked
    //return (1 / (1 + exp(-val)));   //not worked
    if(val < 0) return 0;
    if(val >= 0) return 1;
}

