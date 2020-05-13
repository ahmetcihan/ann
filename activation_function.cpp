#include "mainwindow.h"

double MainWindow::sigmoid_func(double val){
    return (1 / (1 + exp(-val)));
}
double MainWindow::derivative_of_sigmoid_func(double val){
    return (sigmoid_func(val) * (1 - sigmoid_func(val)));

}

