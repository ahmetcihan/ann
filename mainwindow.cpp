#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    double input[2];
    double Y_in,Y_out,delta_Y;
    double w[9],delta_w[9];
    double A_in,B_in,C_in;
    double A_out,B_out,C_out;
    double delta_A,delta_B,delta_C;
    double error;
    double desired_output;

    input[0] = 0;
    input[1] = 0;
    desired_output = 0;

    qDebug() << " input[0] : " << input[0] << " input[1] : " << input[1] << "output : " << desired_output;

    for(u8 i = 0; i < 9; i++){
        w[i] = 0.1 * i + 0.1;
    }

    A_in = input[0]*w[0] + input[1]*w[1];
    B_in = input[0]*w[2] + input[1]*w[3];
    C_in = input[0]*w[4] + input[1]*w[5];

    A_out = sigmoid_func(A_in);
    B_out = sigmoid_func(B_in);
    C_out = sigmoid_func(C_in);

    Y_in = A_out*w[6] + B_out*w[7] + C_out*w[8];
    Y_out = sigmoid_func(Y_in);
    error = desired_output - Y_out;

    delta_Y = derivative_of_sigmoid_func(Y_in) * error;
    delta_A = delta_Y/w[6] * derivative_of_sigmoid_func(A_in);
    delta_B = delta_Y/w[7] * derivative_of_sigmoid_func(B_in);
    delta_C = delta_Y/w[8] * derivative_of_sigmoid_func(C_in);

    qDebug() << "error :" << error;

    if (input[0] == 0){
        delta_w[0] = 0;
        delta_w[2] = 0;
        delta_w[4] = 0;
    }
    else{
        delta_w[0] = delta_A/input[0];
        delta_w[2] = delta_B/input[0];
        delta_w[4] = delta_C/input[0];
    }

    if (input[1] == 0){
        delta_w[1] = 0;
        delta_w[3] = 0;
        delta_w[5] = 0;
    }
    else{
        delta_w[1] = delta_A/input[1];
        delta_w[3] = delta_B/input[1];
        delta_w[5] = delta_C/input[1];
    }

    if(A_out == 0){
        delta_w[6] = 0;
    }
    else{
        delta_w[6] = delta_Y / A_out;
    }

    if(B_out == 0){
        delta_w[7] = 0;
    }
    else{
        delta_w[7] = delta_Y / B_out;
    }

    if(C_out == 0){
        delta_w[8] = 0;
    }
    else{
        delta_w[8] = delta_Y / B_out;
    }

    for(u8 i = 0; i < 9; i++){
        w[i] = w[i] + delta_w[i];
        qDebug() << QString("w[%1]").arg(i) << w[i];
    }

}
double MainWindow::sigmoid_func(double val){
    return (1 / (1 + exp(-val)));
}
double MainWindow::derivative_of_sigmoid_func(double val){
    return (sigmoid_func(val) * (1 - sigmoid_func(val)));
}

MainWindow::~MainWindow()
{
    delete ui;
}
