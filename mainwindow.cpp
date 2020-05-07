#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    double input[2];
    double pre_output;
    double output;
    double w[9];
    double A,B,C;
    double sigmoid_A,sigmoid_B,sigmoid_C;
    double error;
    double desired_output;

    input[0] = 0;
    input[1] = 0;
    desired_output = 0;

    qDebug() << " input[0] : " << input[0] << " input[1] : " << input[1];

    for(u8 i = 0; i < 9; i++){
        w[i] = 0.1 * i + 0.1;
        qDebug() << QString("w[%1] : ").arg(i) << w[i];
    }

    A = input[0]*w[0] + input[1]*w[1];
    B = input[0]*w[2] + input[1]*w[3];
    C = input[0]*w[4] + input[1]*w[5];

    sigmoid_A = sigmoid_func(A);
    sigmoid_B = sigmoid_func(B);
    sigmoid_C = sigmoid_func(C);

    pre_output = sigmoid_A*w[6] + sigmoid_B*w[7] + sigmoid_C*w[8];

    output = sigmoid_func(pre_output);
    error = desired_output - output;

    qDebug() << " output : " << output << "error" << error;

    double A_change_weight;
    double B_change_weight;
    double C_change_weight;
    double w_change_weight[9];

    A_change_weight = output / w[6];
    A_change_weight = A_change_weight * derivative_of_sigmoid_func(A);
    w_change_weight[6] = A_change_weight / sigmoid_A;
    w[6] = w[6] + w_change_weight[6];

    B_change_weight = output / w[7];
    B_change_weight = B_change_weight * derivative_of_sigmoid_func(B);
    w_change_weight[7] = B_change_weight / sigmoid_B;
    w[7] = w[7] + w_change_weight[7];

    C_change_weight = output / w[8];
    C_change_weight = C_change_weight * derivative_of_sigmoid_func(C);
    w_change_weight[8] = C_change_weight / sigmoid_C;
    w[8] = w[8] + w_change_weight[8];

    qDebug() << "w[6]" << w[6];
    qDebug() << "w[7]" << w[7];
    qDebug() << "w[8]" << w[8];

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
