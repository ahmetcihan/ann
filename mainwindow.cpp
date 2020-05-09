#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    double input1[4] = {0,0,1,1};
    double input2[4] = {0,1,0,1};
    double desired_output[4] = {0,1,1,0};
    double calculated_output[4] = {0,0,0,0};
    double Y_in,Y_out,delta_Y;
    double w[9][4],delta_w[9][4];
    double A_in,B_in,C_in;
    double A_out,B_out,C_out;
    double delta_A,delta_B,delta_C;
    double error;
    u32 epoch = 2000;

    for(u8 i = 0; i < 4; i++){
        qDebug() << " input1 : " << input1[i] << " input2 : " << input2[i] << "output : " << desired_output[i];
    }
    for(u8 k = 0; k < 4; k++){
        for(u8 i = 0; i < 9; i++){
            w[i][k] = 0.1 * i + 0.1;
        }
    }

    for(u32 era = 0; era < epoch; era++){
        for(u8 k = 0; k < 4; k++){

            A_in = input1[k]*w[0][k] + input2[k]*w[1][k];
            B_in = input1[k]*w[2][k] + input2[k]*w[3][k];
            C_in = input1[k]*w[4][k] + input2[k]*w[5][k];

            A_out = sigmoid_func(A_in);
            B_out = sigmoid_func(B_in);
            C_out = sigmoid_func(C_in);

            Y_in = A_out*w[6][k] + B_out*w[7][k] + C_out*w[8][k];
            Y_out = sigmoid_func(Y_in);
            error = desired_output[k] - Y_out;
            calculated_output[k] = Y_out;

            delta_Y = derivative_of_sigmoid_func(Y_in) * error;
            delta_A = delta_Y/w[6][k] * derivative_of_sigmoid_func(A_in);
            delta_B = delta_Y/w[7][k] * derivative_of_sigmoid_func(B_in);
            delta_C = delta_Y/w[8][k] * derivative_of_sigmoid_func(C_in);

            if (input1[k] == 0){
                delta_w[0][k] = 0;
                delta_w[2][k] = 0;
                delta_w[4][k] = 0;
            }
            else{
                delta_w[0][k] = delta_A/input1[k];
                delta_w[2][k] = delta_B/input1[k];
                delta_w[4][k] = delta_C/input1[k];
            }

            if (input2[k] == 0){
                delta_w[1][k] = 0;
                delta_w[3][k] = 0;
                delta_w[5][k] = 0;
            }
            else{
                delta_w[1][k] = delta_A/input2[k];
                delta_w[3][k] = delta_B/input2[k];
                delta_w[5][k] = delta_C/input2[k];
            }

            if(A_out == 0){
                delta_w[6][k] = 0;
            }
            else{
                delta_w[6][k] = delta_Y / A_out;
            }

            if(B_out == 0){
                delta_w[7][k] = 0;
            }
            else{
                delta_w[7][k] = delta_Y / B_out;
            }

            if(C_out == 0){
                delta_w[8][k] = 0;
            }
            else{
                delta_w[8][k] = delta_Y / C_out;
            }

            for(u8 i = 0; i < 9; i++){
                w[i][k] = w[i][k] + delta_w[i][k];
            }

        }
    }
    for(u8 k = 0; k < 4; k++){
        for(u8 i = 0; i < 9; i++){
            qDebug() << QString("w[%1]").arg(i) << w[i][k];
        }

    }

    for(u8 k = 0; k < 4; k++){
        qDebug() << "output : " << calculated_output[k];
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
