#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    //xor_ann();

    for(u8 i = 0; i < 16; i++){
        _2_4_2_weights[i] =  0.1 * i + 0.1;
    }

    qDebug() << "**********initial weights**************";
    for(u8 i = 0; i < 16; i++){
        qDebug() << QString("w[%1]").arg(i) << _2_4_2_weights[i];
    }
    qDebug() << "***************************************";

    //_2_3_1_ann_train(0.2,0.3,0.6,17000,_2_3_1_weights);
    _2_4_2_ann_train(10.2,-0.3,0.65,0.7,270,_2_4_2_weights);

    qDebug() << "************final weights**************";

    for(u8 i = 0; i < 16; i++){
        qDebug() << QString("w[%1]").arg(i) << _2_4_2_weights[i];
    }
    qDebug() << "***************************************";

    //_2_3_1_ann_test(0.2,0.3,_2_3_1_weights);
    //_2_4_2_ann_test(10.2,-0.3,_2_4_2_weights);
}
void MainWindow::_2_3_1_ann_test(double input1, double input2, double *weight){
    double A_in,B_in,C_in;
    double A_out,B_out,C_out;
    double Y_in,Y_out;

    A_in = input1*weight[0] + input2*weight[1];
    B_in = input1*weight[2] + input2*weight[3];
    C_in = input1*weight[4] + input2*weight[5];

    A_out = sigmoid_func(A_in);
    B_out = sigmoid_func(B_in);
    C_out = sigmoid_func(C_in);

    Y_in = A_out*weight[6] + B_out*weight[7] + C_out*weight[8];
    Y_out = sigmoid_func(Y_in);

    qDebug() << "tested input1 :" << input1 << "input2 :" << input2 << "output :" << Y_out ;
}
void MainWindow::_2_4_2_ann_test(double input1, double input2, double *weight){
    double A_in,B_in,C_in,D_in;
    double A_out,B_out,C_out,D_out;
    double Y1_in,Y1_out;
    double Y2_in,Y2_out;

    A_in = input1*weight[0] + input2*weight[1];
    B_in = input1*weight[2] + input2*weight[3];
    C_in = input1*weight[4] + input2*weight[5];
    D_in = input1*weight[6] + input2*weight[7];

    A_out = sigmoid_func(A_in);
    B_out = sigmoid_func(B_in);
    C_out = sigmoid_func(C_in);
    D_out = sigmoid_func(D_in);

    Y1_in = A_out*weight[8] + B_out*weight[9] + C_out*weight[10] + D_out*weight[11];
    Y1_out = sigmoid_func(Y1_in);
    Y2_in = A_out*weight[12] + B_out*weight[13] + C_out*weight[14] + D_out*weight[15];
    Y2_out = sigmoid_func(Y2_in);

    qDebug() << "output1 : " << Y1_out << "output2 : " << Y2_out;
}

void MainWindow::_2_4_2_ann_train(double input1,double input2, double desired_output1,  double desired_output2, u32 epoch, double *weight){
    {
        double calculated_output1,calculated_output2;
        double Y1_in,Y1_out,delta_Y1;
        double Y2_in,Y2_out,delta_Y2;
        double delta_w[16];
        double A_in,B_in,C_in,D_in;
        double A_out,B_out,C_out,D_out;
        double delta_A,delta_B,delta_C,delta_D;
        double error1,error2;

        qDebug() << "input1 : " << input1 << " input2 : " << input2 << "output1 : " << desired_output1 << "output2 : " << desired_output2;

        for(u32 era = 0; era < epoch; era++){
            A_in = input1*weight[0] + input2*weight[1];
            B_in = input1*weight[2] + input2*weight[3];
            C_in = input1*weight[4] + input2*weight[5];
            D_in = input1*weight[6] + input2*weight[7];

            A_out = sigmoid_func(A_in);
            B_out = sigmoid_func(B_in);
            C_out = sigmoid_func(C_in);
            D_out = sigmoid_func(D_in);

            Y1_in = A_out*weight[8] + B_out*weight[9] + C_out*weight[10] + D_out*weight[11];
            Y1_out = sigmoid_func(Y1_in);
            error1 = desired_output1 - Y1_out;
            calculated_output1 = Y1_out;

            Y2_in = A_out*weight[12] + B_out*weight[13] + C_out*weight[14] + D_out*weight[15];
            Y2_out = sigmoid_func(Y2_in);
            error2 = desired_output2 - Y2_out;
            calculated_output2 = Y2_out;

            delta_Y1 = derivative_of_sigmoid_func(Y1_in) * error1;
            delta_Y2 = derivative_of_sigmoid_func(Y2_in) * error2;

            delta_A = (delta_Y1/weight[8] + delta_Y2/weight[12]) * derivative_of_sigmoid_func(A_in);
            delta_B = (delta_Y1/weight[9] + delta_Y2/weight[13]) * derivative_of_sigmoid_func(B_in);
            delta_C = (delta_Y1/weight[10] + delta_Y2/weight[14]) * derivative_of_sigmoid_func(C_in);
            delta_D = (delta_Y1/weight[11] + delta_Y2/weight[15]) * derivative_of_sigmoid_func(D_in);

            if(A_out == 0){
                delta_w[8] = 0;
                delta_w[12] = 0;
            }
            else{
                delta_w[8] = delta_Y1 / A_out;
                delta_w[12] = delta_Y2 / A_out;
            }

            if(B_out == 0){
                delta_w[9] = 0;
                delta_w[13] = 0;
            }
            else{
                delta_w[9] = delta_Y1 / B_out;
                delta_w[13] = delta_Y2 / B_out;
            }

            if(C_out == 0){
                delta_w[10] = 0;
                delta_w[14] = 0;
            }
            else{
                delta_w[10] = delta_Y1 / C_out;
                delta_w[14] = delta_Y2 / C_out;
            }

            if(D_out == 0){
                delta_w[11] = 0;
                delta_w[15] = 0;
            }
            else{
                delta_w[11] = delta_Y1 / D_out;
                delta_w[15] = delta_Y2 / D_out;
            }

            if (input1 == 0){
                delta_w[0] = 0;
                delta_w[2] = 0;
                delta_w[4] = 0;
                delta_w[6] = 0;
            }
            else{
                delta_w[0] = delta_A/input1;
                delta_w[2] = delta_B/input1;
                delta_w[4] = delta_C/input1;
                delta_w[6] = delta_D/input1;
            }

            if (input2 == 0){
                delta_w[1] = 0;
                delta_w[3] = 0;
                delta_w[5] = 0;
                delta_w[7] = 0;
            }
            else{
                delta_w[1] = delta_A/input2;
                delta_w[3] = delta_B/input2;
                delta_w[5] = delta_C/input2;
                delta_w[7] = delta_D/input2;
            }


            for(u8 i = 0; i < 16; i++){
                weight[i] = weight[i] + delta_w[i];
            }
        }
        qDebug() << "output1 : " << calculated_output1 << "output2 : " << calculated_output2;

    }
}

void MainWindow::_2_3_1_ann_train(double input1, double input2, double desired_output, u32 epoch, double *weight){
    double calculated_output;
    double Y_in,Y_out,delta_Y;
    double delta_w[9];
    double A_in,B_in,C_in;
    double A_out,B_out,C_out;
    double delta_A,delta_B,delta_C;
    double error;

    qDebug() << "input1 : " << input1 << " input2 : " << input2 << "output : " << desired_output;

    for(u32 era = 0; era < epoch; era++){
        A_in = input1*weight[0] + input2*weight[1];
        B_in = input1*weight[2] + input2*weight[3];
        C_in = input1*weight[4] + input2*weight[5];

        A_out = sigmoid_func(A_in);
        B_out = sigmoid_func(B_in);
        C_out = sigmoid_func(C_in);

        Y_in = A_out*weight[6] + B_out*weight[7] + C_out*weight[8];
        Y_out = sigmoid_func(Y_in);
        error = desired_output - Y_out;
        calculated_output = Y_out;

        delta_Y = derivative_of_sigmoid_func(Y_in) * error;
        delta_A = delta_Y/weight[6] * derivative_of_sigmoid_func(A_in);
        delta_B = delta_Y/weight[7] * derivative_of_sigmoid_func(B_in);
        delta_C = delta_Y/weight[8] * derivative_of_sigmoid_func(C_in);

        if (input1 == 0){
            delta_w[0] = 0;
            delta_w[2] = 0;
            delta_w[4] = 0;
        }
        else{
            delta_w[0] = delta_A/input1;
            delta_w[2] = delta_B/input1;
            delta_w[4] = delta_C/input1;
        }

        if (input2 == 0){
            delta_w[1] = 0;
            delta_w[3] = 0;
            delta_w[5] = 0;
        }
        else{
            delta_w[1] = delta_A/input2;
            delta_w[3] = delta_B/input2;
            delta_w[5] = delta_C/input2;
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
            delta_w[8] = delta_Y / C_out;
        }

        for(u8 i = 0; i < 9; i++){
            weight[i] = weight[i] + delta_w[i];
        }
    }
    qDebug() << "output : " << calculated_output;

}

void MainWindow::xor_ann(void){
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
