#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    input_2_4_2[0] = 10.2;
    input_2_4_2[1] = -0.3;
    desired_output_2_4_2[0] = 0.59;
    desired_output_2_4_2[1] = 0.44;
    for(u8 i = 0; i < 2; i++){
        for(u8 j = 0; j < 4; j++){
            _2_4_2_input_to_hidden_weight[i][j] = 0.1;
        }
    }
    for(u8 i = 0; i < 4; i++){
        for(u8 j = 0; j < 2; j++){
            _2_4_2_hidden_to_output_weight[i][j] = 0.1;
        }
    }

    _2_4_2_ann_train(input_2_4_2,desired_output_2_4_2,270, _2_4_2_input_to_hidden_weight, _2_4_2_hidden_to_output_weight);

    //_2_4_2_ann_test(input_2_4_2,_2_4_2_weights);
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
void MainWindow::_2_4_2_ann_test(double *input, double *weight){
    double A_in,B_in,C_in,D_in;
    double A_out,B_out,C_out,D_out;
    double Y1_in,Y1_out;
    double Y2_in,Y2_out;

    A_in = input[0]*weight[0] + input[1]*weight[1];
    B_in = input[0]*weight[2] + input[1]*weight[3];
    C_in = input[0]*weight[4] + input[1]*weight[5];
    D_in = input[0]*weight[6] + input[1]*weight[7];

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

void MainWindow::_2_4_2_ann_train(double *input, double *desired_output, u32 epoch, double input_to_hidden_weight[2][4], double hidden_to_output_weight[4][2]){
#define INPUT_COUNT 2
#define HIDDEN_COUNT 4
#define OUTPUT_COUNT 2

    double calculated_output[OUTPUT_COUNT];
    double Y_in[OUTPUT_COUNT];
    double Y_out[OUTPUT_COUNT];
    double delta_Y[OUTPUT_COUNT];
    double error[OUTPUT_COUNT];

    double hidden_in[HIDDEN_COUNT];
    double hidden_out[HIDDEN_COUNT];
    double delta_hidden[HIDDEN_COUNT];

    double delta_input_to_hidden_weight[2][4];
    double delta_hidden_to_output_weight[4][2];

    for(u8 i = 0; i < INPUT_COUNT; i++){
        qDebug() << QString("input[%1] : ").arg(i) << input[i];
    }
    for(u8 i = 0; i < OUTPUT_COUNT; i++){
        qDebug() << QString("desired_output[%1] : ").arg(i) << desired_output[i];
    }

    for(u32 era = 0; era < epoch; era++){
        hidden_in[0] = input[0]*input_to_hidden_weight[0][0] + input[1]*input_to_hidden_weight[1][0];
        hidden_in[1] = input[0]*input_to_hidden_weight[0][1] + input[1]*input_to_hidden_weight[1][1];
        hidden_in[2] = input[0]*input_to_hidden_weight[0][2] + input[1]*input_to_hidden_weight[1][2];
        hidden_in[3] = input[0]*input_to_hidden_weight[0][3] + input[1]*input_to_hidden_weight[1][3];

        hidden_out[0] = sigmoid_func(hidden_in[0]);
        hidden_out[1] = sigmoid_func(hidden_in[1]);
        hidden_out[2] = sigmoid_func(hidden_in[2]);
        hidden_out[3] = sigmoid_func(hidden_in[3]);

        Y_in[0] = hidden_out[0]*hidden_to_output_weight[0][0] + hidden_out[1]*hidden_to_output_weight[1][0] +
                hidden_out[2]*hidden_to_output_weight[2][0] + hidden_out[3]*hidden_to_output_weight[3][0];
        Y_out[0] = sigmoid_func(Y_in[0]);
        error[0] = desired_output[0] - Y_out[0];
        calculated_output[0] = Y_out[0];

        Y_in[1] = hidden_out[0]*hidden_to_output_weight[0][1] + hidden_out[1]*hidden_to_output_weight[1][1] +
                hidden_out[2]*hidden_to_output_weight[2][1] + hidden_out[3]*hidden_to_output_weight[3][1];
        Y_out[1] = sigmoid_func(Y_in[1]);
        error[1] = desired_output[1] - Y_out[1];
        calculated_output[1] = Y_out[1];

        delta_Y[0] = derivative_of_sigmoid_func(Y_in[0]) * error[0];
        delta_Y[1] = derivative_of_sigmoid_func(Y_in[1]) * error[1];

        delta_hidden[0] = (delta_Y[0]/hidden_to_output_weight[0][0] + delta_Y[1]/hidden_to_output_weight[0][1]) * derivative_of_sigmoid_func(hidden_in[0]);
        delta_hidden[1] = (delta_Y[0]/hidden_to_output_weight[1][0] + delta_Y[1]/hidden_to_output_weight[1][1]) * derivative_of_sigmoid_func(hidden_in[1]);
        delta_hidden[2] = (delta_Y[0]/hidden_to_output_weight[2][0] + delta_Y[1]/hidden_to_output_weight[2][1]) * derivative_of_sigmoid_func(hidden_in[2]);
        delta_hidden[3] = (delta_Y[0]/hidden_to_output_weight[3][0] + delta_Y[1]/hidden_to_output_weight[3][1]) * derivative_of_sigmoid_func(hidden_in[3]);

        if(hidden_out[0] == 0){
            delta_hidden_to_output_weight[0][0] = 0;
            delta_hidden_to_output_weight[0][1] = 0;
        }
        else{
            delta_hidden_to_output_weight[0][0] = delta_Y[0] / hidden_out[0];
            delta_hidden_to_output_weight[0][1] = delta_Y[1] / hidden_out[0];
        }

        if(hidden_out[1] == 0){
            delta_hidden_to_output_weight[1][0] = 0;
            delta_hidden_to_output_weight[1][1] = 0;
        }
        else{
            delta_hidden_to_output_weight[1][0] = delta_Y[0] / hidden_out[1];
            delta_hidden_to_output_weight[1][1] = delta_Y[1] / hidden_out[1];
        }

        if(hidden_out[2] == 0){
            delta_hidden_to_output_weight[2][0] = 0;
            delta_hidden_to_output_weight[1][1] = 0;
        }
        else{
            delta_hidden_to_output_weight[2][0] = delta_Y[0] / hidden_out[2];
            delta_hidden_to_output_weight[2][1] = delta_Y[1] / hidden_out[2];
        }

        if(hidden_out[3] == 0){
            delta_hidden_to_output_weight[3][0] = 0;
            delta_hidden_to_output_weight[3][1] = 0;
        }
        else{
            delta_hidden_to_output_weight[3][0] = delta_Y[0] / hidden_out[3];
            delta_hidden_to_output_weight[3][1] = delta_Y[1] / hidden_out[3];
        }

        if (input[0] == 0){
            delta_input_to_hidden_weight[0][0] = 0;
            delta_input_to_hidden_weight[0][1] = 0;
            delta_input_to_hidden_weight[0][2] = 0;
            delta_input_to_hidden_weight[0][3] = 0;
        }
        else{
            delta_input_to_hidden_weight[0][0] = delta_hidden[0]/input[0];
            delta_input_to_hidden_weight[0][1] = delta_hidden[1]/input[0];
            delta_input_to_hidden_weight[0][2] = delta_hidden[2]/input[0];
            delta_input_to_hidden_weight[0][3] = delta_hidden[3]/input[0];
        }

        if (input[1] == 0){
            delta_input_to_hidden_weight[1][0] = 0;
            delta_input_to_hidden_weight[1][1] = 0;
            delta_input_to_hidden_weight[1][2] = 0;
            delta_input_to_hidden_weight[1][3] = 0;
        }
        else{
            delta_input_to_hidden_weight[1][0] = delta_hidden[0]/input[1];
            delta_input_to_hidden_weight[1][1] = delta_hidden[1]/input[1];
            delta_input_to_hidden_weight[1][2] = delta_hidden[2]/input[1];
            delta_input_to_hidden_weight[1][3] = delta_hidden[3]/input[1];
        }


        for(u8 i = 0; i < INPUT_COUNT; i++){
            for(u8 j = 0; j < HIDDEN_COUNT; j++){
                input_to_hidden_weight[i][j] = input_to_hidden_weight[i][j] + delta_input_to_hidden_weight[i][j];
            }
        }
        for(u8 i = 0; i < HIDDEN_COUNT; i++){
            for(u8 j = 0; j < OUTPUT_COUNT; j++){
                hidden_to_output_weight[i][j] = hidden_to_output_weight[i][j] + delta_hidden_to_output_weight[i][j];
            }
        }
    }
    for(u8 i = 0; i < OUTPUT_COUNT; i++){
        qDebug() << QString("output[%1] : ").arg(i) << calculated_output[i];
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
