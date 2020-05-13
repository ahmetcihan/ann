#include "mainwindow.h"

void MainWindow::_2_5_2_ann_train(void){
    double input1[4] = {0,0,1,1};
    double input2[4] = {0,1,0,1};
    double desired_output_1[4] = {0,1,1,0};
    double desired_output_2[4] = {1,0,0,1};

    double calculated_output_1[4] = {0,0,0,0};
    double calculated_output_2[4] = {0,0,0,0};
    double Y_in_1,Y_out_1;
    double Y_in_2,Y_out_2;
    double w_input_to_hidden[2][5];
    double w_hidden_to_output[5][2];

    double A_in,B_in,C_in,D_in,E_in;
    double A_out,B_out,C_out,D_out,E_out;
    double output_error_1;
    double output_error_2;

    double bias1 = 0.1;
    double bias2 = 0.2;
    double bias3 = 0.3;
    double bias4 = 0.4;
    double bias5 = 0.5;
    double bias_output_1 = 0.6;
    double bias_output_2 = 0.6;

    double error1,error2,error3,error4,error5;

    u32 epoch = 200000;

    double global_error_1,global_error_2;

    for(u8 i = 0; i < 4; i++){
        qDebug() << " input1 : " << input1[i] << " input2 : " << input2[i] << "output1 : " << desired_output_1[i] << "output2 : " << desired_output_2[i];
    }

    w_input_to_hidden[0][0] = 0.1;
    w_input_to_hidden[0][1] = 0.1;
    w_input_to_hidden[0][2] = 0.1;
    w_input_to_hidden[0][3] = 0.1;
    w_input_to_hidden[0][4] = 0.1;

    w_input_to_hidden[1][0] = 0.1;
    w_input_to_hidden[1][1] = 0.1;
    w_input_to_hidden[1][2] = 0.1;
    w_input_to_hidden[1][3] = 0.1;
    w_input_to_hidden[1][4] = 0.1;

    w_hidden_to_output[0][0] = 0.1;
    w_hidden_to_output[1][0] = 0.1;
    w_hidden_to_output[2][0] = 0.1;
    w_hidden_to_output[3][0] = 0.1;
    w_hidden_to_output[4][0] = 0.1;

    w_hidden_to_output[0][1] = 0.1;
    w_hidden_to_output[1][1] = 0.1;
    w_hidden_to_output[2][1] = 0.1;
    w_hidden_to_output[3][1] = 0.1;
    w_hidden_to_output[4][1] = 0.1;

    for(u32 era = 0; era < epoch; era++){
        for(u8 k = 0; k < 4; k++){

            A_in = input1[k]*w_input_to_hidden[0][0] + input2[k]*w_input_to_hidden[1][0] + bias1;
            B_in = input1[k]*w_input_to_hidden[0][1] + input2[k]*w_input_to_hidden[1][1] + bias2;
            C_in = input1[k]*w_input_to_hidden[0][2] + input2[k]*w_input_to_hidden[1][2] + bias3;
            D_in = input1[k]*w_input_to_hidden[0][3] + input2[k]*w_input_to_hidden[1][3] + bias4;
            E_in = input1[k]*w_input_to_hidden[0][4] + input2[k]*w_input_to_hidden[1][4] + bias5;

            A_out = sigmoid_func(A_in);
            B_out = sigmoid_func(B_in);
            C_out = sigmoid_func(C_in);
            D_out = sigmoid_func(D_in);
            E_out = sigmoid_func(E_in);

            Y_in_1 = A_out*w_hidden_to_output[0][0] + B_out*w_hidden_to_output[1][0] + C_out*w_hidden_to_output[2][0] +
                   D_out*w_hidden_to_output[3][0] + E_out*w_hidden_to_output[4][0] +bias_output_1;

            Y_out_1 = sigmoid_func(Y_in_1);
            output_error_1 = desired_output_1[k] - Y_out_1;

            calculated_output_1[k] = Y_out_1;

            Y_in_2 = A_out*w_hidden_to_output[0][1] + B_out*w_hidden_to_output[1][1] + C_out*w_hidden_to_output[2][1] +
                   D_out*w_hidden_to_output[3][1] + E_out*w_hidden_to_output[4][1] +bias_output_2;

            Y_out_2 = sigmoid_func(Y_in_2);
            output_error_2 = desired_output_2[k] - Y_out_2;

            calculated_output_2[k] = Y_out_2;


            global_error_1 = derivative_of_sigmoid_func(Y_in_1) * output_error_1;
            global_error_2 = derivative_of_sigmoid_func(Y_in_2) * output_error_2;

            w_hidden_to_output[0][0] += global_error_1 * A_out;
            w_hidden_to_output[1][0] += global_error_1 * B_out;
            w_hidden_to_output[2][0] += global_error_1 * C_out;
            w_hidden_to_output[3][0] += global_error_1 * D_out;
            w_hidden_to_output[4][0] += global_error_1 * E_out;
            bias_output_1 += global_error_1;

            w_hidden_to_output[0][1] += global_error_2 * A_out;
            w_hidden_to_output[1][1] += global_error_2 * B_out;
            w_hidden_to_output[2][1] += global_error_2 * C_out;
            w_hidden_to_output[3][1] += global_error_2 * D_out;
            w_hidden_to_output[4][1] += global_error_2 * E_out;
            bias_output_2 += global_error_2;


            error1 = derivative_of_sigmoid_func(A_in) * global_error_1 * w_hidden_to_output[0][0] +
                    derivative_of_sigmoid_func(A_in) * global_error_2 * w_hidden_to_output[0][1];
            error2 = derivative_of_sigmoid_func(B_in) * global_error_1 * w_hidden_to_output[1][0] +
                    derivative_of_sigmoid_func(B_in) * global_error_2 * w_hidden_to_output[1][1];
            error3 = derivative_of_sigmoid_func(C_in) * global_error_1 * w_hidden_to_output[2][0] +
                    derivative_of_sigmoid_func(C_in) * global_error_2 * w_hidden_to_output[2][1];
            error4 = derivative_of_sigmoid_func(D_in) * global_error_1 * w_hidden_to_output[3][0] +
                    derivative_of_sigmoid_func(D_in) * global_error_2 * w_hidden_to_output[3][1];
            error5 = derivative_of_sigmoid_func(E_in) * global_error_1 * w_hidden_to_output[4][0] +
                    derivative_of_sigmoid_func(E_in) * global_error_2 * w_hidden_to_output[4][1];

            w_input_to_hidden[0][0] += error1 * input1[k];
            w_input_to_hidden[0][1] += error2 * input1[k];
            w_input_to_hidden[0][2] += error3 * input1[k];
            w_input_to_hidden[0][3] += error4 * input1[k];
            w_input_to_hidden[0][4] += error5 * input1[k];

            w_input_to_hidden[1][0] += error1 * input2[k];
            w_input_to_hidden[1][1] += error2 * input2[k];
            w_input_to_hidden[1][2] += error3 * input2[k];
            w_input_to_hidden[1][3] += error4 * input2[k];
            w_input_to_hidden[1][4] += error5 * input2[k];
            bias1 +=error1;
            bias2 +=error2;
            bias3 +=error3;
            bias4 +=error4;
            bias5 +=error5;
        }
    }

    for(u8 i = 0; i < 2; i++){
        for(u8 j = 0; j < 5; j++){
            qDebug() << "in_to_h_w[i][j] : " << w_input_to_hidden[i][j];
        }
    }
    for(u8 i = 0; i < 5; i++){
        for(u8 j = 0; j < 2; j++){
            qDebug() << "h_to_o_w[i][j] : " << w_hidden_to_output[i][j];
        }
    }
    qDebug() << "bias1" << bias1;
    qDebug() << "bias2" << bias2;
    qDebug() << "bias3" << bias3;
    qDebug() << "bias4" << bias4;
    qDebug() << "bias5" << bias5;
    qDebug() << "bias_output_1" << bias_output_1;
    qDebug() << "bias_output_2" << bias_output_2;
    for(u8 k = 0; k < 4; k++){
        qDebug() << "output1 : " << calculated_output_1[k] << "output2 : " << calculated_output_2[k];
    }
}
void MainWindow::_2_5_1_ann_train(void){
    double input1[4] = {0,0,1,1};
    double input2[4] = {0,1,0,1};
    double desired_output[4] = {0,1,1,0};
    double calculated_output[4] = {0,0,0,0};
    double Y_in,Y_out;
    double w_input_to_hidden[2][5];
    double w_hidden_to_output[5];

    double A_in,B_in,C_in,D_in,E_in;
    double A_out,B_out,C_out,D_out,E_out;
    double output_error;

    double bias1 = 0.1;
    double bias2 = 0.2;
    double bias3 = 0.3;
    double bias4 = 0.4;
    double bias5 = 0.5;
    double bias_output = 0.6;

    double error1,error2,error3,error4,error5;

    u32 epoch = 200000;

    double global_error;

    for(u8 i = 0; i < 4; i++){
        qDebug() << " input1 : " << input1[i] << " input2 : " << input2[i] << "output : " << desired_output[i];
    }

    w_input_to_hidden[0][0] = 0.1;
    w_input_to_hidden[0][1] = 0.1;
    w_input_to_hidden[0][2] = 0.1;
    w_input_to_hidden[0][3] = 0.1;
    w_input_to_hidden[0][4] = 0.1;

    w_input_to_hidden[1][0] = 0.1;
    w_input_to_hidden[1][1] = 0.1;
    w_input_to_hidden[1][2] = 0.1;
    w_input_to_hidden[1][3] = 0.1;
    w_input_to_hidden[1][4] = 0.1;

    w_hidden_to_output[0] = 0.1;
    w_hidden_to_output[1] = 0.1;
    w_hidden_to_output[2] = 0.1;
    w_hidden_to_output[3] = 0.1;
    w_hidden_to_output[4] = 0.1;

    for(u32 era = 0; era < epoch; era++){
        for(u8 k = 0; k < 4; k++){

            A_in = input1[k]*w_input_to_hidden[0][0] + input2[k]*w_input_to_hidden[1][0] + bias1;
            B_in = input1[k]*w_input_to_hidden[0][1] + input2[k]*w_input_to_hidden[1][1] + bias2;
            C_in = input1[k]*w_input_to_hidden[0][2] + input2[k]*w_input_to_hidden[1][2] + bias3;
            D_in = input1[k]*w_input_to_hidden[0][3] + input2[k]*w_input_to_hidden[1][3] + bias4;
            E_in = input1[k]*w_input_to_hidden[0][4] + input2[k]*w_input_to_hidden[1][4] + bias5;

            A_out = sigmoid_func(A_in);
            B_out = sigmoid_func(B_in);
            C_out = sigmoid_func(C_in);
            D_out = sigmoid_func(D_in);
            E_out = sigmoid_func(E_in);

            Y_in = A_out*w_hidden_to_output[0] + B_out*w_hidden_to_output[1] + C_out*w_hidden_to_output[2] +
                   D_out*w_hidden_to_output[3] + E_out*w_hidden_to_output[4] +bias_output;
            Y_out = sigmoid_func(Y_in);
            output_error = desired_output[k] - Y_out;

            calculated_output[k] = Y_out;

            global_error = derivative_of_sigmoid_func(Y_in) * output_error;

            w_hidden_to_output[0] += global_error * A_out;
            w_hidden_to_output[1] += global_error * B_out;
            w_hidden_to_output[2] += global_error * C_out;
            w_hidden_to_output[3] += global_error * D_out;
            w_hidden_to_output[4] += global_error * E_out;
            bias_output += global_error;


            error1 = derivative_of_sigmoid_func(A_in) * global_error * w_hidden_to_output[0];
            error2 = derivative_of_sigmoid_func(B_in) * global_error * w_hidden_to_output[1];
            error3 = derivative_of_sigmoid_func(C_in) * global_error * w_hidden_to_output[2];
            error4 = derivative_of_sigmoid_func(D_in) * global_error * w_hidden_to_output[3];
            error5 = derivative_of_sigmoid_func(E_in) * global_error * w_hidden_to_output[4];

            w_input_to_hidden[0][0] += error1 * input1[k];
            w_input_to_hidden[0][1] += error2 * input1[k];
            w_input_to_hidden[0][2] += error3 * input1[k];
            w_input_to_hidden[0][3] += error4 * input1[k];
            w_input_to_hidden[0][4] += error5 * input1[k];

            w_input_to_hidden[1][0] += error1 * input2[k];
            w_input_to_hidden[1][1] += error2 * input2[k];
            w_input_to_hidden[1][2] += error3 * input2[k];
            w_input_to_hidden[1][3] += error4 * input2[k];
            w_input_to_hidden[1][4] += error5 * input2[k];
            bias1 +=error1;
            bias2 +=error2;
            bias3 +=error3;
            bias4 +=error4;
            bias5 +=error5;
        }
    }

    for(u8 i = 0; i < 2; i++){
        for(u8 j = 0; j < 5; j++){
            qDebug() << "in_to_h_w[i][j] : " << w_input_to_hidden[i][j];
        }
    }
    for(u8 j = 0; j < 5; j++){
        qDebug() << "h_to_o_w[i] : " << w_hidden_to_output[j];
    }
    qDebug() << "bias1" << bias1;
    qDebug() << "bias2" << bias2;
    qDebug() << "bias3" << bias3;
    qDebug() << "bias4" << bias4;
    qDebug() << "bias5" << bias5;
    qDebug() << "bias_output" << bias_output;
    for(u8 k = 0; k < 4; k++){
        qDebug() << "output : " << calculated_output[k];
    }
}
