#include "mainwindow.h"

void ann::_2_5_3_1_ann_train(void){
    double input1[4] = {0,0,1,1};
    double input2[4] = {0,1,0,1};
    double desired_output[4] = {0,1,1,0};

    double calculated_output[4];

    double w_input_to_hidden[2][5];
    double w_hidden_to_hidden[5][3];
    double w_hidden_to_output[3];

    double hidden_neuron_in_1[5];
    double hidden_neuron_out_1[5];
    double hidden_neuron_bias_1[5];
    double hidden_neuron_error_1[5];

    double hidden_neuron_in_2[3];
    double hidden_neuron_out_2[3];
    double hidden_neuron_bias_2[3];
    double hidden_neuron_error_2[3];

    double Y_in,Y_out;
    double bias_output = 0.6;
    double output_error;
    double global_error;

    u32 epoch = 1000000;

    for(u8 i = 0; i < 4; i++){
        qDebug() << " input1 : " << input1[i] << " input2 : " << input2[i] << "output : " << desired_output[i];
    }

    for(u8 i = 0; i < 5; i++){
        hidden_neuron_bias_1[i] = 0.1 + i*0.1;
    }
    for(u8 i = 0; i < 3; i++){
        hidden_neuron_bias_2[i] = 0.4 + i*0.1;
    }

    for(u8 i = 0; i < 2; i++){
        for(u8 j = 0; j < 5; j++){
            w_input_to_hidden[i][j] = 0.1;
        }
    }
    for(u8 i = 0; i < 5; i++){
        for(u8 j = 0; j < 3; j++){
            w_hidden_to_hidden[i][j] = 0.1;
        }
    }
    for(u8 i = 0; i < 3; i++){
        w_hidden_to_output[i] = 0.1;
    }

    for(u32 era = 0; era < epoch; era++){
        for(u8 k = 0; k < 4; k++){

            hidden_neuron_in_1[0] = input1[k]*w_input_to_hidden[0][0] +
                                    input2[k]*w_input_to_hidden[1][0] +
                                    hidden_neuron_bias_1[0];
            hidden_neuron_in_1[1] = input1[k]*w_input_to_hidden[0][1] +
                                    input2[k]*w_input_to_hidden[1][1] +
                                    hidden_neuron_bias_1[1];
            hidden_neuron_in_1[2] = input1[k]*w_input_to_hidden[0][2] +
                                    input2[k]*w_input_to_hidden[1][2] +
                                    hidden_neuron_bias_1[2];
            hidden_neuron_in_1[3] = input1[k]*w_input_to_hidden[0][3] +
                                    input2[k]*w_input_to_hidden[1][3] +
                                    hidden_neuron_bias_1[3];
            hidden_neuron_in_1[4] = input1[k]*w_input_to_hidden[0][4] +
                                    input2[k]*w_input_to_hidden[1][4] +
                                    hidden_neuron_bias_1[4];

            for(u8 i = 0; i < 5; i++){
                hidden_neuron_out_1[i] = sigmoid_func(hidden_neuron_in_1[i]);
            }

            hidden_neuron_in_2[0] = hidden_neuron_out_1[0]*w_hidden_to_hidden[0][0] +
                                    hidden_neuron_out_1[1]*w_hidden_to_hidden[1][0] +
                                    hidden_neuron_out_1[2]*w_hidden_to_hidden[2][0] +
                                    hidden_neuron_out_1[3]*w_hidden_to_hidden[3][0] +
                                    hidden_neuron_out_1[4]*w_hidden_to_hidden[4][0] +
                                    hidden_neuron_bias_2[0];
            hidden_neuron_in_2[1] = hidden_neuron_out_1[0]*w_hidden_to_hidden[0][1] +
                                    hidden_neuron_out_1[1]*w_hidden_to_hidden[1][1] +
                                    hidden_neuron_out_1[2]*w_hidden_to_hidden[2][1] +
                                    hidden_neuron_out_1[3]*w_hidden_to_hidden[3][1] +
                                    hidden_neuron_out_1[4]*w_hidden_to_hidden[4][1] +
                                    hidden_neuron_bias_2[1];
            hidden_neuron_in_2[2] = hidden_neuron_out_1[0]*w_hidden_to_hidden[0][2] +
                                    hidden_neuron_out_1[1]*w_hidden_to_hidden[1][2] +
                                    hidden_neuron_out_1[2]*w_hidden_to_hidden[2][2] +
                                    hidden_neuron_out_1[3]*w_hidden_to_hidden[3][2] +
                                    hidden_neuron_out_1[4]*w_hidden_to_hidden[4][2] +
                                    hidden_neuron_bias_2[2];

            for(u8 i = 0; i < 3; i++){
                hidden_neuron_out_2[i] = sigmoid_func(hidden_neuron_in_2[i]);
            }

            Y_in =  hidden_neuron_out_2[0]*w_hidden_to_output[0] +
                    hidden_neuron_out_2[1]*w_hidden_to_output[1] +
                    hidden_neuron_out_2[2]*w_hidden_to_output[2] +
                    bias_output;
            Y_out = sigmoid_func(Y_in);
            output_error = desired_output[k] - Y_out;
            calculated_output[k] = Y_out;

            global_error = derivative_of_sigmoid_func(Y_in) * output_error;

            for(u8 i = 0; i < 3; i++){
                w_hidden_to_output[i] += global_error * hidden_neuron_out_2[i];
            }

            bias_output += global_error;

            for(u8 i = 0; i < 3; i++){
                hidden_neuron_error_2[i] = derivative_of_sigmoid_func(hidden_neuron_in_2[i]) * global_error * w_hidden_to_output[i];
            }

            w_hidden_to_hidden[0][0] += hidden_neuron_error_2[0] * hidden_neuron_out_1[0];
            w_hidden_to_hidden[1][0] += hidden_neuron_error_2[0] * hidden_neuron_out_1[1];
            w_hidden_to_hidden[2][0] += hidden_neuron_error_2[0] * hidden_neuron_out_1[2];
            w_hidden_to_hidden[3][0] += hidden_neuron_error_2[0] * hidden_neuron_out_1[3];
            w_hidden_to_hidden[4][0] += hidden_neuron_error_2[0] * hidden_neuron_out_1[4];

            w_hidden_to_hidden[0][1] += hidden_neuron_error_2[1] * hidden_neuron_out_1[0];
            w_hidden_to_hidden[1][1] += hidden_neuron_error_2[1] * hidden_neuron_out_1[1];
            w_hidden_to_hidden[2][1] += hidden_neuron_error_2[1] * hidden_neuron_out_1[2];
            w_hidden_to_hidden[3][1] += hidden_neuron_error_2[1] * hidden_neuron_out_1[3];
            w_hidden_to_hidden[4][1] += hidden_neuron_error_2[1] * hidden_neuron_out_1[4];

            w_hidden_to_hidden[0][2] += hidden_neuron_error_2[2] * hidden_neuron_out_1[0];
            w_hidden_to_hidden[1][2] += hidden_neuron_error_2[2] * hidden_neuron_out_1[1];
            w_hidden_to_hidden[2][2] += hidden_neuron_error_2[2] * hidden_neuron_out_1[2];
            w_hidden_to_hidden[3][2] += hidden_neuron_error_2[2] * hidden_neuron_out_1[3];
            w_hidden_to_hidden[4][2] += hidden_neuron_error_2[2] * hidden_neuron_out_1[4];

            for(u8 i = 0; i < 3; i++){
                hidden_neuron_bias_2[i] += hidden_neuron_error_2[i];
            }

            hidden_neuron_error_1[0] =  derivative_of_sigmoid_func(hidden_neuron_in_1[0]) * hidden_neuron_error_2[0] * w_hidden_to_hidden[0][0] +
                                        derivative_of_sigmoid_func(hidden_neuron_in_1[0]) * hidden_neuron_error_2[1] * w_hidden_to_hidden[0][1] +
                                        derivative_of_sigmoid_func(hidden_neuron_in_1[0]) * hidden_neuron_error_2[2] * w_hidden_to_hidden[0][2];

            hidden_neuron_error_1[1] =  derivative_of_sigmoid_func(hidden_neuron_in_1[1]) * hidden_neuron_error_2[0] * w_hidden_to_hidden[1][0] +
                                        derivative_of_sigmoid_func(hidden_neuron_in_1[1]) * hidden_neuron_error_2[1] * w_hidden_to_hidden[1][1] +
                                        derivative_of_sigmoid_func(hidden_neuron_in_1[1]) * hidden_neuron_error_2[2] * w_hidden_to_hidden[1][2];

            hidden_neuron_error_1[2] =  derivative_of_sigmoid_func(hidden_neuron_in_1[2]) * hidden_neuron_error_2[0] * w_hidden_to_hidden[2][0] +
                                        derivative_of_sigmoid_func(hidden_neuron_in_1[2]) * hidden_neuron_error_2[1] * w_hidden_to_hidden[2][1] +
                                        derivative_of_sigmoid_func(hidden_neuron_in_1[2]) * hidden_neuron_error_2[2] * w_hidden_to_hidden[2][2];

            hidden_neuron_error_1[3] =  derivative_of_sigmoid_func(hidden_neuron_in_1[3]) * hidden_neuron_error_2[0] * w_hidden_to_hidden[3][0] +
                                        derivative_of_sigmoid_func(hidden_neuron_in_1[3]) * hidden_neuron_error_2[1] * w_hidden_to_hidden[3][1] +
                                        derivative_of_sigmoid_func(hidden_neuron_in_1[3]) * hidden_neuron_error_2[2] * w_hidden_to_hidden[3][2];

            hidden_neuron_error_1[4] =  derivative_of_sigmoid_func(hidden_neuron_in_1[4]) * hidden_neuron_error_2[0] * w_hidden_to_hidden[4][0] +
                                        derivative_of_sigmoid_func(hidden_neuron_in_1[4]) * hidden_neuron_error_2[1] * w_hidden_to_hidden[4][1] +
                                        derivative_of_sigmoid_func(hidden_neuron_in_1[4]) * hidden_neuron_error_2[2] * w_hidden_to_hidden[4][2];

            w_input_to_hidden[0][0] += hidden_neuron_error_1[0] * input1[k];
            w_input_to_hidden[0][1] += hidden_neuron_error_1[1] * input1[k];
            w_input_to_hidden[0][2] += hidden_neuron_error_1[2] * input1[k];
            w_input_to_hidden[0][3] += hidden_neuron_error_1[3] * input1[k];
            w_input_to_hidden[0][4] += hidden_neuron_error_1[4] * input1[k];

            w_input_to_hidden[1][0] += hidden_neuron_error_1[0] * input2[k];
            w_input_to_hidden[1][1] += hidden_neuron_error_1[1] * input2[k];
            w_input_to_hidden[1][2] += hidden_neuron_error_1[2] * input2[k];
            w_input_to_hidden[1][3] += hidden_neuron_error_1[3] * input2[k];
            w_input_to_hidden[1][4] += hidden_neuron_error_1[4] * input2[k];

            for(u8 i = 0; i < 5; i++){
                hidden_neuron_bias_1[i] +=hidden_neuron_error_1[i];
            }
        }
    }

    for(u8 i = 0; i < 2; i++){
        for(u8 j = 0; j < 5; j++){
            qDebug() << "in_to_h_w[i][j] : " << w_input_to_hidden[i][j];
        }
    }
    for(u8 i = 0; i < 5; i++){
        for(u8 j = 0; j < 3; j++){
            qDebug() << "h_to_h_w[i][j] : " << w_hidden_to_hidden[i][j];
        }
    }
    for(u8 j = 0; j < 3; j++){
        qDebug() << "h_to_o_w[i] : " << w_hidden_to_output[j];
    }
    qDebug() << "hidden_neuron_bias_1[0]" << hidden_neuron_bias_1[0];
    qDebug() << "hidden_neuron_bias_1[1]" << hidden_neuron_bias_1[1];
    qDebug() << "hidden_neuron_bias_1[2]" << hidden_neuron_bias_1[2];
    qDebug() << "hidden_neuron_bias_1[3]" << hidden_neuron_bias_1[3];
    qDebug() << "hidden_neuron_bias_1[4]" << hidden_neuron_bias_1[4];

    qDebug() << "hidden_neuron_bias_2[0]" << hidden_neuron_bias_2[0];
    qDebug() << "hidden_neuron_bias_2[1]" << hidden_neuron_bias_2[1];
    qDebug() << "hidden_neuron_bias_2[2]" << hidden_neuron_bias_2[2];

    qDebug() << "bias_output" << bias_output;

    for(u8 k = 0; k < 4; k++){
        qDebug() << "output : " << calculated_output[k];
    }
}
void ann::_2_5_2_1_ann_train(void){
    double input1[4] = {0,0,1,1};
    double input2[4] = {0,1,0,1};
    double desired_output[4] = {0,1,1,0};

    double calculated_output[4];
    double Y_in,Y_out;

    double w_input_to_hidden[2][5];
    double w_hidden_to_hidden[5][2];
    double w_hidden_to_output[2];

    double hidden_neuron_in_1[5];
    double hidden_neuron_out_1[5];
    double hidden_neuron_bias_1[5];
    double hidden_neuron_error_1[5];

    double hidden_neuron_in_2[2];
    double hidden_neuron_out_2[2];
    double hidden_neuron_bias_2[2];
    double hidden_neuron_error_2[2];

    double bias_output = 0.6;
    double output_error;
    double global_error;

    u32 epoch = 1000000;

    for(u8 i = 0; i < 4; i++){
        qDebug() << " input1 : " << input1[i] << " input2 : " << input2[i] << "output : " << desired_output[i];
    }

    for(u8 i = 0; i < 5; i++){
        hidden_neuron_bias_1[i] = 0.1 + i*0.1;
    }
    for(u8 i = 0; i < 2; i++){
        hidden_neuron_bias_2[i] = 0.4 + i*0.1;
    }

    for(u8 i = 0; i < 2; i++){
        for(u8 j = 0; j < 5; j++){
            w_input_to_hidden[i][j] = 0.1;
        }
    }
    for(u8 i = 0; i < 5; i++){
        for(u8 j = 0; j < 2; j++){
            w_hidden_to_hidden[i][j] = 0.1;
        }
    }
    for(u8 i = 0; i < 2; i++){
        w_hidden_to_output[i] = 0.1;
    }

    for(u32 era = 0; era < epoch; era++){
        for(u8 k = 0; k < 4; k++){

            hidden_neuron_in_1[0] = input1[k]*w_input_to_hidden[0][0] +
                                    input2[k]*w_input_to_hidden[1][0] +
                                    hidden_neuron_bias_1[0];
            hidden_neuron_in_1[1] = input1[k]*w_input_to_hidden[0][1] +
                                    input2[k]*w_input_to_hidden[1][1] +
                                    hidden_neuron_bias_1[1];
            hidden_neuron_in_1[2] = input1[k]*w_input_to_hidden[0][2] +
                                    input2[k]*w_input_to_hidden[1][2] +
                                    hidden_neuron_bias_1[2];
            hidden_neuron_in_1[3] = input1[k]*w_input_to_hidden[0][3] +
                                    input2[k]*w_input_to_hidden[1][3] +
                                    hidden_neuron_bias_1[3];
            hidden_neuron_in_1[4] = input1[k]*w_input_to_hidden[0][4] +
                                    input2[k]*w_input_to_hidden[1][4] +
                                    hidden_neuron_bias_1[4];

            for(u8 i = 0; i < 5; i++){
                hidden_neuron_out_1[i] = sigmoid_func(hidden_neuron_in_1[i]);
            }

            hidden_neuron_in_2[0] = hidden_neuron_out_1[0]*w_hidden_to_hidden[0][0] +
                                    hidden_neuron_out_1[1]*w_hidden_to_hidden[1][0] +
                                    hidden_neuron_out_1[2]*w_hidden_to_hidden[2][0] +
                                    hidden_neuron_out_1[3]*w_hidden_to_hidden[3][0] +
                                    hidden_neuron_out_1[4]*w_hidden_to_hidden[4][0] +
                                    hidden_neuron_bias_2[0];
            hidden_neuron_in_2[1] = hidden_neuron_out_1[0]*w_hidden_to_hidden[0][1] +
                                    hidden_neuron_out_1[1]*w_hidden_to_hidden[1][1] +
                                    hidden_neuron_out_1[2]*w_hidden_to_hidden[2][1] +
                                    hidden_neuron_out_1[3]*w_hidden_to_hidden[3][1] +
                                    hidden_neuron_out_1[4]*w_hidden_to_hidden[4][1] +
                                    hidden_neuron_bias_2[1];

            for(u8 i = 0; i < 2; i++){
                hidden_neuron_out_2[i] = sigmoid_func(hidden_neuron_in_2[i]);
            }

            Y_in =  hidden_neuron_out_2[0]*w_hidden_to_output[0] +
                    hidden_neuron_out_2[1]*w_hidden_to_output[1] +
                    bias_output;
            Y_out = sigmoid_func(Y_in);
            output_error = desired_output[k] - Y_out;
            calculated_output[k] = Y_out;

            global_error = derivative_of_sigmoid_func(Y_in) * output_error;

            for(u8 i = 0; i < 2; i++){
                w_hidden_to_output[i] += global_error * hidden_neuron_out_2[i];
            }

            bias_output += global_error;

            for(u8 i = 0; i < 2; i++){
                hidden_neuron_error_2[i] = derivative_of_sigmoid_func(hidden_neuron_in_2[i]) * global_error * w_hidden_to_output[i];
            }

            w_hidden_to_hidden[0][0] += hidden_neuron_error_2[0] * hidden_neuron_out_1[0];
            w_hidden_to_hidden[1][0] += hidden_neuron_error_2[0] * hidden_neuron_out_1[1];
            w_hidden_to_hidden[2][0] += hidden_neuron_error_2[0] * hidden_neuron_out_1[2];
            w_hidden_to_hidden[3][0] += hidden_neuron_error_2[0] * hidden_neuron_out_1[3];
            w_hidden_to_hidden[4][0] += hidden_neuron_error_2[0] * hidden_neuron_out_1[4];

            w_hidden_to_hidden[0][1] += hidden_neuron_error_2[1] * hidden_neuron_out_1[0];
            w_hidden_to_hidden[1][1] += hidden_neuron_error_2[1] * hidden_neuron_out_1[1];
            w_hidden_to_hidden[2][1] += hidden_neuron_error_2[1] * hidden_neuron_out_1[2];
            w_hidden_to_hidden[3][1] += hidden_neuron_error_2[1] * hidden_neuron_out_1[3];
            w_hidden_to_hidden[4][1] += hidden_neuron_error_2[1] * hidden_neuron_out_1[4];

            for(u8 i = 0; i < 2; i++){
                hidden_neuron_bias_2[i] += hidden_neuron_error_2[i];
            }

            hidden_neuron_error_1[0] =  derivative_of_sigmoid_func(hidden_neuron_in_1[0]) * hidden_neuron_error_2[0] * w_hidden_to_hidden[0][0] +
                                        derivative_of_sigmoid_func(hidden_neuron_in_1[0]) * hidden_neuron_error_2[1] * w_hidden_to_hidden[0][1];
            hidden_neuron_error_1[1] =  derivative_of_sigmoid_func(hidden_neuron_in_1[1]) * hidden_neuron_error_2[0] * w_hidden_to_hidden[1][0] +
                                        derivative_of_sigmoid_func(hidden_neuron_in_1[1]) * hidden_neuron_error_2[1] * w_hidden_to_hidden[1][1];
            hidden_neuron_error_1[2] =  derivative_of_sigmoid_func(hidden_neuron_in_1[2]) * hidden_neuron_error_2[0] * w_hidden_to_hidden[2][0] +
                                        derivative_of_sigmoid_func(hidden_neuron_in_1[2]) * hidden_neuron_error_2[1] * w_hidden_to_hidden[2][1];
            hidden_neuron_error_1[3] =  derivative_of_sigmoid_func(hidden_neuron_in_1[3]) * hidden_neuron_error_2[0] * w_hidden_to_hidden[3][0] +
                                        derivative_of_sigmoid_func(hidden_neuron_in_1[3]) * hidden_neuron_error_2[1] * w_hidden_to_hidden[3][1];
            hidden_neuron_error_1[4] =  derivative_of_sigmoid_func(hidden_neuron_in_1[4]) * hidden_neuron_error_2[0] * w_hidden_to_hidden[4][0] +
                                        derivative_of_sigmoid_func(hidden_neuron_in_1[4]) * hidden_neuron_error_2[1] * w_hidden_to_hidden[4][1];

            w_input_to_hidden[0][0] += hidden_neuron_error_1[0] * input1[k];
            w_input_to_hidden[0][1] += hidden_neuron_error_1[1] * input1[k];
            w_input_to_hidden[0][2] += hidden_neuron_error_1[2] * input1[k];
            w_input_to_hidden[0][3] += hidden_neuron_error_1[3] * input1[k];
            w_input_to_hidden[0][4] += hidden_neuron_error_1[4] * input1[k];

            w_input_to_hidden[1][0] += hidden_neuron_error_1[0] * input2[k];
            w_input_to_hidden[1][1] += hidden_neuron_error_1[1] * input2[k];
            w_input_to_hidden[1][2] += hidden_neuron_error_1[2] * input2[k];
            w_input_to_hidden[1][3] += hidden_neuron_error_1[3] * input2[k];
            w_input_to_hidden[1][4] += hidden_neuron_error_1[4] * input2[k];

            for(u8 i = 0; i < 5; i++){
                hidden_neuron_bias_1[i] +=hidden_neuron_error_1[i];
            }
        }
    }

    for(u8 i = 0; i < 2; i++){
        for(u8 j = 0; j < 5; j++){
            qDebug() << "in_to_h_w[i][j] : " << w_input_to_hidden[i][j];
        }
    }
    for(u8 i = 0; i < 5; i++){
        for(u8 j = 0; j < 2; j++){
            qDebug() << "h_to_h_w[i][j] : " << w_hidden_to_hidden[i][j];
        }
    }
    for(u8 j = 0; j < 2; j++){
        qDebug() << "h_to_o_w[i] : " << w_hidden_to_output[j];
    }
    qDebug() << "hidden_neuron_bias_1[0]" << hidden_neuron_bias_1[0];
    qDebug() << "hidden_neuron_bias_1[1]" << hidden_neuron_bias_1[1];
    qDebug() << "hidden_neuron_bias_1[2]" << hidden_neuron_bias_1[2];
    qDebug() << "hidden_neuron_bias_1[3]" << hidden_neuron_bias_1[3];
    qDebug() << "hidden_neuron_bias_1[4]" << hidden_neuron_bias_1[4];

    qDebug() << "hidden_neuron_bias_2[0]" << hidden_neuron_bias_2[0];
    qDebug() << "hidden_neuron_bias_2[1]" << hidden_neuron_bias_2[1];
    qDebug() << "bias_output" << bias_output;

    for(u8 k = 0; k < 4; k++){
        qDebug() << "output : " << calculated_output[k];
    }
}
void ann::_2_3_2_1_ann_train(void){
    double input1[4] = {0,0,1,1};
    double input2[4] = {0,1,0,1};
    double desired_output[4] = {0,1,1,0};
    double calculated_output[4] = {0,0,0,0};
    double Y_in,Y_out;
    double w_input_to_hidden[2][3];
    double w_hidden_to_hidden[3][2];
    double w_hidden_to_output[2];

    double A_in,B_in,C_in;
    double A_out,B_out,C_out;
    double output_error;

    double K_in, L_in;
    double K_out, L_out;

    double biasA = 0.1;
    double biasB = 0.2;
    double biasC = 0.3;
    double bias_output = 0.6;

    double biasK = 0.4;
    double biasL = 0.5;

    double errorA,errorB,errorC;
    double errorK,errorL;

    u32 epoch = 200000;

    double global_error;

    for(u8 i = 0; i < 4; i++){
        qDebug() << " input1 : " << input1[i] << " input2 : " << input2[i] << "output : " << desired_output[i];
    }

    w_input_to_hidden[0][0] = 0.1;
    w_input_to_hidden[0][1] = 0.1;
    w_input_to_hidden[0][2] = 0.1;

    w_input_to_hidden[1][0] = 0.1;
    w_input_to_hidden[1][1] = 0.1;
    w_input_to_hidden[1][2] = 0.1;

    w_hidden_to_hidden[0][0] = 0.1;
    w_hidden_to_hidden[1][0] = 0.1;
    w_hidden_to_hidden[2][0] = 0.1;
    w_hidden_to_hidden[0][1] = 0.1;
    w_hidden_to_hidden[1][1] = 0.1;
    w_hidden_to_hidden[2][1] = 0.1;

    w_hidden_to_output[0] = 0.1;
    w_hidden_to_output[1] = 0.1;

    for(u32 era = 0; era < epoch; era++){
        for(u8 k = 0; k < 4; k++){

            A_in = input1[k]*w_input_to_hidden[0][0] + input2[k]*w_input_to_hidden[1][0] + biasA;
            B_in = input1[k]*w_input_to_hidden[0][1] + input2[k]*w_input_to_hidden[1][1] + biasB;
            C_in = input1[k]*w_input_to_hidden[0][2] + input2[k]*w_input_to_hidden[1][2] + biasC;

            A_out = sigmoid_func(A_in);
            B_out = sigmoid_func(B_in);
            C_out = sigmoid_func(C_in);

            K_in = A_out*w_hidden_to_hidden[0][0] + B_out*w_hidden_to_hidden[1][0] + C_out*w_hidden_to_hidden[2][0] + biasK;
            L_in = A_out*w_hidden_to_hidden[0][1] + B_out*w_hidden_to_hidden[1][1] + C_out*w_hidden_to_hidden[2][1] + biasL;

            K_out = sigmoid_func(K_in);
            L_out = sigmoid_func(L_in);

            Y_in = K_out*w_hidden_to_output[0] + L_out*w_hidden_to_output[1] + bias_output;
            Y_out = sigmoid_func(Y_in);
            output_error = desired_output[k] - Y_out;
            calculated_output[k] = Y_out;

            global_error = derivative_of_sigmoid_func(Y_in) * output_error;

            w_hidden_to_output[0] += global_error * K_out;
            w_hidden_to_output[1] += global_error * L_out;
            bias_output += global_error;

            errorK = derivative_of_sigmoid_func(K_in) * global_error * w_hidden_to_output[0];
            errorL = derivative_of_sigmoid_func(L_in) * global_error * w_hidden_to_output[1];

            w_hidden_to_hidden[0][0] += errorK * A_out;
            w_hidden_to_hidden[1][0] += errorK * B_out;
            w_hidden_to_hidden[2][0] += errorK * C_out;
            w_hidden_to_hidden[0][1] += errorL * A_out;
            w_hidden_to_hidden[1][1] += errorL * B_out;
            w_hidden_to_hidden[2][1] += errorL * C_out;
            biasK += errorK;
            biasL += errorL;

            errorA = derivative_of_sigmoid_func(A_in) * errorK * w_hidden_to_hidden[0][0] + derivative_of_sigmoid_func(A_in) * errorL * w_hidden_to_hidden[0][1];
            errorB = derivative_of_sigmoid_func(B_in) * errorK * w_hidden_to_hidden[1][0] + derivative_of_sigmoid_func(B_in) * errorL * w_hidden_to_hidden[1][1];
            errorC = derivative_of_sigmoid_func(C_in) * errorK * w_hidden_to_hidden[2][0] + derivative_of_sigmoid_func(C_in) * errorL * w_hidden_to_hidden[2][1];

            w_input_to_hidden[0][0] += errorA * input1[k];
            w_input_to_hidden[0][1] += errorB * input1[k];
            w_input_to_hidden[0][2] += errorC * input1[k];

            w_input_to_hidden[1][0] += errorA * input2[k];
            w_input_to_hidden[1][1] += errorB * input2[k];
            w_input_to_hidden[1][2] += errorC * input2[k];
            biasA +=errorA;
            biasB +=errorB;
            biasC +=errorC;
        }
    }

    for(u8 i = 0; i < 2; i++){
        for(u8 j = 0; j < 3; j++){
            qDebug() << "in_to_h_w[i][j] : " << w_input_to_hidden[i][j];
        }
    }
    for(u8 i = 0; i < 3; i++){
        for(u8 j = 0; j < 2; j++){
            qDebug() << "h_to_h_w[i][j] : " << w_hidden_to_hidden[i][j];
        }
    }
    for(u8 j = 0; j < 2; j++){
        qDebug() << "h_to_o_w[i] : " << w_hidden_to_output[j];
    }
    qDebug() << "biasA" << biasA;
    qDebug() << "biasB" << biasB;
    qDebug() << "biasC" << biasC;
    qDebug() << "biasK" << biasK;
    qDebug() << "biasL" << biasL;
    qDebug() << "bias_output" << bias_output;
    for(u8 k = 0; k < 4; k++){
        qDebug() << "output : " << calculated_output[k];
    }
}

void ann::_2_2_1_ann_genetic(void){
    double input1 = 1;
    double input2 = 1;
    double desired_output = 1;
    double calculated_output = 0;
    double Y_in,Y_out;
    double w_input_to_hidden[2][2];
    double w_hidden_to_output[2];
    double biasA;
    double biasB;
    double bias_output;

    double A_in,B_in;
    double A_out,B_out;
    double output_error;

    double population_1[9] = {0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19};
    double population_2[9] = {0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29};
    double population_3[9] = {0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39};
    double population[3][9];

    memcpy(&population[0][0],&population_1[0],9);
    memcpy(&population[1][0],&population_2[0],9);
    memcpy(&population[2][0],&population_3[0],9);

    qDebug() << " input1 : " << input1 << " input2 : " << input2 << "output : " << desired_output;

    w_input_to_hidden[0][0] = population[][0];
    w_input_to_hidden[0][1] = population_1[1];
    w_input_to_hidden[1][0] = population_1[2];
    w_input_to_hidden[1][1] = population_1[3];
    w_hidden_to_output[0] = population_1[4];
    w_hidden_to_output[1] = population_1[5];
    biasA = population_1[6];
    biasB = population_1[7];
    bias_output = population_1[8];

    A_in = input1*w_input_to_hidden[0][0] + input2*w_input_to_hidden[1][0] + biasA;
    B_in = input1*w_input_to_hidden[0][1] + input2*w_input_to_hidden[1][1] + biasB;

    A_out = sigmoid_func(A_in);
    B_out = sigmoid_func(B_in);

    Y_in = A_out*w_hidden_to_output[0] + B_out*w_hidden_to_output[1] + bias_output;
    Y_out = sigmoid_func(Y_in);
    output_error = desired_output - Y_out;
    calculated_output = Y_out;

    qDebug() << "output : " << calculated_output << "output err :" << output_error;

    for(u8 i = 0; i < 2; i++){
        for(u8 j = 0; j < 2; j++){
            qDebug() << QString("in_to_h_w[%1][%2] :").arg(i).arg(j) << w_input_to_hidden[i][j];
        }
    }
    for(u8 j = 0; j < 2; j++){
        qDebug() << QString("h_to_o_w[%1] :").arg(j) << w_hidden_to_output[j];
    }
    qDebug() << "biasA" << biasA;
    qDebug() << "biasB" << biasB;
    qDebug() << "bias_output" << bias_output;
}
void ann::_2_2_1_ann_train(void){
    double input1 = 1;
    double input2 = 1;
    double desired_output = 1;
    double calculated_output = 0;
    double Y_in,Y_out;
    double w_input_to_hidden[2][2];
    double w_hidden_to_output[2];

    double A_in,B_in;
    double A_out,B_out;
    double output_error;

    double biasA = 0.1;
    double biasB = 0.2;
    double bias_output = 0.6;

    double errorA,errorB;

    u32 epoch = 50;
    double learning_rate = 1;

    double global_error;
    double err_sum;

    qDebug() << " input1 : " << input1 << " input2 : " << input2 << "output : " << desired_output;

    w_input_to_hidden[0][0] = 0.1;
    w_input_to_hidden[0][1] = 0.1;

    w_input_to_hidden[1][0] = 0.1;
    w_input_to_hidden[1][1] = 0.1;

    w_hidden_to_output[0] = 0.1;
    w_hidden_to_output[1] = 0.1;

    for(u32 era = 0; era < epoch; era++){
        A_in = input1*w_input_to_hidden[0][0] + input2*w_input_to_hidden[1][0] + biasA;
        B_in = input1*w_input_to_hidden[0][1] + input2*w_input_to_hidden[1][1] + biasB;

        A_out = sigmoid_func(A_in);
        B_out = sigmoid_func(B_in);

        Y_in = A_out*w_hidden_to_output[0] + B_out*w_hidden_to_output[1] + bias_output;
        Y_out = sigmoid_func(Y_in);
        output_error = desired_output - Y_out;
        calculated_output = Y_out;

        err_sum = output_error;

        global_error = derivative_of_sigmoid_func(Y_in) * output_error;

        w_hidden_to_output[0] += global_error * A_out * learning_rate;
        w_hidden_to_output[1] += global_error * B_out * learning_rate;

        bias_output += global_error * learning_rate;

        errorA = derivative_of_sigmoid_func(A_in) * global_error * w_hidden_to_output[0];
        errorB = derivative_of_sigmoid_func(B_in) * global_error * w_hidden_to_output[1];

        w_input_to_hidden[0][0] += errorA * input1 * learning_rate;
        w_input_to_hidden[0][1] += errorB * input1 * learning_rate;

        w_input_to_hidden[1][0] += errorA * input2 * learning_rate;
        w_input_to_hidden[1][1] += errorB * input2 * learning_rate;

        biasA += errorA * learning_rate;
        biasB += errorB * learning_rate;

        qDebug() << QString("era-%1").arg(era) << "error :" << err_sum;
    }

    for(u8 i = 0; i < 2; i++){
        for(u8 j = 0; j < 2; j++){
            qDebug() << QString("in_to_h_w[%1][%2] :").arg(i).arg(j) << w_input_to_hidden[i][j];
        }
    }
    for(u8 j = 0; j < 2; j++){
        qDebug() << QString("h_to_o_w[%1] :").arg(j) << w_hidden_to_output[j];
    }
    qDebug() << "biasA" << biasA;
    qDebug() << "biasB" << biasB;
    qDebug() << "bias_output" << bias_output;
    qDebug() << "output : " << calculated_output;
}
void ann::_2_3_1_ann_train(void){
    double input1[4] = {0,0,1,1};
    double input2[4] = {0,1,0,1};
    double desired_output[4] = {0,1,1,0};
    double calculated_output[4] = {0,0,0,0};
    double Y_in,Y_out;
    double w_input_to_hidden[2][3];
    double w_hidden_to_output[3];

    double A_in,B_in,C_in;
    double A_out,B_out,C_out;
    double output_error;

    double biasA = 0.1;
    double biasB = 0.2;
    double biasC = 0.3;
    double bias_output = 0.6;

    double errorA,errorB,errorC;

    u32 epoch = 1000;
    double learning_rate = 1;

    double global_error;
    double err_sum[4];
    double current_err = 0;
    static double last_err = 0;

    for(u8 i = 0; i < 4; i++){
        qDebug() << " input1 : " << input1[i] << " input2 : " << input2[i] << "output : " << desired_output[i];
    }

    w_input_to_hidden[0][0] = 0.1;
    w_input_to_hidden[0][1] = 0.1;
    w_input_to_hidden[0][2] = 0.1;

    w_input_to_hidden[1][0] = 0.1;
    w_input_to_hidden[1][1] = 0.1;
    w_input_to_hidden[1][2] = 0.1;

    w_hidden_to_output[0] = 0.1;
    w_hidden_to_output[1] = 0.1;
    w_hidden_to_output[2] = 0.1;

    for(u32 era = 0; era < epoch; era++){
        for(u8 k = 0; k < 4; k++){

            A_in = input1[k]*w_input_to_hidden[0][0] + input2[k]*w_input_to_hidden[1][0] + biasA;
            B_in = input1[k]*w_input_to_hidden[0][1] + input2[k]*w_input_to_hidden[1][1] + biasB;
            C_in = input1[k]*w_input_to_hidden[0][2] + input2[k]*w_input_to_hidden[1][2] + biasC;

            A_out = sigmoid_func(A_in);
            B_out = sigmoid_func(B_in);
            C_out = sigmoid_func(C_in);

            Y_in = A_out*w_hidden_to_output[0] + B_out*w_hidden_to_output[1] + C_out*w_hidden_to_output[2] + bias_output;
            Y_out = sigmoid_func(Y_in);
            output_error = desired_output[k] - Y_out;
            calculated_output[k] = Y_out;

            err_sum[k] = output_error;

            global_error = derivative_of_sigmoid_func(Y_in) * output_error;

            w_hidden_to_output[0] += global_error * A_out * learning_rate;
            w_hidden_to_output[1] += global_error * B_out * learning_rate;
            w_hidden_to_output[2] += global_error * C_out * learning_rate;

            bias_output += global_error * learning_rate;

            errorA = derivative_of_sigmoid_func(A_in) * global_error * w_hidden_to_output[0];
            errorB = derivative_of_sigmoid_func(B_in) * global_error * w_hidden_to_output[1];
            errorC = derivative_of_sigmoid_func(C_in) * global_error * w_hidden_to_output[2];

            w_input_to_hidden[0][0] += errorA * input1[k] * learning_rate;
            w_input_to_hidden[0][1] += errorB * input1[k] * learning_rate;
            w_input_to_hidden[0][2] += errorC * input1[k] * learning_rate;

            w_input_to_hidden[1][0] += errorA * input2[k] * learning_rate;
            w_input_to_hidden[1][1] += errorB * input2[k] * learning_rate;
            w_input_to_hidden[1][2] += errorC * input2[k] * learning_rate;

            biasA += errorA * learning_rate;
            biasB += errorB * learning_rate;
            biasC += errorC * learning_rate;
        }
        current_err = (err_sum[0]*err_sum[0] + err_sum[1]*err_sum[1] + err_sum[2]*err_sum[2] + err_sum[3]*err_sum[3])/4;
        qDebug() << "err :" << current_err << "\t err_dif" << (current_err - last_err);
        last_err = current_err;
    }

    for(u8 i = 0; i < 2; i++){
        for(u8 j = 0; j < 3; j++){
            qDebug() << QString("in_to_h_w[%1][%2] :").arg(i).arg(j) << w_input_to_hidden[i][j];
        }
    }
    for(u8 j = 0; j < 3; j++){
        qDebug() << QString("h_to_o_w[%1] :").arg(j) << w_hidden_to_output[j];
    }
    qDebug() << "biasA" << biasA;
    qDebug() << "biasB" << biasB;
    qDebug() << "biasC" << biasC;
    qDebug() << "bias_output" << bias_output;
    for(u8 k = 0; k < 4; k++){
        qDebug() << "output : " << calculated_output[k];
    }
}
void ann::_2_5_2_ann_train(void){
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
void ann::_2_5_1_ann_train(void){
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
