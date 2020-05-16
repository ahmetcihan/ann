#include "mainwindow.h"

void MainWindow::_2_5_3_2_ann_tryout(void){
    net_2_5_3_2.input[0][0] = 0;            net_2_5_3_2.input[1][0] = 0;
    net_2_5_3_2.input[0][1] = 0;            net_2_5_3_2.input[1][1] = 1;
    net_2_5_3_2.input[0][2] = 1;            net_2_5_3_2.input[1][2] = 0;
    net_2_5_3_2.input[0][3] = 1;            net_2_5_3_2.input[1][3] = 1;
    net_2_5_3_2.desired_output[0][0] = 0;   net_2_5_3_2.desired_output[1][0] = 1;
    net_2_5_3_2.desired_output[0][1] = 1;   net_2_5_3_2.desired_output[1][1] = 0;
    net_2_5_3_2.desired_output[0][2] = 1;   net_2_5_3_2.desired_output[1][2] = 0;
    net_2_5_3_2.desired_output[0][3] = 0;   net_2_5_3_2.desired_output[1][3] = 1;

    net_2_5_3_2.output_bias[0] = 0.6;
    net_2_5_3_2.output_bias[1] = 0.6;

    for(u8 i = 0; i < 2; i++){
        for(u8 j = 0; j < 4; j++){
            qDebug() << QString(" input[%1][%2] : ").arg(i).arg(j) << net_2_5_3_2.input[i][j];
        }
    }
    for(u8 i = 0; i < 2; i++){
        for(u8 j = 0; j < 4; j++){
            qDebug() << QString(" output[%1][%2] : ").arg(i).arg(j) << net_2_5_3_2.desired_output[i][j];
        }
    }

    for(u8 i = 0; i < 5; i++){
        net_2_5_3_2.hidden_neuron_bias_1[i] = 0.1 + i*0.1;
    }
    for(u8 i = 0; i < 3; i++){
        net_2_5_3_2.hidden_neuron_bias_2[i] = 0.4 + i*0.1;
    }

    for(u8 i = 0; i < 2; i++){
        for(u8 j = 0; j < 5; j++){
            net_2_5_3_2.w_input_to_hidden[i][j] = 0.1;
        }
    }
    for(u8 i = 0; i < 5; i++){
        for(u8 j = 0; j < 3; j++){
            net_2_5_3_2.w_hidden_to_hidden[i][j] = 0.1;
        }
    }
    for(u8 i = 0; i < 3; i++){
        for(u8 j = 0; j < 2; j++){
            net_2_5_3_2.w_hidden_to_output[i][j] = 0.1;
        }
    }
    _2_5_3_2_ann_train( net_2_5_3_2.input, net_2_5_3_2.desired_output,
                        net_2_5_3_2.hidden_neuron_bias_1,net_2_5_3_2.hidden_neuron_bias_2,net_2_5_3_2.output_bias,
                        net_2_5_3_2.w_input_to_hidden,net_2_5_3_2.w_hidden_to_hidden,net_2_5_3_2.w_hidden_to_output,
                        200000, 0.1);

}

void MainWindow::_2_5_3_2_ann_train(    double input[2][4], double desired_output[2][4],
                                        double hidden_neuron_bias_1[5], double hidden_neuron_bias_2[3], double output_bias[2],
                                        double w_input_to_hidden[2][5], double w_hidden_to_hidden[5][3], double w_hidden_to_output[3][2],
                                        u32 epoch, double learning_rate){

    double hidden_neuron_in_1[5];
    double hidden_neuron_out_1[5];
    double hidden_neuron_error_1[5];

    double hidden_neuron_in_2[3];
    double hidden_neuron_out_2[3];
    double hidden_neuron_error_2[3];

    double output_error[2];
    double global_error[2];
    double output_in[2];
    double output_out[2];

    double calculated_output[2][4];


    for(u32 era = 0; era < epoch; era++){
        for(u8 k = 0; k < 4; k++){

            for(u8 i = 0; i < 5; i++){
                hidden_neuron_in_1[i] = hidden_neuron_bias_1[i];
            }
            for(u8 i = 0; i < 5; i++){
                for(u8 j = 0; j < 2; j++){
                    hidden_neuron_in_1[i] += input[j][k]*w_input_to_hidden[j][i];
                }
            }
            for(u8 i = 0; i < 5; i++){
                hidden_neuron_out_1[i] = sigmoid_func(hidden_neuron_in_1[i]);
            }
            for(u8 i = 0; i < 3; i++){
                hidden_neuron_in_2[i] = hidden_neuron_bias_2[i];
            }
            for(u8 i = 0; i < 3; i++){
                for(u8 j = 0; j < 5; j++){
                    hidden_neuron_in_2[i] += hidden_neuron_out_1[j]*w_hidden_to_hidden[j][i];
                }
            }
            for(u8 i = 0; i < 3; i++){
                hidden_neuron_out_2[i] = sigmoid_func(hidden_neuron_in_2[i]);
            }

            for(u8 j = 0; j < 2; j++){
                output_in[j] =  output_bias[j];
                for(u8 i = 0; i < 3; i++){
                    output_in[j] += hidden_neuron_out_2[i]*w_hidden_to_output[i][j];
                }
                output_out[j]   = sigmoid_func(output_in[j]);
                output_error[j] = desired_output[j][k] - output_out[j];
                calculated_output[j][k] = output_out[j];
                global_error[j] = derivative_of_sigmoid_func(output_in[j]) * output_error[j];
            }

            for(u8 j = 0; j < 2; j++){
                for(u8 i = 0; i < 3; i++){
                    w_hidden_to_output[i][j] += global_error[j] * hidden_neuron_out_2[i] * learning_rate;
                }
            }

            for(u8 i = 0; i < 2; i++){
                output_bias[i] += global_error[i] * learning_rate;
            }

            for(u8 i = 0; i < 3; i++){
                hidden_neuron_error_2[i] =  derivative_of_sigmoid_func(hidden_neuron_in_2[i]) * global_error[0] * w_hidden_to_output[i][0] +
                                            derivative_of_sigmoid_func(hidden_neuron_in_2[i]) * global_error[1] * w_hidden_to_output[i][1];
            }

            w_hidden_to_hidden[0][0] += hidden_neuron_error_2[0] * hidden_neuron_out_1[0] * learning_rate;
            w_hidden_to_hidden[1][0] += hidden_neuron_error_2[0] * hidden_neuron_out_1[1] * learning_rate;
            w_hidden_to_hidden[2][0] += hidden_neuron_error_2[0] * hidden_neuron_out_1[2] * learning_rate;
            w_hidden_to_hidden[3][0] += hidden_neuron_error_2[0] * hidden_neuron_out_1[3] * learning_rate;
            w_hidden_to_hidden[4][0] += hidden_neuron_error_2[0] * hidden_neuron_out_1[4] * learning_rate;

            w_hidden_to_hidden[0][1] += hidden_neuron_error_2[1] * hidden_neuron_out_1[0] * learning_rate;
            w_hidden_to_hidden[1][1] += hidden_neuron_error_2[1] * hidden_neuron_out_1[1] * learning_rate;
            w_hidden_to_hidden[2][1] += hidden_neuron_error_2[1] * hidden_neuron_out_1[2] * learning_rate;
            w_hidden_to_hidden[3][1] += hidden_neuron_error_2[1] * hidden_neuron_out_1[3] * learning_rate;
            w_hidden_to_hidden[4][1] += hidden_neuron_error_2[1] * hidden_neuron_out_1[4] * learning_rate;

            w_hidden_to_hidden[0][2] += hidden_neuron_error_2[2] * hidden_neuron_out_1[0] * learning_rate;
            w_hidden_to_hidden[1][2] += hidden_neuron_error_2[2] * hidden_neuron_out_1[1] * learning_rate;
            w_hidden_to_hidden[2][2] += hidden_neuron_error_2[2] * hidden_neuron_out_1[2] * learning_rate;
            w_hidden_to_hidden[3][2] += hidden_neuron_error_2[2] * hidden_neuron_out_1[3] * learning_rate;
            w_hidden_to_hidden[4][2] += hidden_neuron_error_2[2] * hidden_neuron_out_1[4] * learning_rate;

            for(u8 i = 0; i < 3; i++){
                hidden_neuron_bias_2[i] += hidden_neuron_error_2[i] * learning_rate;
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

            w_input_to_hidden[0][0] += hidden_neuron_error_1[0] * input[0][k] * learning_rate;
            w_input_to_hidden[0][1] += hidden_neuron_error_1[1] * input[0][k] * learning_rate;
            w_input_to_hidden[0][2] += hidden_neuron_error_1[2] * input[0][k] * learning_rate;
            w_input_to_hidden[0][3] += hidden_neuron_error_1[3] * input[0][k] * learning_rate;
            w_input_to_hidden[0][4] += hidden_neuron_error_1[4] * input[0][k] * learning_rate;

            w_input_to_hidden[1][0] += hidden_neuron_error_1[0] * input[1][k] * learning_rate;
            w_input_to_hidden[1][1] += hidden_neuron_error_1[1] * input[1][k] * learning_rate;
            w_input_to_hidden[1][2] += hidden_neuron_error_1[2] * input[1][k] * learning_rate;
            w_input_to_hidden[1][3] += hidden_neuron_error_1[3] * input[1][k] * learning_rate;
            w_input_to_hidden[1][4] += hidden_neuron_error_1[4] * input[1][k] * learning_rate;

            for(u8 i = 0; i < 5; i++){
                hidden_neuron_bias_1[i] +=hidden_neuron_error_1[i] * learning_rate;
            }
        }
        qDebug() << "training status % " << (era*100)/epoch;
    }

    for(u8 i = 0; i < 2; i++){
        for(u8 j = 0; j < 5; j++){
            qDebug() << QString("in_to_h_w[%1][%2] : ").arg(i).arg(j) << w_input_to_hidden[i][j];
        }
    }
    for(u8 i = 0; i < 5; i++){
        for(u8 j = 0; j < 3; j++){
            qDebug() << QString("h_to_h_w[%1][%2] : ").arg(i).arg(j) << w_hidden_to_hidden[i][j];
        }
    }
    for(u8 i = 0; i < 3; i++){
        for(u8 j = 0; j < 2; j++){
            qDebug() << QString("h_to_0_w[%1][%2] : ").arg(i).arg(j) << w_hidden_to_output[i][j];
        }
    }

    for(u8 i = 0; i < 5; i++){
        qDebug() << QString("hidden_neuron_bias_1[%1] : ").arg(i) << hidden_neuron_bias_1[i];
    }
    for(u8 i = 0; i < 3; i++){
        qDebug() << QString("hidden_neuron_bias_2[%1] : ").arg(i) << hidden_neuron_bias_2[i];
    }

    for(u8 i = 0; i < 2; i++){
        qDebug() << QString("output_bias[%1]").arg(i) << output_bias[i];
    }

    for(u8 i = 0; i < 2; i++){
        for(u8 j = 0; j < 4; j++){
            qDebug() << QString("output[%1][%2] :").arg(i).arg(j) << calculated_output[i][j];
        }
    }

}
void MainWindow::_2_5_3_1_ann_train(void){
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
void MainWindow::_2_5_2_1_ann_train(void){
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
void MainWindow::_2_3_2_1_ann_train(void){
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
void MainWindow::_2_3_1_ann_train(void){
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

            global_error = derivative_of_sigmoid_func(Y_in) * output_error;

            w_hidden_to_output[0] += global_error * A_out;
            w_hidden_to_output[1] += global_error * B_out;
            w_hidden_to_output[2] += global_error * C_out;
            bias_output += global_error;

            errorA = derivative_of_sigmoid_func(A_in) * global_error * w_hidden_to_output[0];
            errorB = derivative_of_sigmoid_func(B_in) * global_error * w_hidden_to_output[1];
            errorC = derivative_of_sigmoid_func(C_in) * global_error * w_hidden_to_output[2];

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
    for(u8 j = 0; j < 3; j++){
        qDebug() << "h_to_o_w[i] : " << w_hidden_to_output[j];
    }
    qDebug() << "biasA" << biasA;
    qDebug() << "biasB" << biasB;
    qDebug() << "biasC" << biasC;
    qDebug() << "bias_output" << bias_output;
    for(u8 k = 0; k < 4; k++){
        qDebug() << "output : " << calculated_output[k];
    }
}
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
