#include "mainwindow.h"

void MainWindow::advanced_2_5_4_2_tryout(void){
    net_2_5_4_2.input[0][0] = 0;    net_2_5_4_2.input[1][0] = 0;
    net_2_5_4_2.input[0][1] = 0;    net_2_5_4_2.input[1][1] = 1;
    net_2_5_4_2.input[0][2] = 1;    net_2_5_4_2.input[1][2] = 0;
    net_2_5_4_2.input[0][3] = 1;    net_2_5_4_2.input[1][3] = 1;
    net_2_5_4_2.desired_output[0][0] = 0;    net_2_5_4_2.desired_output[1][0] = 1;
    net_2_5_4_2.desired_output[0][1] = 1;    net_2_5_4_2.desired_output[1][1] = 0;
    net_2_5_4_2.desired_output[0][2] = 1;    net_2_5_4_2.desired_output[1][2] = 0;
    net_2_5_4_2.desired_output[0][3] = 0;    net_2_5_4_2.desired_output[1][3] = 1;

    for(u8 i = 0; i < 5; i++){
        net_2_5_4_2.hidden_bias_1[i] = 0.1 + 0.1*i;
    }
    for(u8 i = 0; i < 4; i++){
        net_2_5_4_2.hidden_bias_2[i] = 0.1 + 0.1*i;
    }

    for(u8 i = 0; i < 2; i++){
        net_2_5_4_2.output_bias[i] = 0.1;
    }
    for(u8 i = 0; i < 2; i++){
        for(u8 j = 0; j < 5; j++){
            net_2_5_4_2.w_input_to_hidden[i][j] = 0.1;
        }
    }
    for(u8 i = 0; i < 5; i++){
        for(u8 j = 0; j < 4; j++){
            net_2_5_4_2.w_hidden_to_hidden[i][j] = 0.1;
        }
    }
    for(u8 i = 0; i < 4; i++){
        for(u8 j = 0; j < 2; j++){
            net_2_5_4_2.w_hidden_to_output[i][j] = 0.1;
        }
    }

    for(u8 i = 0; i < 2; i++){
        for(u8 j = 0; j < 4; j++){
            qDebug() << QString(" input[%1][%2] : ").arg(i).arg(j) << net_2_5_4_2.input[i][j];
        }
    }
    for(u8 i = 0; i < 2; i++){
        for(u8 j = 0; j < 4; j++){
            qDebug() << QString(" output[%1][%2] : ").arg(i).arg(j) << net_2_5_4_2.desired_output[i][j];
        }
    }


    advanced_2_5_4_2_ann_train(net_2_5_4_2.input, net_2_5_4_2.desired_output,
                             net_2_5_4_2.hidden_bias_1,net_2_5_4_2.hidden_bias_2,net_2_5_4_2.output_bias,
                             net_2_5_4_2.w_input_to_hidden,net_2_5_4_2.w_hidden_to_hidden,net_2_5_4_2.w_hidden_to_output,
                             10000, 2);

    double my_input[2] = {0,1};
    advanced_2_5_4_2_ann_test(my_input,
                             net_2_5_4_2.hidden_bias_1,net_2_5_4_2.hidden_bias_2,net_2_5_4_2.output_bias,
                             net_2_5_4_2.w_input_to_hidden,net_2_5_4_2.w_hidden_to_hidden,net_2_5_4_2.w_hidden_to_output);
}

void MainWindow::advanced_2_5_4_2_ann_test(   double input[2],
                                            double hidden_bias_1[5],double hidden_bias_2[4], double output_bias[2],
                                            double w_input_to_hidden[2][5], double w_hidden_to_hidden[5][4], double w_hidden_to_output[4][2]){

#define INPUT_COUNT     2
#define HIDDEN_COUNT_1  5
#define HIDDEN_COUNT_2  4
#define OUTPUT_COUNT    2

    double hidden_neuron_in_1[HIDDEN_COUNT_1];
    double hidden_neuron_out_1[HIDDEN_COUNT_1];
    double hidden_neuron_in_2[HIDDEN_COUNT_2];
    double hidden_neuron_out_2[HIDDEN_COUNT_2];
    double output_neuron_in[OUTPUT_COUNT];
    double output_neuron_out[OUTPUT_COUNT];

    //////////////input to hidden layer///////////////
    for(u8 i = 0; i < HIDDEN_COUNT_1; i++){
        hidden_neuron_in_1[i] = hidden_bias_1[i];
    }
    for(u8 j = 0; j < INPUT_COUNT; j++){
        for(u8 i = 0; i < HIDDEN_COUNT_1; i++){
            hidden_neuron_in_1[i] += input[j]*w_input_to_hidden[j][i];
        }
    }
    for(u8 i = 0; i < HIDDEN_COUNT_1; i++){
        hidden_neuron_out_1[i] = sigmoid_func(hidden_neuron_in_1[i]);
    }
    ///////////////////////////////////////////////////

    ////////////hidden1 to hidden2 layer///////////////
    for(u8 i = 0; i < HIDDEN_COUNT_2; i++){
        hidden_neuron_in_2[i] = hidden_bias_2[i];
    }
    for(u8 j = 0; j < HIDDEN_COUNT_1; j++){
        for(u8 i = 0; i < HIDDEN_COUNT_2; i++){
            hidden_neuron_in_2[i] += hidden_neuron_out_1[j]*w_hidden_to_hidden[j][i];
        }
    }
    for(u8 i = 0; i < HIDDEN_COUNT_2; i++){
        hidden_neuron_out_2[i] = sigmoid_func(hidden_neuron_in_2[i]);
    }
    ////////////////////////////////////////////////////

    //////////////hidden2 to output layer///////////////
    for(u8 i = 0; i < OUTPUT_COUNT; i++){
        output_neuron_in[i] = output_bias[i];
    }

    for(u8 j = 0; j < OUTPUT_COUNT; j++){
        for(u8 i = 0; i < HIDDEN_COUNT_2; i++){
            output_neuron_in[j] += hidden_neuron_out_2[i]*w_hidden_to_output[i][j];
        }
        output_neuron_out[j] = sigmoid_func(output_neuron_in[j]);
    }
    ///////////////////////////////////////////////////

    qDebug() << "**********testing***************";
    for(u8 i = 0; i < OUTPUT_COUNT; i++){
        qDebug() << QString("output[%1] :").arg(i) << output_neuron_out[i];
    }

}

void MainWindow::advanced_2_5_4_2_ann_train(  double input[2][4], double desired_output[2][4],
                                            double hidden_bias_1[5],double hidden_bias_2[4], double output_bias[2],
                                            double w_input_to_hidden[2][5],double w_hidden_to_hidden[5][4],double w_hidden_to_output[4][2],
                                            u32 epoch,double learning_rate){
#define INPUT_COUNT     2
#define HIDDEN_COUNT_1  5
#define HIDDEN_COUNT_2  4
#define OUTPUT_COUNT    2
#define IO_ARRAY_LENGTH 4

    double calculated_output_neuron[OUTPUT_COUNT][IO_ARRAY_LENGTH];
    double output_neuron_in[OUTPUT_COUNT];
    double output_neuron_out[OUTPUT_COUNT];
    double output_error[OUTPUT_COUNT];
    double global_error[OUTPUT_COUNT];

    double hidden_neuron_in_1[HIDDEN_COUNT_1];
    double hidden_neuron_out_1[HIDDEN_COUNT_1];
    double hidden_error_1[HIDDEN_COUNT_1];

    double hidden_neuron_in_2[HIDDEN_COUNT_2];
    double hidden_neuron_out_2[HIDDEN_COUNT_2];
    double hidden_error_2[HIDDEN_COUNT_2];

    for(u32 era = 0; era < epoch; era++){
        for(u8 k = 0; k < IO_ARRAY_LENGTH; k++){

            //////////////input to hidden layer///////////////
            for(u8 i = 0; i < HIDDEN_COUNT_1; i++){
                hidden_neuron_in_1[i] = hidden_bias_1[i];
            }
            for(u8 j = 0; j < INPUT_COUNT; j++){
                for(u8 i = 0; i < HIDDEN_COUNT_1; i++){
                    hidden_neuron_in_1[i] += input[j][k]*w_input_to_hidden[j][i];
                }
            }
            for(u8 i = 0; i < HIDDEN_COUNT_1; i++){
                hidden_neuron_out_1[i] = sigmoid_func(hidden_neuron_in_1[i]);
            }
            ///////////////////////////////////////////////////

            ////////////hidden1 to hidden2 layer///////////////
            for(u8 i = 0; i < HIDDEN_COUNT_2; i++){
                hidden_neuron_in_2[i] = hidden_bias_2[i];
            }
            for(u8 j = 0; j < HIDDEN_COUNT_1; j++){
                for(u8 i = 0; i < HIDDEN_COUNT_2; i++){
                    hidden_neuron_in_2[i] += hidden_neuron_out_1[j]*w_hidden_to_hidden[j][i];
                }
            }
            for(u8 i = 0; i < HIDDEN_COUNT_2; i++){
                hidden_neuron_out_2[i] = sigmoid_func(hidden_neuron_in_2[i]);
            }
            ////////////////////////////////////////////////////

            //////////////hidden2 to output layer///////////////
            for(u8 i = 0; i < OUTPUT_COUNT; i++){
                output_neuron_in[i] = output_bias[i];
            }

            for(u8 j = 0; j < OUTPUT_COUNT; j++){
                for(u8 i = 0; i < HIDDEN_COUNT_2; i++){
                    output_neuron_in[j] += hidden_neuron_out_2[i]*w_hidden_to_output[i][j];
                }
                output_neuron_out[j] = sigmoid_func(output_neuron_in[j]);
                output_error[j] = desired_output[j][k] - output_neuron_out[j];
                calculated_output_neuron[j][k] = output_neuron_out[j];
            }
            ///////////////////////////////////////////////////

            //////////////output error calculation/////////////
            for(u8 i = 0; i < OUTPUT_COUNT; i++){
                global_error[i] = derivative_of_sigmoid_func(output_neuron_in[i]) * output_error[i];
            }
            for(u8 j = 0; j < OUTPUT_COUNT; j++){
                for(u8 i = 0; i < HIDDEN_COUNT_2; i++){
                    w_hidden_to_output[i][j] += global_error[j] * hidden_neuron_out_2[i] * learning_rate;
                }
            }
            for(u8 i = 0; i < OUTPUT_COUNT; i++){
                output_bias[i] += global_error[i] * learning_rate;
            }
            ///////////////////////////////////////////////////

            /////////////hidden_2 error calculation////////////
            for(u8 i = 0; i < HIDDEN_COUNT_2; i++){
                hidden_error_2[i] = 0;
            }
            for(u8 i = 0; i < HIDDEN_COUNT_2; i++){
                for(u8 j = 0; j < OUTPUT_COUNT; j++){
                    hidden_error_2[i] += derivative_of_sigmoid_func(hidden_neuron_in_2[i]) * global_error[j] * w_hidden_to_output[i][j];
                }
            }
            for(u8 i = 0; i < HIDDEN_COUNT_2; i++){
                hidden_bias_2[i] +=hidden_error_2[i] * learning_rate;
            }
            for(u8 j = 0; j < HIDDEN_COUNT_1; j++){
                for(u8 i = 0; i < HIDDEN_COUNT_2; i++){
                    w_hidden_to_hidden[j][i] += hidden_error_2[i] * hidden_neuron_out_1[j] * learning_rate;
                }
            }
            ///////////////////////////////////////////////////

            /////////////hidden_1 error calculation////////////
            for(u8 i = 0; i < HIDDEN_COUNT_1; i++){
                hidden_error_1[i] = 0;
            }
            for(u8 i = 0; i < HIDDEN_COUNT_1; i++){
                for(u8 j = 0; j < HIDDEN_COUNT_2; j++){
                    hidden_error_1[i] += derivative_of_sigmoid_func(hidden_neuron_in_1[i]) * hidden_error_2[j] * w_hidden_to_hidden[i][j];
                }
            }
            for(u8 i = 0; i < HIDDEN_COUNT_1; i++){
                hidden_bias_1[i] +=hidden_error_1[i] * learning_rate;
            }
            for(u8 j = 0; j < INPUT_COUNT; j++){
                for(u8 i = 0; i < HIDDEN_COUNT_1; i++){
                    w_input_to_hidden[j][i] += hidden_error_1[i] * input[j][k] * learning_rate;
                }
            }
            ///////////////////////////////////////////////////

        }
        //qDebug() << "training status % " << (era*100)/epoch;
    }

    for(u8 i = 0; i < OUTPUT_COUNT; i++){
        for(u8 j = 0; j < IO_ARRAY_LENGTH; j++){
            qDebug() << QString("output[%1][%2] :").arg(i).arg(j) << calculated_output_neuron[i][j];
        }
    }
}
