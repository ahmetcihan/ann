#include "mainwindow.h"

#define INPUT_COUNT     64
#define HIDDEN_COUNT_1  32
#define HIDDEN_COUNT_2  32
#define OUTPUT_COUNT    5
#define IO_ARRAY_LENGTH 5

void MainWindow::_64_128_32_5_ann_tryout(void){
    image_to_array("/home/ahmet/Desktop/QT-Projects/ANN/zero.png",zero_image);
    image_to_array("/home/ahmet/Desktop/QT-Projects/ANN/add.png",addition_image);
    image_to_array("/home/ahmet/Desktop/QT-Projects/ANN/divide.png",divide_image);
    image_to_array("/home/ahmet/Desktop/QT-Projects/ANN/minus.png",minus_image);
    image_to_array("/home/ahmet/Desktop/QT-Projects/ANN/multiply.png",multiply_image);

    for(u8 i = 0; i < 8; i++){
        for(u8 j = 0; j < 8; j++){
            net_64_128_32_5.input[8*i + j][0] = zero_image[i][j];
        }
    }
    for(u8 i = 0; i < 8; i++){
        for(u8 j = 0; j < 8; j++){
            net_64_128_32_5.input[8*i + j][1] = addition_image[i][j];
        }
    }
    for(u8 i = 0; i < 8; i++){
        for(u8 j = 0; j < 8; j++){
            net_64_128_32_5.input[8*i + j][2] = divide_image[i][j];
        }
    }
    for(u8 i = 0; i < 8; i++){
        for(u8 j = 0; j < 8; j++){
            net_64_128_32_5.input[8*i + j][3] = minus_image[i][j];
        }
    }
    for(u8 i = 0; i < 8; i++){
        for(u8 j = 0; j < 8; j++){
            net_64_128_32_5.input[8*i + j][4] = multiply_image[i][j];
        }
    }

    for(u8 i = 0; i < 5; i++){
        for(u8 j = 0; j < 5; j++){
            net_64_128_32_5.desired_output[i][j] = 0;
        }
    }
    net_64_128_32_5.desired_output[0][0] = 1;
    net_64_128_32_5.desired_output[1][1] = 1;
    net_64_128_32_5.desired_output[2][2] = 1;
    net_64_128_32_5.desired_output[3][3] = 1;
    net_64_128_32_5.desired_output[4][4] = 1;

//    for(u8 i = 0; i < INPUT_COUNT; i++){
//        for(u8 j = 0; j < IO_ARRAY_LENGTH; j++){
//            qDebug() << QString(" input[%1][%2] : ").arg(i).arg(j) << net_64_128_32_5.input[i][j];
//        }
//    }
//    for(u8 i = 0; i < OUTPUT_COUNT; i++){
//        for(u8 j = 0; j < IO_ARRAY_LENGTH; j++){
//            qDebug() << QString(" output[%1][%2] : ").arg(i).arg(j) << net_64_128_32_5.desired_output[i][j];
//        }
//    }

    for(u8 i = 0; i < HIDDEN_COUNT_1; i++){
        net_64_128_32_5.hidden_neuron_bias_1[i] = 0.1 + i*0.01;
    }
    for(u8 i = 0; i < HIDDEN_COUNT_2; i++){
        net_64_128_32_5.hidden_neuron_bias_2[i] = 0.4 + i*0.01;
    }
    for(u8 i = 0; i < OUTPUT_COUNT; i++){
        net_64_128_32_5.output_bias[i] = 0.2;
    }

    for(u8 i = 0; i < INPUT_COUNT; i++){
        for(u8 j = 0; j < HIDDEN_COUNT_1; j++){
            net_64_128_32_5.w_input_to_hidden[i][j] = 0.1;
        }
    }
    for(u8 i = 0; i < HIDDEN_COUNT_1; i++){
        for(u8 j = 0; j < HIDDEN_COUNT_2; j++){
            net_64_128_32_5.w_hidden_to_hidden[i][j] = 0.1;
        }
    }
    for(u8 i = 0; i < HIDDEN_COUNT_2; i++){
        for(u8 j = 0; j < OUTPUT_COUNT; j++){
            net_64_128_32_5.w_hidden_to_output[i][j] = 0.1;
        }
    }
    _64_128_32_5_ann_train( net_64_128_32_5.input, net_64_128_32_5.desired_output, net_64_128_32_5.calculated_output,
                        net_64_128_32_5.hidden_neuron_bias_1,net_64_128_32_5.hidden_neuron_bias_2,net_64_128_32_5.output_bias,
                        net_64_128_32_5.w_input_to_hidden,net_64_128_32_5.w_hidden_to_hidden,net_64_128_32_5.w_hidden_to_output,
                        10000000, 0.00001);

    for(u8 i = 0; i < 5; i++){
        for(u8 j = 0; j < 5; j++){
            qDebug() << QString("desired output[%1][%2] : ").arg(i).arg(j) << net_64_128_32_5.desired_output[i][j] <<
                        QString("calculated output[%1][%2] : ").arg(i).arg(j) << net_64_128_32_5.calculated_output[i][j];
        }
    }

//    _64_128_32_5_ann_show_weights(net_64_128_32_5.hidden_neuron_bias_1,net_64_128_32_5.hidden_neuron_bias_2,net_64_128_32_5.output_bias,
//                              net_64_128_32_5.w_input_to_hidden,net_64_128_32_5.w_hidden_to_hidden,net_64_128_32_5.w_hidden_to_output);

//    image_to_array("/home/ahmet/Desktop/QT-Projects/ANN/tester.png",test_image);

//    for(u8 i = 0; i < 8; i++){
//        for(u8 j = 0; j < 8; j++){
//            net_64_128_32_5.test_input[8*i + j] = test_image[i][j];
//        }
//    }
//    _64_128_32_5_ann_test(net_64_128_32_5.test_input,
//                        net_64_128_32_5.hidden_neuron_bias_1,net_64_128_32_5.hidden_neuron_bias_2,net_64_128_32_5.output_bias,
//                        net_64_128_32_5.w_input_to_hidden,net_64_128_32_5.w_hidden_to_hidden,net_64_128_32_5.w_hidden_to_output);

}
void MainWindow::_64_128_32_5_ann_test( double input[64],
                                        double hidden_neuron_bias_1[128], double hidden_neuron_bias_2[32], double output_bias[5],
                                        double w_input_to_hidden[64][128], double w_hidden_to_hidden[128][32], double w_hidden_to_output[32][5]){
    double hidden_neuron_in_1[HIDDEN_COUNT_1];
    double hidden_neuron_out_1[HIDDEN_COUNT_1];

    double hidden_neuron_in_2[HIDDEN_COUNT_2];
    double hidden_neuron_out_2[HIDDEN_COUNT_2];

    double output_in[OUTPUT_COUNT];
    double calculated_output[OUTPUT_COUNT];

    qDebug() << "*************TESTING***********************************";
    for(u8 i = 0; i < INPUT_COUNT; i++){
        qDebug() << QString("input[%1] :").arg(i) << input[i];
    }


    for(u8 i = 0; i < HIDDEN_COUNT_1; i++){
        hidden_neuron_in_1[i] = hidden_neuron_bias_1[i];
    }
    for(u8 i = 0; i < HIDDEN_COUNT_1; i++){
        for(u8 j = 0; j < INPUT_COUNT; j++){
            hidden_neuron_in_1[i] += input[j]*w_input_to_hidden[j][i];
        }
    }
    for(u8 i = 0; i < HIDDEN_COUNT_1; i++){
        hidden_neuron_out_1[i] = sigmoid_func(hidden_neuron_in_1[i]);
    }
    for(u8 i = 0; i < HIDDEN_COUNT_2; i++){
        hidden_neuron_in_2[i] = hidden_neuron_bias_2[i];
    }
    for(u8 i = 0; i < HIDDEN_COUNT_2; i++){
        for(u8 j = 0; j < HIDDEN_COUNT_1; j++){
            hidden_neuron_in_2[i] += hidden_neuron_out_1[j]*w_hidden_to_hidden[j][i];
        }
    }
    for(u8 i = 0; i < HIDDEN_COUNT_2; i++){
        hidden_neuron_out_2[i] = sigmoid_func(hidden_neuron_in_2[i]);
    }

    for(u8 j = 0; j < OUTPUT_COUNT; j++){
        output_in[j] =  output_bias[j];
        for(u8 i = 0; i < HIDDEN_COUNT_2; i++){
            output_in[j] += hidden_neuron_out_2[i]*w_hidden_to_output[i][j];
        }
        calculated_output[j]   = sigmoid_func(output_in[j]);
    }

    for(u8 i = 0; i < OUTPUT_COUNT; i++){
        qDebug() << QString("output[%1] :").arg(i) << calculated_output[i];
    }
}

void MainWindow::_64_128_32_5_ann_train(double input[64][5], double desired_output[5][5], double calculated_output[5][5],
                                        double hidden_neuron_bias_1[128], double hidden_neuron_bias_2[64], double output_bias[5],
                                        double w_input_to_hidden[64][128], double w_hidden_to_hidden[128][32], double w_hidden_to_output[32][5],
                                        u32 epoch, double learning_rate){

    double hidden_neuron_in_1[HIDDEN_COUNT_1];
    double hidden_neuron_out_1[HIDDEN_COUNT_1];
    double hidden_neuron_error_1[HIDDEN_COUNT_1];

    double hidden_neuron_in_2[HIDDEN_COUNT_2];
    double hidden_neuron_out_2[HIDDEN_COUNT_2];
    double hidden_neuron_error_2[HIDDEN_COUNT_2];

    double output_error[OUTPUT_COUNT];
    double global_error[OUTPUT_COUNT];
    double output_in[OUTPUT_COUNT];
    double output_out[OUTPUT_COUNT];

    for(u32 era = 0; era < epoch; era++){
        for(u8 k = 0; k < IO_ARRAY_LENGTH; k++){

            for(u8 i = 0; i < HIDDEN_COUNT_1; i++){
                hidden_neuron_in_1[i] = hidden_neuron_bias_1[i];
            }
            for(u8 i = 0; i < HIDDEN_COUNT_1; i++){
                for(u8 j = 0; j < INPUT_COUNT; j++){
                    hidden_neuron_in_1[i] += input[j][k]*w_input_to_hidden[j][i];
                }
            }
            for(u8 i = 0; i < HIDDEN_COUNT_1; i++){
                hidden_neuron_out_1[i] = sigmoid_func(hidden_neuron_in_1[i]);
            }
            for(u8 i = 0; i < HIDDEN_COUNT_2; i++){
                hidden_neuron_in_2[i] = hidden_neuron_bias_2[i];
            }
            for(u8 i = 0; i < HIDDEN_COUNT_2; i++){
                for(u8 j = 0; j < HIDDEN_COUNT_1; j++){
                    hidden_neuron_in_2[i] += hidden_neuron_out_1[j]*w_hidden_to_hidden[j][i];
                }
            }
            for(u8 i = 0; i < HIDDEN_COUNT_2; i++){
                hidden_neuron_out_2[i] = sigmoid_func(hidden_neuron_in_2[i]);
            }

            for(u8 j = 0; j < OUTPUT_COUNT; j++){
                output_in[j] =  output_bias[j];
                for(u8 i = 0; i < HIDDEN_COUNT_2; i++){
                    output_in[j] += hidden_neuron_out_2[i]*w_hidden_to_output[i][j];
                }
                output_out[j]   = sigmoid_func(output_in[j]);
                output_error[j] = desired_output[j][k] - output_out[j];
                calculated_output[j][k] = output_out[j];
                global_error[j] = derivative_of_sigmoid_func(output_in[j]) * output_error[j];
            }

            for(u8 j = 0; j < OUTPUT_COUNT; j++){
                for(u8 i = 0; i < HIDDEN_COUNT_2; i++){
                    w_hidden_to_output[i][j] += global_error[j] * hidden_neuron_out_2[i] * learning_rate;
                }
            }

            for(u8 i = 0; i < OUTPUT_COUNT; i++){
                output_bias[i] += global_error[i] * learning_rate;
            }
            for(u8 i = 0; i < HIDDEN_COUNT_2; i++){
                hidden_neuron_error_2[i] = 0;
            }

            for(u8 i = 0; i < HIDDEN_COUNT_2; i++){
                for(u8 j = 0; j < OUTPUT_COUNT; j++){
                    hidden_neuron_error_2[i] += derivative_of_sigmoid_func(hidden_neuron_in_2[i]) * global_error[j] * w_hidden_to_output[i][j];
                }
            }
            for(u8 i = 0; i < HIDDEN_COUNT_2; i++){
                for(u8 j = 0; j < HIDDEN_COUNT_1; j++){
                    w_hidden_to_hidden[j][i] += hidden_neuron_error_2[i] * hidden_neuron_out_1[j] * learning_rate;
                }
            }

            for(u8 i = 0; i < HIDDEN_COUNT_2; i++){
                hidden_neuron_bias_2[i] += hidden_neuron_error_2[i] * learning_rate;
            }

            for(u8 i = 0; i < HIDDEN_COUNT_1; i++){
                hidden_neuron_error_1[i] = 0;
            }

            for(u8 i = 0; i < HIDDEN_COUNT_1; i++){
                for(u8 j = 0; j < HIDDEN_COUNT_2; j++){
                    hidden_neuron_error_1[i] +=  derivative_of_sigmoid_func(hidden_neuron_in_1[i]) * hidden_neuron_error_2[j] * w_hidden_to_hidden[i][j];
                }
            }
            for(u8 i = 0; i < INPUT_COUNT; i++){
                for(u8 j = 0; j < HIDDEN_COUNT_1; j++){
                    w_input_to_hidden[i][j] += hidden_neuron_error_1[j] * input[i][k] * learning_rate;
                }
            }

            for(u8 i = 0; i < HIDDEN_COUNT_1; i++){
                hidden_neuron_bias_1[i] +=hidden_neuron_error_1[i] * learning_rate;
            }
        }
        qDebug() << "training status % " << (era*100)/epoch;
    }
}
void MainWindow::_64_128_32_5_ann_show_weights( double hidden_neuron_bias_1[128], double hidden_neuron_bias_2[32], double output_bias[5],
                                            double w_input_to_hidden[64][128], double w_hidden_to_hidden[128][32], double w_hidden_to_output[32][5]){
    for(u8 i = 0; i < INPUT_COUNT; i++){
        for(u8 j = 0; j < HIDDEN_COUNT_1; j++){
            qDebug() << QString("in_to_h_w[%1][%2] : ").arg(i).arg(j) << w_input_to_hidden[i][j];
        }
    }
    for(u8 i = 0; i < HIDDEN_COUNT_1; i++){
        for(u8 j = 0; j < HIDDEN_COUNT_2; j++){
            qDebug() << QString("h_to_h_w[%1][%2] : ").arg(i).arg(j) << w_hidden_to_hidden[i][j];
        }
    }
    for(u8 i = 0; i < HIDDEN_COUNT_2; i++){
        for(u8 j = 0; j < OUTPUT_COUNT; j++){
            qDebug() << QString("h_to_0_w[%1][%2] : ").arg(i).arg(j) << w_hidden_to_output[i][j];
        }
    }

    for(u8 i = 0; i < HIDDEN_COUNT_1; i++){
        qDebug() << QString("hidden_neuron_bias_1[%1] : ").arg(i) << hidden_neuron_bias_1[i];
    }
    for(u8 i = 0; i < HIDDEN_COUNT_2; i++){
        qDebug() << QString("hidden_neuron_bias_2[%1] : ").arg(i) << hidden_neuron_bias_2[i];
    }

    for(u8 i = 0; i < OUTPUT_COUNT; i++){
        qDebug() << QString("output_bias[%1]").arg(i) << output_bias[i];
    }

}
