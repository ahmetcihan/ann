#include "mainwindow.h"
#include "ui_mainwindow.h"

void ann::advanced_64_128_5_ann_train(double input[64][5], double desired_output[5][5], double calculated_output[5][5],
                                            double hidden_bias[128], double output_bias[5],
                                            double w_input_to_hidden[64][128], double w_hidden_to_output[128][5],
                                            u32 epoch, double learning_rate){
#define INPUT_COUNT 64
#define HIDDEN_COUNT 128
#define OUTPUT_COUNT 5
#define IO_ARRAY_LENGTH 5

    double output_neuron_in[OUTPUT_COUNT];
    double output_neuron_out[OUTPUT_COUNT];
    double output_error[OUTPUT_COUNT];
    double global_error[OUTPUT_COUNT];

    double hidden_neuron_in[HIDDEN_COUNT];
    double hidden_neuron_out[HIDDEN_COUNT];
    double hidden_error[HIDDEN_COUNT];

    for(u32 era = 0; era < epoch; era++){
        for(u8 k = 0; k < IO_ARRAY_LENGTH; k++){

            for(u8 i = 0; i < HIDDEN_COUNT; i++){
                hidden_neuron_in[i] = hidden_bias[i];
            }

            for(u8 j = 0; j < INPUT_COUNT; j++){
                for(u8 i = 0; i < HIDDEN_COUNT; i++){
                    hidden_neuron_in[i] += input[j][k]*w_input_to_hidden[j][i];
                }
            }

            for(u8 i = 0; i < HIDDEN_COUNT; i++){
                hidden_neuron_out[i] = sigmoid_func(hidden_neuron_in[i]);
            }

            for(u8 i = 0; i < OUTPUT_COUNT; i++){
                output_neuron_in[i] = output_bias[i];
            }

            for(u8 j = 0; j < OUTPUT_COUNT; j++){
                for(u8 i = 0; i < HIDDEN_COUNT; i++){
                    output_neuron_in[j] += hidden_neuron_out[i]*w_hidden_to_output[i][j];
                }
                output_neuron_out[j] = sigmoid_func(output_neuron_in[j]);
                output_error[j] = desired_output[j][k] - output_neuron_out[j];
                calculated_output[j][k] = output_neuron_out[j];
            }

            for(u8 i = 0; i < OUTPUT_COUNT; i++){
                global_error[i] = derivative_of_sigmoid_func(output_neuron_in[i]) * output_error[i];
            }

            for(u8 j = 0; j < OUTPUT_COUNT; j++){
                for(u8 i = 0; i < HIDDEN_COUNT; i++){
                    w_hidden_to_output[i][j] += global_error[j] * hidden_neuron_out[i] * learning_rate;
                }
            }

            for(u8 i = 0; i < OUTPUT_COUNT; i++){
                output_bias[i] += global_error[i] * learning_rate;
            }

            for(u8 i = 0; i < HIDDEN_COUNT; i++){
                hidden_error[i] = 0;
            }

            for(u8 i = 0; i < HIDDEN_COUNT; i++){
                for(u8 j = 0; j < OUTPUT_COUNT; j++){
                    hidden_error[i] += derivative_of_sigmoid_func(hidden_neuron_in[i]) * global_error[j] * w_hidden_to_output[i][j];
                }
            }

            for(u8 j = 0; j < INPUT_COUNT; j++){
                for(u8 i = 0; i < HIDDEN_COUNT; i++){
                    w_input_to_hidden[j][i] += hidden_error[i] * input[j][k] * learning_rate;
                }
            }

            for(u8 i = 0; i < HIDDEN_COUNT; i++){
                hidden_bias[i] +=hidden_error[i] * learning_rate;
            }
        }
        //qDebug() << "training status % " << (era*100)/epoch;
        epoch_status = (era*100)/epoch;

    }
    qDebug() << "training is FINISHED!!";

}
void ann::advanced_64_128_5_ann_test(double input[64],
                                double hidden_bias[128], double output_bias[5],
                                double w_input_to_hidden[64][128],double w_hidden_to_output[128][5]){

#define INPUT_COUNT 64
#define HIDDEN_COUNT 128
#define OUTPUT_COUNT 5

    double hidden_neuron_in[HIDDEN_COUNT];
    double hidden_neuron_out[HIDDEN_COUNT];
    double output_neuron_in[OUTPUT_COUNT];
    double output_neuron_out[OUTPUT_COUNT];

    for(u8 i = 0; i < HIDDEN_COUNT; i++){
        hidden_neuron_in[i] = hidden_bias[i];
    }

    for(u8 j = 0; j < INPUT_COUNT; j++){
        for(u8 i = 0; i < HIDDEN_COUNT; i++){
            hidden_neuron_in[i] += input[j]*w_input_to_hidden[j][i];
        }
    }

    for(u8 i = 0; i < HIDDEN_COUNT; i++){
        hidden_neuron_out[i] = sigmoid_func(hidden_neuron_in[i]);
    }

    for(u8 i = 0; i < OUTPUT_COUNT; i++){
        output_neuron_in[i] = output_bias[i];
    }

    for(u8 j = 0; j < OUTPUT_COUNT; j++){
        for(u8 i = 0; i < HIDDEN_COUNT; i++){
            output_neuron_in[j] += hidden_neuron_out[i]*w_hidden_to_output[i][j];
        }
        output_neuron_out[j] = sigmoid_func(output_neuron_in[j]);
    }

    qDebug() << "**********testing***************";
    for(u8 i = 0; i < OUTPUT_COUNT; i++){
        qDebug() << QString("output[%1] :").arg(i) << output_neuron_out[i];
    }

    qDebug() << "% " << 100*output_neuron_out[0] << "\t" << "ihtimalle sifir isareti";
    qDebug() << "% " << 100*output_neuron_out[1] << "\t" << "ihtimalle toplama isareti";
    qDebug() << "% " << 100*output_neuron_out[2] << "\t" << "ihtimalle bolme isareti";
    qDebug() << "% " << 100*output_neuron_out[3] << "\t" << "ihtimalle cikarma isareti";
    qDebug() << "% " << 100*output_neuron_out[4] << "\t" << "ihtimalle carpma isareti";

    u8 max_value_index = 0;
    double max_value = 0;
    QString str = "";

    for(u8 i = 0; i < 5; i++){
        if(output_neuron_out[i] > max_value){
            max_value = output_neuron_out[i];
            max_value_index = i;
        }
    }
    if(max_value_index == 0)    str = QString("% %1 ihtimalle sifir isareti").arg((u32)(100*output_neuron_out[0]));
    if(max_value_index == 1)    str = QString("% %1 ihtimalle toplama isareti").arg((u32)(100*output_neuron_out[1]));
    if(max_value_index == 2)    str = QString("% %1 ihtimalle bolme isareti").arg((u32)(100*output_neuron_out[2]));
    if(max_value_index == 3)    str = QString("% %1 ihtimalle cikarma isareti").arg((u32)(100*output_neuron_out[3]));
    if(max_value_index == 4)    str = QString("% %1 ihtimalle carpma isareti").arg((u32)(100*output_neuron_out[4]));

    mainwindow->ui->label_64_128_5_test->setText(str);

}

