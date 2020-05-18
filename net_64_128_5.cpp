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

    qDebug() << "% " << 100*output_neuron_out[0] << "\t" << "ihtimal sifir isareti";
    qDebug() << "% " << 100*output_neuron_out[1] << "\t" << "ihtimal toplama isareti";
    qDebug() << "% " << 100*output_neuron_out[2] << "\t" << "ihtimal bolme isareti";
    qDebug() << "% " << 100*output_neuron_out[3] << "\t" << "ihtimal cikarma isareti";
    qDebug() << "% " << 100*output_neuron_out[4] << "\t" << "ihtimal carpma isareti";

    u8 max_value_index = 0;
    double max_value = 0;
    QString str = "";

    for(u8 i = 0; i < 5; i++){
        if(output_neuron_out[i] > max_value){
            max_value = output_neuron_out[i];
            max_value_index = i;
        }
    }
    if(max_value_index == 0)    str = QString("% %1 ihtimal sifir isareti").arg((u32)(100*output_neuron_out[0]));
    if(max_value_index == 1)    str = QString("% %1 ihtimal toplama isareti").arg((u32)(100*output_neuron_out[1]));
    if(max_value_index == 2)    str = QString("% %1 ihtimal bolme isareti").arg((u32)(100*output_neuron_out[2]));
    if(max_value_index == 3)    str = QString("% %1 ihtimal cikarma isareti").arg((u32)(100*output_neuron_out[3]));
    if(max_value_index == 4)    str = QString("% %1 ihtimal carpma isareti").arg((u32)(100*output_neuron_out[4]));

    mainwindow->ui->label_64_128_5_test->setText(str);

}

void MainWindow::_64_128_5_random_initilize_handler(void){
    image_to_array("/home/ahmet/Desktop/QT-Projects/ANN/zero.png",zero_image);
    image_to_array("/home/ahmet/Desktop/QT-Projects/ANN/add.png",addition_image);
    image_to_array("/home/ahmet/Desktop/QT-Projects/ANN/divide.png",divide_image);
    image_to_array("/home/ahmet/Desktop/QT-Projects/ANN/minus.png",minus_image);
    image_to_array("/home/ahmet/Desktop/QT-Projects/ANN/multiply.png",multiply_image);

    for(u8 i = 0; i < 8; i++){
        for(u8 j = 0; j < 8; j++){
            ann_class->net_64_128_5.input[8*i + j][0] = zero_image[i][j];
        }
    }
    for(u8 i = 0; i < 8; i++){
        for(u8 j = 0; j < 8; j++){
            ann_class->net_64_128_5.input[8*i + j][1] = addition_image[i][j];
        }
    }
    for(u8 i = 0; i < 8; i++){
        for(u8 j = 0; j < 8; j++){
            ann_class->net_64_128_5.input[8*i + j][2] = divide_image[i][j];
        }
    }
    for(u8 i = 0; i < 8; i++){
        for(u8 j = 0; j < 8; j++){
            ann_class->net_64_128_5.input[8*i + j][3] = minus_image[i][j];
        }
    }
    for(u8 i = 0; i < 8; i++){
        for(u8 j = 0; j < 8; j++){
            ann_class->net_64_128_5.input[8*i + j][4] = multiply_image[i][j];
        }
    }

    for(u8 i = 0; i < 5; i++){
        for(u8 j = 0; j < 5; j++){
            ann_class->net_64_128_5.desired_output[i][j] = 0;
        }
    }
    ann_class->net_64_128_5.desired_output[0][0] = 1;
    ann_class->net_64_128_5.desired_output[1][1] = 1;
    ann_class->net_64_128_5.desired_output[2][2] = 1;
    ann_class->net_64_128_5.desired_output[3][3] = 1;
    ann_class->net_64_128_5.desired_output[4][4] = 1;

    for(u8 i = 0; i < 128; i++){
        ann_class->net_64_128_5.hidden_bias[i] = 0.1 + 0.01*i;
    }
    for(u8 i = 0; i < 5; i++){
        ann_class->net_64_128_5.output_bias[i] = 0.2;
    }

    for(u8 i = 0; i < 64; i++){
        for(u8 j = 0; j < 128; j++){
            ann_class->net_64_128_5.w_input_to_hidden[i][j] = 0.1;
        }
    }
    for(u8 i = 0; i < 128; i++){
        for(u8 j = 0; j < 5; j++){
            ann_class->net_64_128_5.w_hidden_to_output[i][j] = 0.1;
        }
    }

    ui->label_64_128_5_random_initilize->setText("Initilized randomly");
}
void MainWindow::_64_128_5_train_handler(void){
    ann_class->train_status = 1;
}
void MainWindow::_64_128_5_test_handler(void){
    image_to_array("/home/ahmet/Desktop/QT-Projects/ANN/tester.png",test_image);

    for(u8 i = 0; i < 8; i++){
        for(u8 j = 0; j < 8; j++){
            ann_class->net_64_128_5.test_input[8*i + j] = test_image[i][j];
        }
    }

    ann_class->advanced_64_128_5_ann_test(ann_class->net_64_128_5.test_input,
                             ann_class->net_64_128_5.hidden_bias,ann_class->net_64_128_5.output_bias,
                             ann_class->net_64_128_5.w_input_to_hidden,ann_class->net_64_128_5.w_hidden_to_output);

}
void MainWindow::_64_128_5_show_weights_handler(void){
    for(u8 i = 0; i < 128; i++){
        qDebug() << QString("hidden_bias[%1] : ").arg(i) << ann_class->net_64_128_5.hidden_bias[i];
    }
    for(u8 i = 0; i < 5; i++){
        qDebug() << QString("output_bias[%1] : ").arg(i) << ann_class->net_64_128_5.output_bias[i];
    }

    for(u8 i = 0; i < 64; i++){
        for(u8 j = 0; j < 128; j++){
            qDebug() << QString("w_input_to_hidden[%1][%2] : ").arg(i).arg(j) << ann_class->net_64_128_5.w_input_to_hidden[i][j];
        }
    }
    for(u8 i = 0; i < 128; i++){
        for(u8 j = 0; j < 5; j++){
            qDebug() << QString("w_hidden_to_output[%1][%2] : ").arg(i).arg(j) << ann_class->net_64_128_5.w_hidden_to_output[i][j];
        }
    }
    ui->label_64_128_5_show_weights->setText("Showed..");
}
void MainWindow::_64_128_5_save_weights_handler(void){
    QSettings settings("weights_64_128_5.ini",QSettings::IniFormat);

    settings.beginGroup("w");

    for(u8 i = 0; i < 128; i++){
        settings.setValue(QString("hb-%1").arg(i),ann_class->net_64_128_5.hidden_bias[i]);
    }
    for(u8 i = 0; i < 5; i++){
        settings.setValue(QString("ob-%1").arg(i),ann_class->net_64_128_5.output_bias[i]);
    }

    for(u8 i = 0; i < 64; i++){
        for(u8 j = 0; j < 128; j++){
            settings.setValue(QString("i2h-%1-%2").arg(i).arg(j),ann_class->net_64_128_5.w_input_to_hidden[i][j]);
        }
    }
    for(u8 i = 0; i < 128; i++){
        for(u8 j = 0; j < 5; j++){
            settings.setValue(QString("h2o-%1-%2").arg(i).arg(j),ann_class->net_64_128_5.w_hidden_to_output[i][j]);
        }
    }
    settings.endGroup();
    settings.sync();
    QProcess::execute("sync");

    ui->label_64_128_5_save_weights->setText("Saved..");
}
void MainWindow::_64_128_5_load_saved_weights_handler(void){
    QSettings settings("weights_64_128_5.ini",QSettings::IniFormat);

    for(u8 i = 0; i < 128; i++){
        ann_class->net_64_128_5.hidden_bias[i] = settings.value(QString("w/hb-%1").arg(i)).toDouble();
    }
    for(u8 i = 0; i < 5; i++){
        ann_class->net_64_128_5.output_bias[i] = settings.value(QString("w/ob-%1").arg(i)).toDouble();
    }

    for(u8 i = 0; i < 64; i++){
        for(u8 j = 0; j < 128; j++){
            ann_class->net_64_128_5.w_input_to_hidden[i][j] = settings.value(QString("w/i2h-%1-%2").arg(i).arg(j)).toDouble();
        }
    }
    for(u8 i = 0; i < 128; i++){
        for(u8 j = 0; j < 5; j++){
            ann_class->net_64_128_5.w_hidden_to_output[i][j] = settings.value(QString("w/h2o-%1-%2").arg(i).arg(j)).toDouble();
        }
    }
    ui->label_64_128_5_load_saved_weights->setText("Loaded..");
}
