#include "mainwindow.h"
#include "ui_mainwindow.h"

#define INPUT_COUNT     256
#define HIDDEN_COUNT_1  256
#define HIDDEN_COUNT_2  128
#define OUTPUT_COUNT    26
#define IO_ARRAY_LENGTH 26
#define INPUT_SET       3

double ann::_256_512_512_26_ann_calculate_total_error(void){
    double total_error = 0;
    double aux;

    for(u8 i = 0; i < 26; i++){
        for(u8 j = 0; j < 26; j++){
            aux = net_256_512_512_26.desired_output[i][j] - net_256_512_512_26.calculated_output[i][j];
            aux = aux * aux;
            total_error += aux;
        }
    }
    return total_error;
}
void ann::_256_512_512_26_ann_test( double input[256],
                                    double hidden_neuron_bias_1[512], double hidden_neuron_bias_2[512], double output_bias[26],
                                    double w_input_to_hidden[256][512], double w_hidden_to_hidden[512][512], double w_hidden_to_output[512][26]){
    double hidden_neuron_in_1[HIDDEN_COUNT_1];
    double hidden_neuron_out_1[HIDDEN_COUNT_1];

    double hidden_neuron_in_2[HIDDEN_COUNT_2];
    double hidden_neuron_out_2[HIDDEN_COUNT_2];

    double output_in[OUTPUT_COUNT];
    double calculated_output[OUTPUT_COUNT];

    for(u16 i = 0; i < INPUT_COUNT; i++){
        qDebug() << QString("input[%1] :").arg(i) << input[i];
    }

    for(u16 i = 0; i < HIDDEN_COUNT_1; i++){
        hidden_neuron_in_1[i] = hidden_neuron_bias_1[i];
    }
    for(u16 i = 0; i < HIDDEN_COUNT_1; i++){
        for(u16 j = 0; j < INPUT_COUNT; j++){
            hidden_neuron_in_1[i] += input[j]*w_input_to_hidden[j][i];
        }
    }
    for(u16 i = 0; i < HIDDEN_COUNT_1; i++){
        hidden_neuron_out_1[i] = sigmoid_func(hidden_neuron_in_1[i]);
    }
    for(u16 i = 0; i < HIDDEN_COUNT_2; i++){
        hidden_neuron_in_2[i] = hidden_neuron_bias_2[i];
    }
    for(u16 i = 0; i < HIDDEN_COUNT_2; i++){
        for(u16 j = 0; j < HIDDEN_COUNT_1; j++){
            hidden_neuron_in_2[i] += hidden_neuron_out_1[j]*w_hidden_to_hidden[j][i];
        }
    }
    for(u16 i = 0; i < HIDDEN_COUNT_2; i++){
        hidden_neuron_out_2[i] = sigmoid_func(hidden_neuron_in_2[i]);
    }

    for(u16 j = 0; j < OUTPUT_COUNT; j++){
        output_in[j] =  output_bias[j];
        for(u16 i = 0; i < HIDDEN_COUNT_2; i++){
            output_in[j] += hidden_neuron_out_2[i]*w_hidden_to_output[i][j];
        }
        calculated_output[j]   = sigmoid_func(output_in[j]);
    }

    qDebug() << "**********testing***************";
    for(u16 i = 0; i < OUTPUT_COUNT; i++){
        qDebug() << QString("output[%1] :").arg(i) << calculated_output[i];
    }

    qDebug() << "% " << 100*calculated_output[0] << "\t" << "ihtimal A";
    qDebug() << "% " << 100*calculated_output[1] << "\t" << "ihtimal B";
    qDebug() << "% " << 100*calculated_output[2] << "\t" << "ihtimal C";
    qDebug() << "% " << 100*calculated_output[3] << "\t" << "ihtimal D";
    qDebug() << "% " << 100*calculated_output[4] << "\t" << "ihtimal E";
    qDebug() << "% " << 100*calculated_output[5] << "\t" << "ihtimal F";
    qDebug() << "% " << 100*calculated_output[6] << "\t" << "ihtimal G";
    qDebug() << "% " << 100*calculated_output[7] << "\t" << "ihtimal H";
    qDebug() << "% " << 100*calculated_output[8] << "\t" << "ihtimal I";
    qDebug() << "% " << 100*calculated_output[9] << "\t" << "ihtimal J";
    qDebug() << "% " << 100*calculated_output[10] << "\t" << "ihtimal K";
    qDebug() << "% " << 100*calculated_output[11] << "\t" << "ihtimal L";
    qDebug() << "% " << 100*calculated_output[12] << "\t" << "ihtimal M";
    qDebug() << "% " << 100*calculated_output[13] << "\t" << "ihtimal N";
    qDebug() << "% " << 100*calculated_output[14] << "\t" << "ihtimal O";
    qDebug() << "% " << 100*calculated_output[15] << "\t" << "ihtimal P";
    qDebug() << "% " << 100*calculated_output[16] << "\t" << "ihtimal Q";
    qDebug() << "% " << 100*calculated_output[17] << "\t" << "ihtimal R";
    qDebug() << "% " << 100*calculated_output[18] << "\t" << "ihtimal S";
    qDebug() << "% " << 100*calculated_output[19] << "\t" << "ihtimal T";
    qDebug() << "% " << 100*calculated_output[20] << "\t" << "ihtimal U";
    qDebug() << "% " << 100*calculated_output[21] << "\t" << "ihtimal V";
    qDebug() << "% " << 100*calculated_output[22] << "\t" << "ihtimal W";
    qDebug() << "% " << 100*calculated_output[23] << "\t" << "ihtimal X";
    qDebug() << "% " << 100*calculated_output[24] << "\t" << "ihtimal Y";
    qDebug() << "% " << 100*calculated_output[25] << "\t" << "ihtimal Z";

    u16 max_value_index = 0;
    double max_value = 0;
    QString str = "";

    for(u16 i = 0; i < 26; i++){
        if(calculated_output[i] > max_value){
            max_value = calculated_output[i];
            max_value_index = i;
        }
    }
    double out_strict[26] = {0};

    for(u16 i = 0; i < 26; i++){
        if(calculated_output[i] > 1.0){
            out_strict[i] = calculated_output[i] - 1.0;
            calculated_output[i] = calculated_output[i] - 2*out_strict[i];
        }
    }

    if(max_value_index == 0)    str = QString("% %1 ihtimal A").arg((u32)(100*calculated_output[0]));
    if(max_value_index == 1)    str = QString("% %1 ihtimal B").arg((u32)(100*calculated_output[1]));
    if(max_value_index == 2)    str = QString("% %1 ihtimal C").arg((u32)(100*calculated_output[2]));
    if(max_value_index == 3)    str = QString("% %1 ihtimal D").arg((u32)(100*calculated_output[3]));
    if(max_value_index == 4)    str = QString("% %1 ihtimal E").arg((u32)(100*calculated_output[4]));
    if(max_value_index == 5)    str = QString("% %1 ihtimal F").arg((u32)(100*calculated_output[5]));
    if(max_value_index == 6)    str = QString("% %1 ihtimal G").arg((u32)(100*calculated_output[6]));
    if(max_value_index == 7)    str = QString("% %1 ihtimal H").arg((u32)(100*calculated_output[7]));
    if(max_value_index == 8)    str = QString("% %1 ihtimal I").arg((u32)(100*calculated_output[8]));
    if(max_value_index == 9)    str = QString("% %1 ihtimal J").arg((u32)(100*calculated_output[9]));
    if(max_value_index == 10)    str = QString("% %1 ihtimal K").arg((u32)(100*calculated_output[10]));
    if(max_value_index == 11)    str = QString("% %1 ihtimal L").arg((u32)(100*calculated_output[11]));
    if(max_value_index == 12)    str = QString("% %1 ihtimal M").arg((u32)(100*calculated_output[12]));
    if(max_value_index == 13)    str = QString("% %1 ihtimal N").arg((u32)(100*calculated_output[13]));
    if(max_value_index == 14)    str = QString("% %1 ihtimal O").arg((u32)(100*calculated_output[14]));
    if(max_value_index == 15)    str = QString("% %1 ihtimal P").arg((u32)(100*calculated_output[15]));
    if(max_value_index == 16)    str = QString("% %1 ihtimal Q").arg((u32)(100*calculated_output[16]));
    if(max_value_index == 17)    str = QString("% %1 ihtimal R").arg((u32)(100*calculated_output[17]));
    if(max_value_index == 18)    str = QString("% %1 ihtimal S").arg((u32)(100*calculated_output[18]));
    if(max_value_index == 19)    str = QString("% %1 ihtimal T").arg((u32)(100*calculated_output[19]));
    if(max_value_index == 20)    str = QString("% %1 ihtimal U").arg((u32)(100*calculated_output[20]));
    if(max_value_index == 21)    str = QString("% %1 ihtimal V").arg((u32)(100*calculated_output[21]));
    if(max_value_index == 22)    str = QString("% %1 ihtimal W").arg((u32)(100*calculated_output[22]));
    if(max_value_index == 23)    str = QString("% %1 ihtimal X").arg((u32)(100*calculated_output[23]));
    if(max_value_index == 24)    str = QString("% %1 ihtimal Y").arg((u32)(100*calculated_output[24]));
    if(max_value_index == 25)    str = QString("% %1 ihtimal Z").arg((u32)(100*calculated_output[25]));

    mainwindow->ui->label_256_512_512_26_test->setText(str);


}

void ann::_256_512_512_26_ann_train(double input[256][26*4], double desired_output[26][26], double calculated_output[26][26],
                                    double hidden_neuron_bias_1[512], double hidden_neuron_bias_2[512], double output_bias[26],
                                    double w_input_to_hidden[256][512], double w_hidden_to_hidden[512][512], double w_hidden_to_output[512][26],
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

    double inset_error[INPUT_SET];

    for(u32 era = 0; era < epoch; era++){
        for(u16 inset = 0; inset < INPUT_SET; inset++){
            for(u16 k = 0; k < IO_ARRAY_LENGTH; k++){

                for(u16 i = 0; i < HIDDEN_COUNT_1; i++){
                    hidden_neuron_in_1[i] = hidden_neuron_bias_1[i];
                }
                for(u16 i = 0; i < HIDDEN_COUNT_1; i++){
                    for(u16 j = 0; j < INPUT_COUNT; j++){
                        hidden_neuron_in_1[i] += input[j][k + 26*inset]*w_input_to_hidden[j][i];
                    }
                }
                for(u16 i = 0; i < HIDDEN_COUNT_1; i++){
                    hidden_neuron_out_1[i] = sigmoid_func(hidden_neuron_in_1[i]);
                }
                for(u16 i = 0; i < HIDDEN_COUNT_2; i++){
                    hidden_neuron_in_2[i] = hidden_neuron_bias_2[i];
                }
                for(u16 i = 0; i < HIDDEN_COUNT_2; i++){
                    for(u16 j = 0; j < HIDDEN_COUNT_1; j++){
                        hidden_neuron_in_2[i] += hidden_neuron_out_1[j]*w_hidden_to_hidden[j][i];
                    }
                }
                for(u16 i = 0; i < HIDDEN_COUNT_2; i++){
                    hidden_neuron_out_2[i] = sigmoid_func(hidden_neuron_in_2[i]);
                }

                for(u16 j = 0; j < OUTPUT_COUNT; j++){
                    output_in[j] =  output_bias[j];
                    for(u16 i = 0; i < HIDDEN_COUNT_2; i++){
                        output_in[j] += hidden_neuron_out_2[i]*w_hidden_to_output[i][j];
                    }
                    output_out[j]   = sigmoid_func(output_in[j]);
                    output_error[j] = desired_output[j][k] - output_out[j];
                    calculated_output[j][k] = output_out[j];
                    global_error[j] = derivative_of_sigmoid_func(output_in[j]) * output_error[j];
                }

                for(u16 j = 0; j < OUTPUT_COUNT; j++){
                    for(u16 i = 0; i < HIDDEN_COUNT_2; i++){
                        w_hidden_to_output[i][j] += global_error[j] * hidden_neuron_out_2[i] * learning_rate;
                    }
                }

                for(u16 i = 0; i < OUTPUT_COUNT; i++){
                    output_bias[i] += global_error[i] * learning_rate;
                }
                for(u16 i = 0; i < HIDDEN_COUNT_2; i++){
                    hidden_neuron_error_2[i] = 0;
                }

                for(u16 i = 0; i < HIDDEN_COUNT_2; i++){
                    for(u16 j = 0; j < OUTPUT_COUNT; j++){
                        hidden_neuron_error_2[i] += derivative_of_sigmoid_func(hidden_neuron_in_2[i]) * global_error[j] * w_hidden_to_output[i][j];
                    }
                }
                for(u16 i = 0; i < HIDDEN_COUNT_2; i++){
                    for(u16 j = 0; j < HIDDEN_COUNT_1; j++){
                        w_hidden_to_hidden[j][i] += hidden_neuron_error_2[i] * hidden_neuron_out_1[j] * learning_rate;
                    }
                }

                for(u16 i = 0; i < HIDDEN_COUNT_2; i++){
                    hidden_neuron_bias_2[i] += hidden_neuron_error_2[i] * learning_rate;
                }

                for(u16 i = 0; i < HIDDEN_COUNT_1; i++){
                    hidden_neuron_error_1[i] = 0;
                }

                for(u16 i = 0; i < HIDDEN_COUNT_1; i++){
                    for(u16 j = 0; j < HIDDEN_COUNT_2; j++){
                        hidden_neuron_error_1[i] +=  derivative_of_sigmoid_func(hidden_neuron_in_1[i]) * hidden_neuron_error_2[j] * w_hidden_to_hidden[i][j];
                    }
                }
                for(u16 i = 0; i < INPUT_COUNT; i++){
                    for(u16 j = 0; j < HIDDEN_COUNT_1; j++){
                        w_input_to_hidden[i][j] += hidden_neuron_error_1[j] * input[i][k + 26*inset] * learning_rate;
                    }
                }

                for(u16 i = 0; i < HIDDEN_COUNT_1; i++){
                    hidden_neuron_bias_1[i] +=hidden_neuron_error_1[i] * learning_rate;
                }
            }
            inset_error[inset] = _256_512_512_26_ann_calculate_total_error();
        }
        net_256_512_512_26.total_err = (inset_error[0] + inset_error[1] + inset_error[2])/3;
        epoch_no = era;
        epoch_status = (era*100)/epoch;
        if(stop_the_training == 1) break;
    }
}
void MainWindow::_256_512_512_26_random_initilize_handler(void){
    _256_512_512_26_letters_to_arrays();

    for(u8 k = 0; k < 26; k++){
        for(u16 i = 0; i < 16; i++){
            for(u16 j = 0; j < 16; j++){
                ann_class->net_256_512_512_26.input[16*i + j][k] = alphabet.letter[k][i][j];
            }
        }
    }
    for(u8 k = 26; k < (26*2); k++){
        for(u16 i = 0; i < 16; i++){
            for(u16 j = 0; j < 16; j++){
                ann_class->net_256_512_512_26.input[16*i + j][k] = alphabet_italic.letter[k - 26][i][j];
            }
        }
    }
    for(u8 k = (26*2); k < (26*3); k++){
        for(u16 i = 0; i < 16; i++){
            for(u16 j = 0; j < 16; j++){
                ann_class->net_256_512_512_26.input[16*i + j][k] = alphabet_bold.letter[k - (26*2)][i][j];
            }
        }
    }
    for(u8 k = (26*3); k < (26*4); k++){
        for(u16 i = 0; i < 16; i++){
            for(u16 j = 0; j < 16; j++){
                ann_class->net_256_512_512_26.input[16*i + j][k] = alphabet_bold_italic.letter[k - (26*3)][i][j];
            }
        }
    }

    for(u16 i = 0; i < 26; i++){
        for(u16 j = 0; j < 26; j++){
            ann_class->net_256_512_512_26.desired_output[i][j] = 0;
            if(i == j){
                ann_class->net_256_512_512_26.desired_output[i][j] = 1;
            }
        }
    }

    qsrand(QDateTime::currentMSecsSinceEpoch());

    for(u16 i = 0; i < HIDDEN_COUNT_1; i++){
        ann_class->net_256_512_512_26.hidden_neuron_bias_1[i] = ((double) qrand()/RAND_MAX) * (-2) + 1;
    }
    for(u16 i = 0; i < HIDDEN_COUNT_2; i++){
        ann_class->net_256_512_512_26.hidden_neuron_bias_2[i] = ((double) qrand()/RAND_MAX) * (-2) + 1;
    }
    for(u16 i = 0; i < OUTPUT_COUNT; i++){
        ann_class->net_256_512_512_26.output_bias[i] = ((double) qrand()/RAND_MAX) * (-2) + 1;
    }

    for(u16 i = 0; i < INPUT_COUNT; i++){
        for(u16 j = 0; j < HIDDEN_COUNT_1; j++){
            ann_class->net_256_512_512_26.w_input_to_hidden[i][j] = ((double) qrand()/RAND_MAX) * (-2) + 1;
        }
    }
    for(u16 i = 0; i < HIDDEN_COUNT_1; i++){
        for(u16 j = 0; j < HIDDEN_COUNT_2; j++){
            ann_class->net_256_512_512_26.w_hidden_to_hidden[i][j] = ((double) qrand()/RAND_MAX) * (-2) + 1;
        }
    }
    for(u16 i = 0; i < HIDDEN_COUNT_2; i++){
        for(u16 j = 0; j < OUTPUT_COUNT; j++){
            ann_class->net_256_512_512_26.w_hidden_to_output[i][j] = ((double) qrand()/RAND_MAX) * (-2) + 1;
        }
    }

    ui->label_256_512_512_26_random_initilize->setText("Initilized randomly");
}
void MainWindow::_256_512_512_26_train_handler(void){
    ann_class->train_status = 3;
    ann_class->stop_the_training = 0;
}
void MainWindow::_256_512_512_26_test_handler(void){
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_16x16/tester.png",alphabet.tester);

    for(u16 i = 0; i < 16; i++){
        for(u16 j = 0; j < 16; j++){
            ann_class->net_256_512_512_26.test_input[16*i + j] = alphabet.tester[i][j];
        }
    }

    ann_class->_256_512_512_26_ann_test(ann_class->net_256_512_512_26.test_input,
                                        ann_class->net_256_512_512_26.hidden_neuron_bias_1,
                                        ann_class->net_256_512_512_26.hidden_neuron_bias_2,
                                        ann_class->net_256_512_512_26.output_bias,
                                        ann_class->net_256_512_512_26.w_input_to_hidden,
                                        ann_class->net_256_512_512_26.w_hidden_to_hidden,
                                        ann_class->net_256_512_512_26.w_hidden_to_output);

}
void MainWindow::_256_512_512_26_show_weights_handler(void){
    for(u16 i = 0; i < HIDDEN_COUNT_1; i++){
        qDebug() << QString("hidden_bias_1[%1] : ").arg(i) << ann_class->net_256_512_512_26.hidden_neuron_bias_1[i];
    }
    for(u16 i = 0; i < HIDDEN_COUNT_2; i++){
        qDebug() << QString("hidden_bias_2[%1] : ").arg(i) << ann_class->net_256_512_512_26.hidden_neuron_bias_2[i];
    }
    for(u16 i = 0; i < OUTPUT_COUNT; i++){
        qDebug() << QString("output_bias[%1] : ").arg(i) << ann_class->net_256_512_512_26.output_bias[i];
    }

    for(u16 i = 0; i < INPUT_COUNT; i++){
        for(u16 j = 0; j < HIDDEN_COUNT_1; j++){
            qDebug() << QString("w_input_to_hidden[%1][%2] : ").arg(i).arg(j) << ann_class->net_256_512_512_26.w_input_to_hidden[i][j];
        }
    }
    for(u16 i = 0; i < HIDDEN_COUNT_1; i++){
        for(u16 j = 0; j < HIDDEN_COUNT_2; j++){
            qDebug() << QString("w_hidden_to_hidden[%1][%2] : ").arg(i).arg(j) << ann_class->net_256_512_512_26.w_hidden_to_hidden[i][j];
        }
    }
    for(u16 i = 0; i < HIDDEN_COUNT_2; i++){
        for(u16 j = 0; j < OUTPUT_COUNT; j++){
            qDebug() << QString("w_hidden_to_output[%1][%2] : ").arg(i).arg(j) << ann_class->net_256_512_512_26.w_hidden_to_output[i][j];
        }
    }
    ui->label_256_512_512_26_show_weights->setText("Showed..");
}
void MainWindow::_256_512_512_26_save_weights_handler(void){
    QSettings settings("weights_256_512_512_26.ini",QSettings::IniFormat);

    settings.beginGroup("w");

    for(u16 i = 0; i < HIDDEN_COUNT_1; i++){
        settings.setValue(QString("hb1-%1").arg(i),ann_class->net_256_512_512_26.hidden_neuron_bias_1[i]);
    }
    for(u16 i = 0; i < HIDDEN_COUNT_2; i++){
        settings.setValue(QString("hb2-%2").arg(i),ann_class->net_256_512_512_26.hidden_neuron_bias_2[i]);
    }
    for(u16 i = 0; i < OUTPUT_COUNT; i++){
        settings.setValue(QString("ob-%1").arg(i),ann_class->net_256_512_512_26.output_bias[i]);
    }

    for(u16 i = 0; i < INPUT_COUNT; i++){
        for(u16 j = 0; j < HIDDEN_COUNT_1; j++){
            settings.setValue(QString("i2h-%1-%2").arg(i).arg(j),ann_class->net_256_512_512_26.w_input_to_hidden[i][j]);
        }
    }
    for(u16 i = 0; i < HIDDEN_COUNT_1; i++){
        for(u16 j = 0; j < HIDDEN_COUNT_2; j++){
            settings.setValue(QString("h2h-%1-%2").arg(i).arg(j),ann_class->net_256_512_512_26.w_hidden_to_hidden[i][j]);
        }
    }
    for(u16 i = 0; i < HIDDEN_COUNT_2; i++){
        for(u16 j = 0; j < OUTPUT_COUNT; j++){
            settings.setValue(QString("h2o-%1-%2").arg(i).arg(j),ann_class->net_256_512_512_26.w_hidden_to_output[i][j]);
        }
    }
    settings.endGroup();
    settings.sync();
    QProcess::execute("sync");

    ui->label_256_512_512_26_save_weights->setText("Saved..");
}
void MainWindow::_256_512_512_26_load_saved_weights_handler(void){
    QSettings settings("weights_256_512_512_26.ini",QSettings::IniFormat);

    for(u16 i = 0; i < HIDDEN_COUNT_1; i++){
        ann_class->net_256_512_512_26.hidden_neuron_bias_1[i] = settings.value(QString("w/hb1-%1").arg(i)).toDouble();
    }
    for(u16 i = 0; i < HIDDEN_COUNT_2; i++){
        ann_class->net_256_512_512_26.hidden_neuron_bias_2[i] = settings.value(QString("w/hb2-%1").arg(i)).toDouble();
    }
    for(u16 i = 0; i < OUTPUT_COUNT; i++){
        ann_class->net_256_512_512_26.output_bias[i] = settings.value(QString("w/ob-%1").arg(i)).toDouble();
    }

    for(u16 i = 0; i < INPUT_COUNT; i++){
        for(u16 j = 0; j < HIDDEN_COUNT_1; j++){
            ann_class->net_256_512_512_26.w_input_to_hidden[i][j] = settings.value(QString("w/i2h-%1-%2").arg(i).arg(j)).toDouble();
        }
    }
    for(u16 i = 0; i < HIDDEN_COUNT_1; i++){
        for(u16 j = 0; j < HIDDEN_COUNT_2; j++){
            ann_class->net_256_512_512_26.w_hidden_to_hidden[i][j] = settings.value(QString("w/h2h-%1-%2").arg(i).arg(j)).toDouble();
        }
    }
    for(u16 i = 0; i < HIDDEN_COUNT_2; i++){
        for(u16 j = 0; j < OUTPUT_COUNT; j++){
            ann_class->net_256_512_512_26.w_hidden_to_output[i][j] = settings.value(QString("w/h2o-%1-%2").arg(i).arg(j)).toDouble();
        }
    }
    ui->label_256_512_512_26_load_saved_weights->setText("Loaded..");
}
void MainWindow::_256_512_512_26_stop_train_handler(void){
    ann_class->stop_the_training = 1;
}
void MainWindow::_256_512_512_26_letters_to_arrays(void){
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_16x16/A.png",alphabet.letter[0]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_16x16/B.png",alphabet.letter[1]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_16x16/C.png",alphabet.letter[2]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_16x16/D.png",alphabet.letter[3]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_16x16/E.png",alphabet.letter[4]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_16x16/F.png",alphabet.letter[5]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_16x16/G.png",alphabet.letter[6]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_16x16/H.png",alphabet.letter[7]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_16x16/I.png",alphabet.letter[8]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_16x16/J.png",alphabet.letter[9]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_16x16/K.png",alphabet.letter[10]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_16x16/L.png",alphabet.letter[11]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_16x16/M.png",alphabet.letter[12]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_16x16/N.png",alphabet.letter[13]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_16x16/O.png",alphabet.letter[14]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_16x16/P.png",alphabet.letter[15]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_16x16/Q.png",alphabet.letter[16]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_16x16/R.png",alphabet.letter[17]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_16x16/S.png",alphabet.letter[18]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_16x16/T.png",alphabet.letter[19]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_16x16/U.png",alphabet.letter[20]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_16x16/V.png",alphabet.letter[21]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_16x16/W.png",alphabet.letter[22]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_16x16/X.png",alphabet.letter[23]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_16x16/Y.png",alphabet.letter[24]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_16x16/Z.png",alphabet.letter[25]);

    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_italic_16x16/A.png",alphabet_italic.letter[0]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_italic_16x16/B.png",alphabet_italic.letter[1]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_italic_16x16/C.png",alphabet_italic.letter[2]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_italic_16x16/D.png",alphabet_italic.letter[3]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_italic_16x16/E.png",alphabet_italic.letter[4]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_italic_16x16/F.png",alphabet_italic.letter[5]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_italic_16x16/G.png",alphabet_italic.letter[6]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_italic_16x16/H.png",alphabet_italic.letter[7]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_italic_16x16/I.png",alphabet_italic.letter[8]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_italic_16x16/J.png",alphabet_italic.letter[9]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_italic_16x16/K.png",alphabet_italic.letter[10]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_italic_16x16/L.png",alphabet_italic.letter[11]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_italic_16x16/M.png",alphabet_italic.letter[12]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_italic_16x16/N.png",alphabet_italic.letter[13]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_italic_16x16/O.png",alphabet_italic.letter[14]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_italic_16x16/P.png",alphabet_italic.letter[15]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_italic_16x16/Q.png",alphabet_italic.letter[16]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_italic_16x16/R.png",alphabet_italic.letter[17]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_italic_16x16/S.png",alphabet_italic.letter[18]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_italic_16x16/T.png",alphabet_italic.letter[19]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_italic_16x16/U.png",alphabet_italic.letter[20]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_italic_16x16/V.png",alphabet_italic.letter[21]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_italic_16x16/W.png",alphabet_italic.letter[22]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_italic_16x16/X.png",alphabet_italic.letter[23]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_italic_16x16/Y.png",alphabet_italic.letter[24]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_italic_16x16/Z.png",alphabet_italic.letter[25]);

    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_bold_16x16/A.png",alphabet_bold.letter[0]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_bold_16x16/B.png",alphabet_bold.letter[1]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_bold_16x16/C.png",alphabet_bold.letter[2]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_bold_16x16/D.png",alphabet_bold.letter[3]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_bold_16x16/E.png",alphabet_bold.letter[4]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_bold_16x16/F.png",alphabet_bold.letter[5]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_bold_16x16/G.png",alphabet_bold.letter[6]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_bold_16x16/H.png",alphabet_bold.letter[7]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_bold_16x16/I.png",alphabet_bold.letter[8]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_bold_16x16/J.png",alphabet_bold.letter[9]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_bold_16x16/K.png",alphabet_bold.letter[10]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_bold_16x16/L.png",alphabet_bold.letter[11]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_bold_16x16/M.png",alphabet_bold.letter[12]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_bold_16x16/N.png",alphabet_bold.letter[13]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_bold_16x16/O.png",alphabet_bold.letter[14]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_bold_16x16/P.png",alphabet_bold.letter[15]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_bold_16x16/Q.png",alphabet_bold.letter[16]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_bold_16x16/R.png",alphabet_bold.letter[17]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_bold_16x16/S.png",alphabet_bold.letter[18]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_bold_16x16/T.png",alphabet_bold.letter[19]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_bold_16x16/U.png",alphabet_bold.letter[20]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_bold_16x16/V.png",alphabet_bold.letter[21]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_bold_16x16/W.png",alphabet_bold.letter[22]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_bold_16x16/X.png",alphabet_bold.letter[23]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_bold_16x16/Y.png",alphabet_bold.letter[24]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_bold_16x16/Z.png",alphabet_bold.letter[25]);

    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_bold_italic_16x16/A.png",alphabet_bold_italic.letter[0]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_bold_italic_16x16/B.png",alphabet_bold_italic.letter[1]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_bold_italic_16x16/C.png",alphabet_bold_italic.letter[2]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_bold_italic_16x16/D.png",alphabet_bold_italic.letter[3]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_bold_italic_16x16/E.png",alphabet_bold_italic.letter[4]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_bold_italic_16x16/F.png",alphabet_bold_italic.letter[5]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_bold_italic_16x16/G.png",alphabet_bold_italic.letter[6]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_bold_italic_16x16/H.png",alphabet_bold_italic.letter[7]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_bold_italic_16x16/I.png",alphabet_bold_italic.letter[8]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_bold_italic_16x16/J.png",alphabet_bold_italic.letter[9]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_bold_italic_16x16/K.png",alphabet_bold_italic.letter[10]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_bold_italic_16x16/L.png",alphabet_bold_italic.letter[11]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_bold_italic_16x16/M.png",alphabet_bold_italic.letter[12]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_bold_italic_16x16/N.png",alphabet_bold_italic.letter[13]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_bold_italic_16x16/O.png",alphabet_bold_italic.letter[14]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_bold_italic_16x16/P.png",alphabet_bold_italic.letter[15]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_bold_italic_16x16/Q.png",alphabet_bold_italic.letter[16]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_bold_italic_16x16/R.png",alphabet_bold_italic.letter[17]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_bold_italic_16x16/S.png",alphabet_bold_italic.letter[18]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_bold_italic_16x16/T.png",alphabet_bold_italic.letter[19]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_bold_italic_16x16/U.png",alphabet_bold_italic.letter[20]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_bold_italic_16x16/V.png",alphabet_bold_italic.letter[21]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_bold_italic_16x16/W.png",alphabet_bold_italic.letter[22]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_bold_italic_16x16/X.png",alphabet_bold_italic.letter[23]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_bold_italic_16x16/Y.png",alphabet_bold_italic.letter[24]);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_bold_italic_16x16/Z.png",alphabet_bold_italic.letter[25]);
}
