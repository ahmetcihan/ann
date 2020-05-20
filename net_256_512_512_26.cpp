#include "mainwindow.h"
#include "ui_mainwindow.h"

#define INPUT_COUNT     256
#define HIDDEN_COUNT_1  512
#define HIDDEN_COUNT_2  512
#define OUTPUT_COUNT    26
#define IO_ARRAY_LENGTH 26

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

    mainwindow->ui->label_64_128_32_5_test->setText(str);


}

void ann::_256_512_512_26_ann_train(double input[256][26], double desired_output[26][26], double calculated_output[26][26],
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

    for(u32 era = 0; era < epoch; era++){
        for(u16 k = 0; k < IO_ARRAY_LENGTH; k++){

            for(u16 i = 0; i < HIDDEN_COUNT_1; i++){
                hidden_neuron_in_1[i] = hidden_neuron_bias_1[i];
            }
            for(u16 i = 0; i < HIDDEN_COUNT_1; i++){
                for(u16 j = 0; j < INPUT_COUNT; j++){
                    hidden_neuron_in_1[i] += input[j][k]*w_input_to_hidden[j][i];
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
                    w_input_to_hidden[i][j] += hidden_neuron_error_1[j] * input[i][k] * learning_rate;
                }
            }

            for(u16 i = 0; i < HIDDEN_COUNT_1; i++){
                hidden_neuron_bias_1[i] +=hidden_neuron_error_1[i] * learning_rate;
            }
        }
        epoch_status = (era*100)/epoch;
    }
}
void MainWindow::_256_512_512_26_random_initilize_handler(void){
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_16x16/A.png",alphabet.A);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_16x16/B.png",alphabet.B);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_16x16/C.png",alphabet.C);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_16x16/D.png",alphabet.D);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_16x16/E.png",alphabet.E);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_16x16/F.png",alphabet.F);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_16x16/G.png",alphabet.G);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_16x16/H.png",alphabet.H);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_16x16/I.png",alphabet.I);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_16x16/J.png",alphabet.J);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_16x16/K.png",alphabet.K);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_16x16/L.png",alphabet.L);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_16x16/M.png",alphabet.M);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_16x16/N.png",alphabet.N);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_16x16/O.png",alphabet.O);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_16x16/P.png",alphabet.P);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_16x16/Q.png",alphabet.Q);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_16x16/R.png",alphabet.R);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_16x16/S.png",alphabet.S);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_16x16/T.png",alphabet.T);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_16x16/U.png",alphabet.U);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_16x16/V.png",alphabet.V);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_16x16/W.png",alphabet.W);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_16x16/X.png",alphabet.X);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_16x16/Y.png",alphabet.Y);
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_16x16/Z.png",alphabet.Z);

    for(u16 i = 0; i < 8; i++){
        for(u16 j = 0; j < 8; j++){
            ann_class->net_64_128_32_5.input[8*i + j][0] = zero_image[i][j];
        }
    }
    for(u16 i = 0; i < 8; i++){
        for(u16 j = 0; j < 8; j++){
            ann_class->net_64_128_32_5.input[8*i + j][1] = addition_image[i][j];
        }
    }
    for(u16 i = 0; i < 8; i++){
        for(u16 j = 0; j < 8; j++){
            ann_class->net_64_128_32_5.input[8*i + j][2] = divide_image[i][j];
        }
    }
    for(u16 i = 0; i < 8; i++){
        for(u16 j = 0; j < 8; j++){
            ann_class->net_64_128_32_5.input[8*i + j][3] = minus_image[i][j];
        }
    }
    for(u16 i = 0; i < 8; i++){
        for(u16 j = 0; j < 8; j++){
            ann_class->net_64_128_32_5.input[8*i + j][4] = multiply_image[i][j];
        }
    }

    for(u16 i = 0; i < 5; i++){
        for(u16 j = 0; j < 5; j++){
            ann_class->net_64_128_32_5.desired_output[i][j] = 0;
        }
    }
    ann_class->net_64_128_32_5.desired_output[0][0] = 1;
    ann_class->net_64_128_32_5.desired_output[1][1] = 1;
    ann_class->net_64_128_32_5.desired_output[2][2] = 1;
    ann_class->net_64_128_32_5.desired_output[3][3] = 1;
    ann_class->net_64_128_32_5.desired_output[4][4] = 1;

    for(u16 i = 0; i < HIDDEN_COUNT_1; i++){
        ann_class->net_64_128_32_5.hidden_neuron_bias_1[i] = 0.1 + 0.01*i;
    }
    for(u16 i = 0; i < HIDDEN_COUNT_2; i++){
        ann_class->net_64_128_32_5.hidden_neuron_bias_2[i] = 0.1 + 0.01*i;
    }
    for(u16 i = 0; i < OUTPUT_COUNT; i++){
        ann_class->net_64_128_32_5.output_bias[i] = 0.2;
    }

    for(u16 i = 0; i < INPUT_COUNT; i++){
        for(u16 j = 0; j < HIDDEN_COUNT_1; j++){
            ann_class->net_64_128_32_5.w_input_to_hidden[i][j] = 0.1;
        }
    }
    for(u16 i = 0; i < HIDDEN_COUNT_1; i++){
        for(u16 j = 0; j < HIDDEN_COUNT_2; j++){
            ann_class->net_64_128_32_5.w_hidden_to_hidden[i][j] = 0.1;
        }
    }
    for(u16 i = 0; i < HIDDEN_COUNT_2; i++){
        for(u16 j = 0; j < OUTPUT_COUNT; j++){
            ann_class->net_64_128_32_5.w_hidden_to_output[i][j] = 0.1;
        }
    }

    ui->label_64_128_32_5_random_initilize->setText("Initilized randomly");
}
void MainWindow::_256_512_512_26_train_handler(void){
    ann_class->train_status = 2;
}
void MainWindow::_256_512_512_26_test_handler(void){
    image_to_array_16x16("/home/ahmet/Desktop/arial_fonts_16x16/tester.png",alphabet.tester);

    for(u16 i = 0; i < 8; i++){
        for(u16 j = 0; j < 8; j++){
            ann_class->net_64_128_32_5.test_input[8*i + j] = test_image[i][j];
        }
    }

    ann_class->_64_128_32_5_ann_test(ann_class->net_64_128_32_5.test_input,
                ann_class->net_64_128_32_5.hidden_neuron_bias_1,ann_class->net_64_128_32_5.hidden_neuron_bias_2,ann_class->net_64_128_32_5.output_bias,
                ann_class->net_64_128_32_5.w_input_to_hidden,ann_class->net_64_128_32_5.w_hidden_to_hidden,ann_class->net_64_128_32_5.w_hidden_to_output);

}
void MainWindow::_256_512_512_26_show_weights_handler(void){
    for(u16 i = 0; i < HIDDEN_COUNT_1; i++){
        qDebug() << QString("hidden_bias_1[%1] : ").arg(i) << ann_class->net_64_128_32_5.hidden_neuron_bias_1[i];
    }
    for(u16 i = 0; i < HIDDEN_COUNT_2; i++){
        qDebug() << QString("hidden_bias_2[%1] : ").arg(i) << ann_class->net_64_128_32_5.hidden_neuron_bias_2[i];
    }
    for(u16 i = 0; i < OUTPUT_COUNT; i++){
        qDebug() << QString("output_bias[%1] : ").arg(i) << ann_class->net_64_128_32_5.output_bias[i];
    }

    for(u16 i = 0; i < INPUT_COUNT; i++){
        for(u16 j = 0; j < HIDDEN_COUNT_1; j++){
            qDebug() << QString("w_input_to_hidden[%1][%2] : ").arg(i).arg(j) << ann_class->net_64_128_32_5.w_input_to_hidden[i][j];
        }
    }
    for(u16 i = 0; i < HIDDEN_COUNT_1; i++){
        for(u16 j = 0; j < HIDDEN_COUNT_2; j++){
            qDebug() << QString("w_hidden_to_hidden[%1][%2] : ").arg(i).arg(j) << ann_class->net_64_128_32_5.w_hidden_to_hidden[i][j];
        }
    }
    for(u16 i = 0; i < HIDDEN_COUNT_2; i++){
        for(u16 j = 0; j < OUTPUT_COUNT; j++){
            qDebug() << QString("w_hidden_to_output[%1][%2] : ").arg(i).arg(j) << ann_class->net_64_128_32_5.w_hidden_to_output[i][j];
        }
    }
    ui->label_64_128_32_5_show_weights->setText("Showed..");
}
void MainWindow::_256_512_512_26_save_weights_handler(void){
    QSettings settings("weights_64_128_32_5.ini",QSettings::IniFormat);

    settings.beginGroup("w");

    for(u16 i = 0; i < HIDDEN_COUNT_1; i++){
        settings.setValue(QString("hb1-%1").arg(i),ann_class->net_64_128_32_5.hidden_neuron_bias_1[i]);
    }
    for(u16 i = 0; i < HIDDEN_COUNT_2; i++){
        settings.setValue(QString("hb1-%2").arg(i),ann_class->net_64_128_32_5.hidden_neuron_bias_2[i]);
    }
    for(u16 i = 0; i < OUTPUT_COUNT; i++){
        settings.setValue(QString("ob-%1").arg(i),ann_class->net_64_128_32_5.output_bias[i]);
    }

    for(u16 i = 0; i < INPUT_COUNT; i++){
        for(u16 j = 0; j < HIDDEN_COUNT_1; j++){
            settings.setValue(QString("i2h-%1-%2").arg(i).arg(j),ann_class->net_64_128_32_5.w_input_to_hidden[i][j]);
        }
    }
    for(u16 i = 0; i < HIDDEN_COUNT_1; i++){
        for(u16 j = 0; j < HIDDEN_COUNT_2; j++){
            settings.setValue(QString("h2h-%1-%2").arg(i).arg(j),ann_class->net_64_128_32_5.w_hidden_to_hidden[i][j]);
        }
    }
    for(u16 i = 0; i < HIDDEN_COUNT_2; i++){
        for(u16 j = 0; j < OUTPUT_COUNT; j++){
            settings.setValue(QString("h2o-%1-%2").arg(i).arg(j),ann_class->net_64_128_32_5.w_hidden_to_output[i][j]);
        }
    }
    settings.endGroup();
    settings.sync();
    QProcess::execute("sync");

    ui->label_64_128_32_5_save_weights->setText("Saved..");
}
void MainWindow::_256_512_512_26_load_saved_weights_handler(void){
    QSettings settings("weights_64_128_32_5.ini",QSettings::IniFormat);

    for(u16 i = 0; i < HIDDEN_COUNT_1; i++){
        ann_class->net_64_128_32_5.hidden_neuron_bias_1[i] = settings.value(QString("w/hb1-%1").arg(i)).toDouble();
    }
    for(u16 i = 0; i < HIDDEN_COUNT_2; i++){
        ann_class->net_64_128_32_5.hidden_neuron_bias_2[i] = settings.value(QString("w/hb2-%1").arg(i)).toDouble();
    }
    for(u16 i = 0; i < OUTPUT_COUNT; i++){
        ann_class->net_64_128_32_5.output_bias[i] = settings.value(QString("w/ob-%1").arg(i)).toDouble();
    }

    for(u16 i = 0; i < INPUT_COUNT; i++){
        for(u16 j = 0; j < HIDDEN_COUNT_1; j++){
            ann_class->net_64_128_32_5.w_input_to_hidden[i][j] = settings.value(QString("w/i2h-%1-%2").arg(i).arg(j)).toDouble();
        }
    }
    for(u16 i = 0; i < HIDDEN_COUNT_1; i++){
        for(u16 j = 0; j < HIDDEN_COUNT_2; j++){
            ann_class->net_64_128_32_5.w_hidden_to_hidden[i][j] = settings.value(QString("w/h2h-%1-%2").arg(i).arg(j)).toDouble();
        }
    }
    for(u16 i = 0; i < HIDDEN_COUNT_2; i++){
        for(u16 j = 0; j < OUTPUT_COUNT; j++){
            ann_class->net_64_128_32_5.w_hidden_to_output[i][j] = settings.value(QString("w/h2o-%1-%2").arg(i).arg(j)).toDouble();
        }
    }
    ui->label_64_128_32_5_load_saved_weights->setText("Loaded..");
}
