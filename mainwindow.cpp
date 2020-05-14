#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    advanced_64_128_5_tryout();
}
void MainWindow::advanced_64_128_5_ann_train(double input[64][5], double desired_output[5][5], double calculated_output[5][5],
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
        qDebug() << "training status % " << (era*100)/epoch;
    }
    qDebug() << "training is FINISHED!!";

}
void MainWindow::advanced_64_128_5_ann_test(double input[64],
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

    qDebug() << "Yuzde " << 100*output_neuron_out[0] << "\t" << "ihtimalle sifir isareti";
    qDebug() << "Yuzde " << 100*output_neuron_out[1] << "\t" << "ihtimalle toplama isareti";
    qDebug() << "Yuzde " << 100*output_neuron_out[2] << "\t" << "ihtimalle bolme isareti";
    qDebug() << "Yuzde " << 100*output_neuron_out[3] << "\t" << "ihtimalle cikarma isareti";
    qDebug() << "Yuzde " << 100*output_neuron_out[4] << "\t" << "ihtimalle carpma isareti";

}

void MainWindow::advanced_64_128_5_tryout(void){
    image_to_array("/home/ahmet/Desktop/QT-Projects/ANN/zero.png",zero_image);
    image_to_array("/home/ahmet/Desktop/QT-Projects/ANN/add.png",addition_image);
    image_to_array("/home/ahmet/Desktop/QT-Projects/ANN/divide.png",divide_image);
    image_to_array("/home/ahmet/Desktop/QT-Projects/ANN/minus.png",minus_image);
    image_to_array("/home/ahmet/Desktop/QT-Projects/ANN/multiply.png",multiply_image);

    for(u8 i = 0; i < 8; i++){
        for(u8 j = 0; j < 8; j++){
            net_64_128_5.input[8*i + j][0] = zero_image[i][j];
        }
    }
    for(u8 i = 0; i < 8; i++){
        for(u8 j = 0; j < 8; j++){
            net_64_128_5.input[8*i + j][1] = addition_image[i][j];
        }
    }
    for(u8 i = 0; i < 8; i++){
        for(u8 j = 0; j < 8; j++){
            net_64_128_5.input[8*i + j][2] = divide_image[i][j];
        }
    }
    for(u8 i = 0; i < 8; i++){
        for(u8 j = 0; j < 8; j++){
            net_64_128_5.input[8*i + j][3] = minus_image[i][j];
        }
    }
    for(u8 i = 0; i < 8; i++){
        for(u8 j = 0; j < 8; j++){
            net_64_128_5.input[8*i + j][4] = multiply_image[i][j];
        }
    }

    for(u8 i = 0; i < 5; i++){
        for(u8 j = 0; j < 5; j++){
            net_64_128_5.desired_output[i][j] = 0;
        }
    }
    net_64_128_5.desired_output[0][0] = 1;
    net_64_128_5.desired_output[1][1] = 1;
    net_64_128_5.desired_output[2][2] = 1;
    net_64_128_5.desired_output[3][3] = 1;
    net_64_128_5.desired_output[4][4] = 1;

    for(u8 i = 0; i < 128; i++){
        net_64_128_5.hidden_bias[i] = 0.1 + 0.01*i;
    }
    for(u8 i = 0; i < 5; i++){
        net_64_128_5.output_bias[i] = 0.2;
    }

    for(u8 i = 0; i < 64; i++){
        for(u8 j = 0; j < 128; j++){
            net_64_128_5.w_input_to_hidden[i][j] = 0.1;
        }
    }
    for(u8 i = 0; i < 128; i++){
        for(u8 j = 0; j < 5; j++){
            net_64_128_5.w_hidden_to_output[i][j] = 0.1;
        }
    }

    for(u8 i = 0; i < 64; i++){
        for(u8 j = 0; j < 5; j++){
            qDebug() << QString(" input[%1][%2] : ").arg(i).arg(j) << net_64_128_5.input[i][j];
        }
    }

    advanced_64_128_5_ann_train(net_64_128_5.input, net_64_128_5.desired_output, net_64_128_5.calculated_output,
                             net_64_128_5.hidden_bias,net_64_128_5.output_bias,
                             net_64_128_5.w_input_to_hidden,net_64_128_5.w_hidden_to_output,
                             100000, 0.000132);

    for(u8 i = 0; i < 5; i++){
        for(u8 j = 0; j < 5; j++){
            qDebug() << QString("desired output[%1][%2] : ").arg(i).arg(j) << net_64_128_5.desired_output[i][j] <<
                        QString("calculated output[%1][%2] : ").arg(i).arg(j) << net_64_128_5.calculated_output[i][j];
        }
    }

    image_to_array("/home/ahmet/Desktop/QT-Projects/ANN/tester.png",test_image);

    for(u8 i = 0; i < 8; i++){
        for(u8 j = 0; j < 8; j++){
            net_64_128_5.test_input[8*i + j] = test_image[i][j];
        }
    }

    advanced_64_128_5_ann_test(net_64_128_5.test_input,
                             net_64_128_5.hidden_bias,net_64_128_5.output_bias,
                             net_64_128_5.w_input_to_hidden,net_64_128_5.w_hidden_to_output);

}


void MainWindow::image_to_array(QString location,double image_array[8][8]){
    QImage read_image;

    read_image.load(location);

    for(u8 i = 0; i < read_image.height();i++){
        for(u8 j = 0; j < read_image.width();j++){
            image_array[i][j] = 0;
            if((read_image.pixel(i,j) & 0xFF) == 0xFF){
                image_array[i][j] = 1;
            }
            //qDebug() << QString("array[%1][%2] :").arg(i).arg(j) << QString("%1").arg(image_array[i][j],0,16);
        }
    }
}


MainWindow::~MainWindow()
{
    delete ui;
}
