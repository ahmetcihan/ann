#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    //_2_4_2_tryout();
    image_to_array("/home/ahmet/Desktop/QT-Projects/ANN/add.png",addition_image);
    image_to_array("/home/ahmet/Desktop/QT-Projects/ANN/divide.png",divide_image);
    image_to_array("/home/ahmet/Desktop/QT-Projects/ANN/minus.png",minus_image);
    image_to_array("/home/ahmet/Desktop/QT-Projects/ANN/multiply.png",multiply_image);
    image_to_array("/home/ahmet/Desktop/QT-Projects/ANN/zero.png",zero_image);

    for(u8 i = 0; i < 8; i++){
        for(u8 j = 0; j < 8; j++){
            input_64_128_5[8*i + j][0] = addition_image[i][j];
        }
    }
    for(u8 i = 0; i < 8; i++){
        for(u8 j = 0; j < 8; j++){
            input_64_128_5[8*i + j][1] = divide_image[i][j];
        }
    }
    for(u8 i = 0; i < 8; i++){
        for(u8 j = 0; j < 8; j++){
            input_64_128_5[8*i + j][2] = minus_image[i][j];
        }
    }
    for(u8 i = 0; i < 8; i++){
        for(u8 j = 0; j < 8; j++){
            input_64_128_5[8*i + j][3] = multiply_image[i][j];
        }
    }
    for(u8 i = 0; i < 8; i++){
        for(u8 j = 0; j < 8; j++){
            input_64_128_5[8*i + j][4] = zero_image[i][j];
        }
    }


    for(u8 i = 0; i < 5; i++){
        for(u8 j = 0; j < 5; j++){
            desired_output_64_128_5[i][j] = 0;
        }
    }
    desired_output_64_128_5[0][0] = 1;
    desired_output_64_128_5[1][1] = 1;
    desired_output_64_128_5[2][2] = 1;
    desired_output_64_128_5[3][3] = 1;
    desired_output_64_128_5[4][4] = 1;

    for(u8 i = 0; i < 64; i++){
        for(u8 j = 0; j < 128; j++){
            _64_128_5_input_to_hidden_weight[i][j] = 0.1;
        }
    }
    for(u8 i = 0; i < 128; i++){
        for(u8 j = 0; j < 5; j++){
            _64_128_5_hidden_to_output_weight[i][j] = 0.1;
        }
    }

    _64_128_5_ann_train(input_64_128_5,desired_output_64_128_5,10000,_64_128_5_input_to_hidden_weight,_64_128_5_hidden_to_output_weight);
}
void MainWindow::image_to_array(QString location,u8 image_array[8][8]){
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

void MainWindow::_64_128_5_ann_test(double *input, double input_to_hidden_weight[64][128], double hidden_to_output_weight[128][5]){
    const u32 input_count = 64;
    const u32 hidden_count = 128;
    const u32 output_count = 5;

    double Y_in[output_count];
    double Y_out[output_count];
    double hidden_in[hidden_count];
    double hidden_out[hidden_count];

    for(u8 i = 0; i < hidden_count; i++){
        hidden_in[i] = 0;
        for(u8 j = 0; j < input_count; j++){
            hidden_in[i] += input[j]*input_to_hidden_weight[j][i];
        }
        hidden_out[i] = sigmoid_func(hidden_in[i]);
    }

    for(u8 i = 0; i < output_count; i++){
        Y_in[i] = 0;
        for(u8 j = 0; j < hidden_count; j++){
            Y_in[i] += hidden_out[j]*hidden_to_output_weight[j][i];
        }
        Y_out[i] = sigmoid_func(Y_in[i]);
    }

    qDebug() << "************test result*********************";
    for(u8 i = 0; i < output_count; i++){
        qDebug() << QString("output[%1] : ").arg(i) << Y_out[i];
    }
}
void MainWindow::_64_128_5_ann_train(double input[64][5], double desired_output[5][5], u32 epoch, double input_to_hidden_weight[64][128], double hidden_to_output_weight[128][5]){
#define INPUT_COUNT 64
#define HIDDEN_COUNT 128
#define OUTPUT_COUNT 5
#define IO_ARRAY_LENGTH 2

    double calculated_output[OUTPUT_COUNT][IO_ARRAY_LENGTH];
    double Y_in[OUTPUT_COUNT];
    double Y_out[OUTPUT_COUNT];
    double delta_Y[OUTPUT_COUNT];
    double error[OUTPUT_COUNT];

    double hidden_in[HIDDEN_COUNT];
    double hidden_out[HIDDEN_COUNT];
    double delta_hidden[HIDDEN_COUNT];

    double delta_input_to_hidden_weight[64][128];
    double delta_hidden_to_output_weight[128][5];

    for(u8 j = 0; j < IO_ARRAY_LENGTH; j++){
        for(u8 i = 0; i < INPUT_COUNT; i++){
            qDebug() << QString("input[%1][%2] : ").arg(i).arg(j) << input[i][j];
        }
    }

    for(u32 era = 0; era < epoch; era++){
        for(u8 io_no = 0; io_no < IO_ARRAY_LENGTH; io_no++){
            for(u8 i = 0; i < HIDDEN_COUNT; i++){
                hidden_in[i] = 0;
                for(u8 j = 0; j < INPUT_COUNT; j++){
                    hidden_in[i] += input[j][io_no]*input_to_hidden_weight[j][i];
                }
                hidden_out[i] = sigmoid_func(hidden_in[i]);
            }

            for(u8 i = 0; i < OUTPUT_COUNT; i++){
                Y_in[i] = 0;
                for(u8 j = 0; j < HIDDEN_COUNT; j++){
                    Y_in[i] += hidden_out[j]*hidden_to_output_weight[j][i];
                }
                Y_out[i] = sigmoid_func(Y_in[i]);
                error[i] = desired_output[i][io_no] - Y_out[i];
                calculated_output[i][io_no] = Y_out[i];
                delta_Y[i] = derivative_of_sigmoid_func(Y_in[i]) * error[i];
            }

            for(u8 i = 0; i < HIDDEN_COUNT; i++){
                delta_hidden[i] = 0;
                for(u8 j = 0; j < OUTPUT_COUNT; j++){
                    delta_hidden[i] +=   (delta_Y[j]/hidden_to_output_weight[i][j] * derivative_of_sigmoid_func(hidden_in[i]));
                }
            }

            for(u8 i = 0; i < HIDDEN_COUNT; i++){
                if(hidden_out[i] == 0){
                    for(u8 j = 0; j < OUTPUT_COUNT; j++){
                        delta_hidden_to_output_weight[i][j] = 0;
                    }
                }
                else{
                    for(u8 j = 0; j < OUTPUT_COUNT; j++){
                        delta_hidden_to_output_weight[i][j] = delta_Y[j] / hidden_out[i];
                    }
                }
            }

            for(u8 i = 0; i < INPUT_COUNT; i++){
                if (input[i][io_no] == 0){
                    for(u8 j = 0; j < HIDDEN_COUNT; j++){
                        delta_input_to_hidden_weight[i][j] = 0;
                    }
                }
                else{
                    for(u8 j = 0; j < HIDDEN_COUNT; j++){
                        delta_input_to_hidden_weight[i][j] = delta_hidden[j]/input[i][io_no];
                    }
                }
            }

            for(u8 i = 0; i < INPUT_COUNT; i++){
                for(u8 j = 0; j < HIDDEN_COUNT; j++){
                    input_to_hidden_weight[i][j] = input_to_hidden_weight[i][j] + delta_input_to_hidden_weight[i][j];
                }
            }
            for(u8 i = 0; i < HIDDEN_COUNT; i++){
                for(u8 j = 0; j < OUTPUT_COUNT; j++){
                    hidden_to_output_weight[i][j] = hidden_to_output_weight[i][j] + delta_hidden_to_output_weight[i][j];
                }
            }
        }
    }
    for(u8 j = 0; j < IO_ARRAY_LENGTH; j++){
        for(u8 i = 0; i < OUTPUT_COUNT; i++){
            qDebug() << QString("desired_output[%1][%2] : ").arg(i).arg(j) << desired_output[i][j]
                        << QString("output[%1][%2] : ").arg(i).arg(j) << calculated_output[i][j];
        }
    }

}

void MainWindow::_2_4_2_tryout(void){
    input_2_4_2[0][0] = 3;
    input_2_4_2[1][0] = 6;
    desired_output_2_4_2[0][0] = 0.3;
    desired_output_2_4_2[1][0] = 0.6;

    input_2_4_2[0][1] = 3;
    input_2_4_2[1][1] = 6;
    desired_output_2_4_2[0][1] = 0.3;
    desired_output_2_4_2[1][1] = 0.6;

    for(u8 i = 0; i < 2; i++){
        for(u8 j = 0; j < 4; j++){
            _2_4_2_input_to_hidden_weight[i][j] = 0.1;
        }
    }
    for(u8 i = 0; i < 4; i++){
        for(u8 j = 0; j < 2; j++){
            _2_4_2_hidden_to_output_weight[i][j] = 0.1;
        }
    }

    //_2_4_2_ann_train(input_2_4_2,desired_output_2_4_2,100000, _2_4_2_input_to_hidden_weight, _2_4_2_hidden_to_output_weight);

    //double denemeci[2] = {3,6};
    //_2_4_2_ann_test(denemeci, _2_4_2_input_to_hidden_weight, _2_4_2_hidden_to_output_weight);
}
/*
void MainWindow::_2_4_2_ann_test(double *input, double input_to_hidden_weight[2][4], double hidden_to_output_weight[4][2]){
#define INPUT_COUNT 2
#define HIDDEN_COUNT 4
#define OUTPUT_COUNT 2

    double Y_in[OUTPUT_COUNT];
    double Y_out[OUTPUT_COUNT];
    double hidden_in[HIDDEN_COUNT];
    double hidden_out[HIDDEN_COUNT];

    for(u8 i = 0; i < HIDDEN_COUNT; i++){
        hidden_in[i] = 0;
        for(u8 j = 0; j < INPUT_COUNT; j++){
            hidden_in[i] += input[j]*input_to_hidden_weight[j][i];
        }
        hidden_out[i] = sigmoid_func(hidden_in[i]);
    }

    for(u8 i = 0; i < OUTPUT_COUNT; i++){
        Y_in[i] = 0;
        for(u8 j = 0; j < HIDDEN_COUNT; j++){
            Y_in[i] += hidden_out[j]*hidden_to_output_weight[j][i];
        }
        Y_out[i] = sigmoid_func(Y_in[i]);
    }

    qDebug() << "************test result*********************";
    for(u8 i = 0; i < OUTPUT_COUNT; i++){
        qDebug() << QString("output[%1] : ").arg(i) << Y_out[i];
    }
}

void MainWindow::_2_4_2_ann_train(double input[2][2], double desired_output[2][2], u32 epoch, double input_to_hidden_weight[2][4], double hidden_to_output_weight[4][2]){
#define INPUT_COUNT 2
#define HIDDEN_COUNT 4
#define OUTPUT_COUNT 2
#define IO_ARRAY_LENGTH 2

    double calculated_output[OUTPUT_COUNT][IO_ARRAY_LENGTH];
    double Y_in[OUTPUT_COUNT];
    double Y_out[OUTPUT_COUNT];
    double delta_Y[OUTPUT_COUNT];
    double error[OUTPUT_COUNT];

    double hidden_in[HIDDEN_COUNT];
    double hidden_out[HIDDEN_COUNT];
    double delta_hidden[HIDDEN_COUNT];

    double delta_input_to_hidden_weight[2][4];
    double delta_hidden_to_output_weight[4][2];

    for(u8 j = 0; j < IO_ARRAY_LENGTH; j++){
        for(u8 i = 0; i < INPUT_COUNT; i++){
            qDebug() << QString("input[%1][%2] : ").arg(i).arg(j) << input[i][j];
        }
    }
    for(u8 j = 0; j < IO_ARRAY_LENGTH; j++){
        for(u8 i = 0; i < OUTPUT_COUNT; i++){
            qDebug() << QString("desired_output[%1][%2] : ").arg(i).arg(j) << desired_output[i][j];
        }
    }

    for(u32 era = 0; era < epoch; era++){
        for(u8 io_no = 0; io_no < IO_ARRAY_LENGTH; io_no++){
            for(u8 i = 0; i < HIDDEN_COUNT; i++){
                hidden_in[i] = 0;
                for(u8 j = 0; j < INPUT_COUNT; j++){
                    hidden_in[i] += input[j][io_no]*input_to_hidden_weight[j][i];
                }
                hidden_out[i] = sigmoid_func(hidden_in[i]);
            }

            for(u8 i = 0; i < OUTPUT_COUNT; i++){
                Y_in[i] = 0;
                for(u8 j = 0; j < HIDDEN_COUNT; j++){
                    Y_in[i] += hidden_out[j]*hidden_to_output_weight[j][i];
                }
                Y_out[i] = sigmoid_func(Y_in[i]);
                error[i] = desired_output[i][io_no] - Y_out[i];
                calculated_output[i][io_no] = Y_out[i];
                delta_Y[i] = derivative_of_sigmoid_func(Y_in[i]) * error[i];
            }

            for(u8 i = 0; i < HIDDEN_COUNT; i++){
                delta_hidden[i] = 0;
                for(u8 j = 0; j < OUTPUT_COUNT; j++){
                    delta_hidden[i] +=   (delta_Y[j]/hidden_to_output_weight[i][j] * derivative_of_sigmoid_func(hidden_in[i]));
                }
            }

            for(u8 i = 0; i < HIDDEN_COUNT; i++){
                if(hidden_out[i] == 0){
                    for(u8 j = 0; j < OUTPUT_COUNT; j++){
                        delta_hidden_to_output_weight[i][j] = 0;
                    }
                }
                else{
                    for(u8 j = 0; j < OUTPUT_COUNT; j++){
                        delta_hidden_to_output_weight[i][j] = delta_Y[j] / hidden_out[i];
                    }
                }
            }

            for(u8 i = 0; i < INPUT_COUNT; i++){
                if (input[i][io_no] == 0){
                    for(u8 j = 0; j < HIDDEN_COUNT; j++){
                        delta_input_to_hidden_weight[i][j] = 0;
                    }
                }
                else{
                    for(u8 j = 0; j < HIDDEN_COUNT; j++){
                        delta_input_to_hidden_weight[i][j] = delta_hidden[j]/input[i][io_no];
                    }
                }
            }

            for(u8 i = 0; i < INPUT_COUNT; i++){
                for(u8 j = 0; j < HIDDEN_COUNT; j++){
                    input_to_hidden_weight[i][j] = input_to_hidden_weight[i][j] + delta_input_to_hidden_weight[i][j];
                }
            }
            for(u8 i = 0; i < HIDDEN_COUNT; i++){
                for(u8 j = 0; j < OUTPUT_COUNT; j++){
                    hidden_to_output_weight[i][j] = hidden_to_output_weight[i][j] + delta_hidden_to_output_weight[i][j];
                }
            }
        }
    }
    for(u8 j = 0; j < IO_ARRAY_LENGTH; j++){
        for(u8 i = 0; i < OUTPUT_COUNT; i++){
            qDebug() << QString("output[%1][%2] : ").arg(i).arg(j) << calculated_output[i][j];
        }
    }
}
*/
void MainWindow::xor_ann(void){
    double input1[4] = {0,0,1,1};
    double input2[4] = {0,1,0,1};
    double desired_output[4] = {0,1,1,0};
    double calculated_output[4] = {0,0,0,0};
    double Y_in,Y_out,delta_Y;
    double w[9][4],delta_w[9][4];
    double A_in,B_in,C_in;
    double A_out,B_out,C_out;
    double delta_A,delta_B,delta_C;
    double error;
    u32 epoch = 2000;

    for(u8 i = 0; i < 4; i++){
        qDebug() << " input1 : " << input1[i] << " input2 : " << input2[i] << "output : " << desired_output[i];
    }
    for(u8 k = 0; k < 4; k++){
        for(u8 i = 0; i < 9; i++){
            w[i][k] = 0.1 * i + 0.1;
        }
    }

    for(u32 era = 0; era < epoch; era++){
        for(u8 k = 0; k < 4; k++){

            A_in = input1[k]*w[0][k] + input2[k]*w[1][k];
            B_in = input1[k]*w[2][k] + input2[k]*w[3][k];
            C_in = input1[k]*w[4][k] + input2[k]*w[5][k];

            A_out = sigmoid_func(A_in);
            B_out = sigmoid_func(B_in);
            C_out = sigmoid_func(C_in);

            Y_in = A_out*w[6][k] + B_out*w[7][k] + C_out*w[8][k];
            Y_out = sigmoid_func(Y_in);
            error = desired_output[k] - Y_out;
            calculated_output[k] = Y_out;

            delta_Y = derivative_of_sigmoid_func(Y_in) * error;
            delta_A = delta_Y/w[6][k] * derivative_of_sigmoid_func(A_in);
            delta_B = delta_Y/w[7][k] * derivative_of_sigmoid_func(B_in);
            delta_C = delta_Y/w[8][k] * derivative_of_sigmoid_func(C_in);

            if (input1[k] == 0){
                delta_w[0][k] = 0;
                delta_w[2][k] = 0;
                delta_w[4][k] = 0;
            }
            else{
                delta_w[0][k] = delta_A/input1[k];
                delta_w[2][k] = delta_B/input1[k];
                delta_w[4][k] = delta_C/input1[k];
            }

            if (input2[k] == 0){
                delta_w[1][k] = 0;
                delta_w[3][k] = 0;
                delta_w[5][k] = 0;
            }
            else{
                delta_w[1][k] = delta_A/input2[k];
                delta_w[3][k] = delta_B/input2[k];
                delta_w[5][k] = delta_C/input2[k];
            }

            if(A_out == 0){
                delta_w[6][k] = 0;
            }
            else{
                delta_w[6][k] = delta_Y / A_out;
            }

            if(B_out == 0){
                delta_w[7][k] = 0;
            }
            else{
                delta_w[7][k] = delta_Y / B_out;
            }

            if(C_out == 0){
                delta_w[8][k] = 0;
            }
            else{
                delta_w[8][k] = delta_Y / C_out;
            }

            for(u8 i = 0; i < 9; i++){
                w[i][k] = w[i][k] + delta_w[i][k];
            }

        }
    }
    for(u8 k = 0; k < 4; k++){
        for(u8 i = 0; i < 9; i++){
            qDebug() << QString("w[%1]").arg(i) << w[i][k];
        }

    }

    for(u8 k = 0; k < 4; k++){
        qDebug() << "output : " << calculated_output[k];
    }
}
double MainWindow::sigmoid_func(double val){
    return (1 / (1 + exp(-val)));
}
double MainWindow::derivative_of_sigmoid_func(double val){
    return (sigmoid_func(val) * (1 - sigmoid_func(val)));
}

MainWindow::~MainWindow()
{
    delete ui;
}
