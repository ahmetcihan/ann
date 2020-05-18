#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    ann_class = new ann(this);

    _100_msec_timer = new QTimer(this);
    _100_msec_timer->setInterval(100);
    _100_msec_timer->start();
    connect(this->_100_msec_timer,SIGNAL(timeout()),this,SLOT(_100_msec_timer_handle()));

    connect(ui->pushButton_64_128_5_random_initilize,SIGNAL(clicked(bool)),this,SLOT(_64_128_5_random_initilize_handler()));
    connect(ui->pushButton_64_128_5_train,SIGNAL(clicked(bool)),this,SLOT(_64_128_5_train_handler()));
    connect(ui->pushButton_64_128_5_test,SIGNAL(clicked(bool)),this,SLOT(_64_128_5_test_handler()));
    connect(ui->pushButton_64_128_5_show_weights,SIGNAL(clicked(bool)),this,SLOT(_64_128_5_show_weights_handler()));
    connect(ui->pushButton_64_128_5_save_weights,SIGNAL(clicked(bool)),this,SLOT(_64_128_5_save_weights_handler()));
    connect(ui->pushButton_64_128_5_load_saved_weights,SIGNAL(clicked(bool)),this,SLOT(_64_128_5_load_saved_weights_handler()));

}

void MainWindow::_100_msec_timer_handle(void){
    if(ann_class->train_status == 1){
        ui->label_64_128_5_train->setText(QString("training status %  %1").arg(ann_class->epoch_status));
        qDebug() << QString("training status %  %1").arg(ann_class->epoch_status);
    }
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
