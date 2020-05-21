#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    ann_class = new ann(this);

    _100_msec_timer = new QTimer(this);
    _100_msec_timer->setInterval(1000);
    _100_msec_timer->start();
    connect(this->_100_msec_timer,SIGNAL(timeout()),this,SLOT(_100_msec_timer_handle()));

    connect(ui->pushButton_64_128_5_random_initilize,SIGNAL(clicked(bool)),this,SLOT(_64_128_5_random_initilize_handler()));
    connect(ui->pushButton_64_128_5_train,SIGNAL(clicked(bool)),this,SLOT(_64_128_5_train_handler()));
    connect(ui->pushButton_64_128_5_test,SIGNAL(clicked(bool)),this,SLOT(_64_128_5_test_handler()));
    connect(ui->pushButton_64_128_5_show_weights,SIGNAL(clicked(bool)),this,SLOT(_64_128_5_show_weights_handler()));
    connect(ui->pushButton_64_128_5_save_weights,SIGNAL(clicked(bool)),this,SLOT(_64_128_5_save_weights_handler()));
    connect(ui->pushButton_64_128_5_load_saved_weights,SIGNAL(clicked(bool)),this,SLOT(_64_128_5_load_saved_weights_handler()));

    connect(ui->pushButton_64_128_32_5_random_initilize,SIGNAL(clicked(bool)),this,SLOT(_64_128_32_5_random_initilize_handler()));
    connect(ui->pushButton_64_128_32_5_train,SIGNAL(clicked(bool)),this,SLOT(_64_128_32_5_train_handler()));
    connect(ui->pushButton_64_128_32_5_test,SIGNAL(clicked(bool)),this,SLOT(_64_128_32_5_test_handler()));
    connect(ui->pushButton_64_128_32_5_show_weights,SIGNAL(clicked(bool)),this,SLOT(_64_128_32_5_show_weights_handler()));
    connect(ui->pushButton_64_128_32_5_save_weights,SIGNAL(clicked(bool)),this,SLOT(_64_128_32_5_save_weights_handler()));
    connect(ui->pushButton_64_128_32_5_load_saved_weights,SIGNAL(clicked(bool)),this,SLOT(_64_128_32_5_load_saved_weights_handler()));

    connect(ui->pushButton_256_512_512_26_random_initilize,SIGNAL(clicked(bool)),this,SLOT(_256_512_512_26_random_initilize_handler()));
    connect(ui->pushButton_256_512_512_26_train,SIGNAL(clicked(bool)),this,SLOT(_256_512_512_26_train_handler()));
    connect(ui->pushButton_256_512_512_26_test,SIGNAL(clicked(bool)),this,SLOT(_256_512_512_26_test_handler()));
    connect(ui->pushButton_256_512_512_26_show_weights,SIGNAL(clicked(bool)),this,SLOT(_256_512_512_26_show_weights_handler()));
    connect(ui->pushButton_256_512_512_26_save_weights,SIGNAL(clicked(bool)),this,SLOT(_256_512_512_26_save_weights_handler()));
    connect(ui->pushButton_256_512_512_26_load_saved_weights,SIGNAL(clicked(bool)),this,SLOT(_256_512_512_26_load_saved_weights_handler()));
    connect(ui->pushButton_256_512_512_26_stop_train,SIGNAL(clicked(bool)),this,SLOT(_256_512_512_26_stop_train_handler()));
}

void MainWindow::_100_msec_timer_handle(void){
    if(ann_class->train_status == 1){
        ui->label_64_128_5_train->setText(QString("training status %  %1").arg(ann_class->epoch_status));
    }
    else if(ann_class->train_status == 2){
        ui->label_64_128_32_5_train->setText(QString("training status %  %1").arg(ann_class->epoch_status));
    }
    else if(ann_class->train_status == 3){
        ui->label_256_512_512_26_train->setText(QString("training status %  %1").arg(ann_class->epoch_status));
        ui->label_256_512_512_26_train_status->setText(QString("Epoch : %1 , Error : %2 , ob-0 : %3").
                                                        arg(ann_class->epoch_no).arg(ann_class->net_256_512_512_26.total_err).
                                                        arg(ann_class->net_256_512_512_26.output_bias[0]));
        ui->label_256_512_512_26_train_status_2->setText(QString("hb1-0 : %1 , hb2-0 : %2 , wh2h-0 : %3").
                                                        arg(ann_class->net_256_512_512_26.hidden_neuron_bias_1[0]).
                                                        arg(ann_class->net_256_512_512_26.hidden_neuron_bias_2[0]).
                                                        arg(ann_class->net_256_512_512_26.w_hidden_to_hidden[0][0]));
    }
}

void MainWindow::image_to_array_8x8(QString location, u8 image_array[8][8]){
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
void MainWindow::image_to_array_16x16(QString location, u8 image_array[16][16]){
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
