#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    _100_msec_timer = new QTimer(this);
    _100_msec_timer->setInterval(100);
    _100_msec_timer->start();
    connect(this->_100_msec_timer,SIGNAL(timeout()),this,SLOT(_100_msec_timer_handle()));


    connect(ui->pushButton_64_128_5_random_initilize,SIGNAL(clicked(bool)),this,SLOT(_64_128_5_random_initilize_handler()));
    connect(ui->pushButton_64_128_5_train,SIGNAL(clicked(bool)),this,SLOT(_64_128_5_train_handler()));
    connect(ui->pushButton_64_128_5_test,SIGNAL(clicked(bool)),this,SLOT(_64_128_5_test_handler()));
    connect(ui->pushButton_64_128_5_show_weights,SIGNAL(clicked(bool)),this,SLOT(_64_128_5_show_weights_handler()));

    train_status = 0;
}
void MainWindow::_100_msec_timer_handle(void){
    if(train_status == 1){
        ui->label_64_128_5_train->setText(QString("training status %  %1").arg(epoch_status));
    }
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
