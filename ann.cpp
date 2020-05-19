#include "ann.h"
#include "mainwindow.h"
#include "ui_ann.h"
#include "ui_mainwindow.h"

ann::ann(MainWindow *master, QWidget *parent) :
    QWidget(parent),
    ui(new Ui::ann)
{
    ui->setupUi(this);
    mainwindow = master;
    train_status = 0;

    thread_1 = new QThread(this);
    thread_timer = new QTimer(0); //parent must be null
    thread_timer->setInterval(100);
    thread_timer->moveToThread(thread_1);
    connect(thread_timer, SIGNAL(timeout()), SLOT(thread_handler()), Qt::DirectConnection);
    QObject::connect(thread_1, SIGNAL(started()), thread_timer, SLOT(start()));
    thread_1->start();

}
void ann::thread_handler(void){
    if(train_status == 1){
        advanced_64_128_5_ann_train(net_64_128_5.input, net_64_128_5.desired_output, net_64_128_5.calculated_output,
                                 net_64_128_5.hidden_bias,net_64_128_5.output_bias,
                                 net_64_128_5.w_input_to_hidden,net_64_128_5.w_hidden_to_output,
                                 1000, 0.001);

        for(u8 i = 0; i < 5; i++){
            for(u8 j = 0; j < 5; j++){
                qDebug() << QString("desired output[%1][%2] : ").arg(i).arg(j) << net_64_128_5.desired_output[i][j] <<
                            QString("calculated output[%1][%2] : ").arg(i).arg(j) << net_64_128_5.calculated_output[i][j];
            }
        }

        double total_error = 0;
        double aux;

        for(u8 i = 0; i < 5; i++){
            for(u8 j = 0; j < 5; j++){
                aux = net_64_128_5.desired_output[i][j] - net_64_128_5.calculated_output[i][j];
                aux = aux * aux;
                total_error += aux;
            }
        }

        mainwindow->ui->label_64_128_5_train->setText(QString("Trained. Total error is %1").arg(total_error));
        train_status = 0;
    }
    else if(train_status == 2){
        _64_128_32_5_ann_train(net_64_128_32_5.input, net_64_128_32_5.desired_output, net_64_128_32_5.calculated_output,
                                 net_64_128_32_5.hidden_neuron_bias_1,net_64_128_32_5.hidden_neuron_bias_2,net_64_128_32_5.output_bias,
                                 net_64_128_32_5.w_input_to_hidden,net_64_128_32_5.w_hidden_to_hidden,net_64_128_32_5.w_hidden_to_output,
                                 1000, 0.01);

        for(u8 i = 0; i < 5; i++){
            for(u8 j = 0; j < 5; j++){
                qDebug() << QString("desired output[%1][%2] : ").arg(i).arg(j) << net_64_128_32_5.desired_output[i][j] <<
                            QString("calculated output[%1][%2] : ").arg(i).arg(j) << net_64_128_32_5.calculated_output[i][j];
            }
        }

        double total_error = 0;
        double aux;

        for(u8 i = 0; i < 5; i++){
            for(u8 j = 0; j < 5; j++){
                aux = net_64_128_32_5.desired_output[i][j] - net_64_128_32_5.calculated_output[i][j];
                aux = aux * aux;
                total_error += aux;
            }
        }

        mainwindow->ui->label_64_128_32_5_train->setText(QString("Trained. Total error is %1").arg(total_error));
        train_status = 0;
    }
}

ann::~ann()
{
    delete ui;
}
