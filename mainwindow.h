#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QDebug>
#include <math.h>
#include <QTimer>
#include "ann.h"

typedef unsigned char   u8;
typedef unsigned int    u32;

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
    Ui::MainWindow *ui;

    double test_image[8][8];
    double zero_image[8][8];
    double minus_image[8][8];
    double addition_image[8][8];
    double multiply_image[8][8];
    double divide_image[8][8];

    void image_to_array(QString location, double image_array[8][8]);

private:
    ann *ann_class;
    QTimer *_100_msec_timer;

private slots:
    void _100_msec_timer_handle(void);

    void _64_128_5_random_initilize_handler(void);
    void _64_128_5_train_handler(void);
    void _64_128_5_test_handler(void);
    void _64_128_5_show_weights_handler(void);

};

#endif // MAINWINDOW_H
