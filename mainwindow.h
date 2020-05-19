#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QDebug>
#include <math.h>
#include <QTimer>
#include "ann.h"
#include <QThread>
#include <QtConcurrentRun>
#include <QSettings>
#include <QProcess>

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

    u8 test_image[8][8];
    u8 zero_image[8][8];
    u8 minus_image[8][8];
    u8 addition_image[8][8];
    u8 multiply_image[8][8];
    u8 divide_image[8][8];

    void image_to_array(QString location, u8 image_array[8][8]);

private:
    ann *ann_class;
    QTimer *_100_msec_timer;

private slots:
    void _100_msec_timer_handle(void);

    void _64_128_5_random_initilize_handler(void);
    void _64_128_5_load_saved_weights_handler(void);
    void _64_128_5_train_handler(void);
    void _64_128_5_test_handler(void);
    void _64_128_5_show_weights_handler(void);
    void _64_128_5_save_weights_handler(void);

    void _64_128_32_5_random_initilize_handler(void);
    void _64_128_32_5_load_saved_weights_handler(void);
    void _64_128_32_5_train_handler(void);
    void _64_128_32_5_test_handler(void);
    void _64_128_32_5_show_weights_handler(void);
    void _64_128_32_5_save_weights_handler(void);

};

#endif // MAINWINDOW_H
