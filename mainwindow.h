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
#include <QDateTime>
#include <QPainter>
#include <QPixmap>

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

    struct _alphabet{
        u8 letter[26][16][16];
        u8 tester[16][16];
    };
    struct _alphabet alphabet,alphabet_italic,alphabet_bold,alphabet_bold_italic;

    void image_to_array_8x8(QString location, u8 image_array[8][8]);
    void image_to_array_16x16(QString location, u8 image_array[16][16]);

private:
    ann *ann_class;
    QTimer *_100_msec_timer;

private slots:
    void _100_msec_timer_handle(void);

    void scan_picture(void);

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

    void _256_512_512_26_random_initilize_handler(void);
    void _256_512_512_26_load_saved_weights_handler(void);
    void _256_512_512_26_train_handler(void);
    void _256_512_512_26_test_handler(void);
    void _256_512_512_26_show_weights_handler(void);
    void _256_512_512_26_save_weights_handler(void);
    void _256_512_512_26_stop_train_handler(void);
    void _256_512_512_26_letters_to_arrays(void);

};

#endif // MAINWINDOW_H
