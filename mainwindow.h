#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QDebug>
#include <math.h>

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

private:
    Ui::MainWindow *ui;

    double zero_image[8][8];
    double minus_image[8][8];
    double addition_image[8][8];
    double multiply_image[8][8];
    double divide_image[8][8];

    double input_2_4_2[2][2];
    double desired_output_2_4_2[2][2];

    double _2_4_2_input_to_hidden_weight[2][4];
    double _2_4_2_hidden_to_output_weight[4][2];

    double sigmoid_func(double val);
    double derivative_of_sigmoid_func(double val);

    void xor_ann(void);
    void _2_4_2_ann_train(double input[2][2], double desired_output[2][2], u32 epoch, double input_to_hidden_weight[2][4], double hidden_to_output_weight[4][2]);
    void _2_4_2_ann_test(double *input, double input_to_hidden_weight[2][4], double hidden_to_output_weight[4][2]);

    void _2_4_2_tryout(void);
    void _64_128_5_tryout(void);

    double input_64_128_5[64][5];
    double desired_output_64_128_5[5][5];
    double _64_128_5_input_to_hidden_weight[64][128];
    double _64_128_5_hidden_to_output_weight[128][5];

    void _64_128_5_ann_train(double input[64][5], double desired_output[5][5], u32 epoch, double input_to_hidden_weight[64][128], double hidden_to_output_weight[128][5]);
    void _64_128_5_ann_test(double *input, double input_to_hidden_weight[64][128], double hidden_to_output_weight[128][5]);

    void image_to_array(QString location, double image_array[8][8]);

};

#endif // MAINWINDOW_H
