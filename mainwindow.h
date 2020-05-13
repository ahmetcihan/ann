#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QDebug>
#include <math.h>
//#include <qmath.h>

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

    struct _2_5_2_str{
        double input[2][4];
        double desired_output[2][4];
        double hidden_bias[5];
        double output_bias[2];
        double w_input_to_hidden[2][5];
        double w_hidden_to_output[5][2];
    };
    struct _2_5_2_str net_2_5_2;

    struct _64_128_5_str{
        double input[64][5];
        double desired_output[5][5];
        double hidden_bias[128];
        double output_bias[5];
        double w_input_to_hidden[64][128];
        double w_hidden_to_output[128][5];
    };
    struct _64_128_5_str net_64_128_5;

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

    void _2_5_1_ann_train(void);
    void _2_5_2_ann_train(void);

    void advanced_2_5_2_ann_train(  double input[2][4], double desired_output[2][4],
                                    double hidden_bias[5], double output_bias[2],
                                    double w_input_to_hidden[2][5],double w_hidden_to_output[5][2],
                                    u32 epoch,double learning_rate);
    void advanced_2_5_2_ann_test(   double input[2],
                                    double hidden_bias[5], double output_bias[2],
                                    double w_input_to_hidden[2][5], double w_hidden_to_output[5][2]);
    void advanced_2_5_2_tryout(void);

    void advanced_64_128_5_ann_train(  double input[64][5], double desired_output[5][5],
                                    double hidden_bias[128], double output_bias[5],
                                    double w_input_to_hidden[64][128],double w_hidden_to_output[128][5],
                                    u32 epoch,double learning_rate);

    void advanced_64_128_5_tryout(void);


    double input_64_128_5[64][5];
    double desired_output_64_128_5[5][5];
    double _64_128_5_input_to_hidden_weight[64][128];
    double _64_128_5_hidden_to_output_weight[128][5];

    void _64_128_5_ann_train(double input[64][5], double desired_output[5][5], u32 epoch, double input_to_hidden_weight[64][128], double hidden_to_output_weight[128][5]);
    void _64_128_5_ann_test(double *input, double input_to_hidden_weight[64][128], double hidden_to_output_weight[128][5]);
    void _64_128_5_tryout(void);

    void image_to_array(QString location, double image_array[8][8]);

};

#endif // MAINWINDOW_H
