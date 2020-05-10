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

    double _2_3_1_weights[9];
    double _2_4_2_weights[16];

    double sigmoid_func(double val);
    double derivative_of_sigmoid_func(double val);

    void xor_ann(void);
    void _2_4_2_ann_train(double input1,double input2, double desired_output1,  double desired_output2, u32 epoch, double *weight);
    void _2_3_1_ann_train(double input1,double input2, double desired_output, u32 epoch, double *weight);
    void _2_3_1_ann_test(double input1,double input2, double *weight);
};

#endif // MAINWINDOW_H
