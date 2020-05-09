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

    double sigmoid_func(double val);
    double derivative_of_sigmoid_func(double val);
};

#endif // MAINWINDOW_H
