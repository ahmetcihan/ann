#include "ann.h"
#include "mainwindow.h"
#include "ui_ann.h"

ann::ann(MainWindow *master, QWidget *parent) :
    QWidget(parent),
    ui(new Ui::ann)
{
    ui->setupUi(this);
    mainwindow = master;
    train_status = 0;

}

ann::~ann()
{
    delete ui;
}
