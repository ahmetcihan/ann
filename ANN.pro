#-------------------------------------------------
#
# Project created by QtCreator 2020-05-03T23:47:20
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = ANN
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
    activation_function.cpp \
    net_2_5_2.cpp \
    net_easy.cpp \
    net_64_128_5.cpp

HEADERS  += mainwindow.h

FORMS    += mainwindow.ui
