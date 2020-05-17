#-------------------------------------------------
#
# Project created by QtCreator 2020-05-03T23:47:20
#
#-------------------------------------------------

QT       += core gui concurrent

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = ANN
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
    activation_function.cpp \
    net_2_5_2.cpp \
    net_easy.cpp \
    net_64_128_5.cpp \
    net_2_5_3_2.cpp \
    net_64_128_32_5.cpp \
    ann.cpp

HEADERS  += mainwindow.h \
    ann.h

FORMS    += mainwindow.ui \
    ann.ui
