#ifndef ANN_H
#define ANN_H

#include <QWidget>

namespace Ui {
class ann;
}
class MainWindow;

typedef unsigned char   u8;
typedef unsigned int    u32;

class ann : public QWidget
{
    Q_OBJECT

public:
    explicit ann(MainWindow *master, QWidget *parent = 0);
    ~ann();

    QThread *thread_1;

    u8 train_status;
    u8 epoch_status;

    struct _64_128_32_5_str{
        double input[64][5];
        double desired_output[5][5];
        double calculated_output[5][5];
        double hidden_neuron_bias_1[128];
        double hidden_neuron_bias_2[32];
        double output_bias[5];
        double w_input_to_hidden[64][128];
        double w_hidden_to_hidden[128][32];
        double w_hidden_to_output[32][5];
        double test_input[64];
    };
    struct _64_128_32_5_str net_64_128_32_5;

    struct _2_5_3_2_str{
        double input[2][4];
        double desired_output[2][4];
        double hidden_neuron_bias_1[5];
        double hidden_neuron_bias_2[3];
        double output_bias[2];
        double w_input_to_hidden[2][5];
        double w_hidden_to_hidden[5][3];
        double w_hidden_to_output[3][2];
    };
    struct _2_5_3_2_str net_2_5_3_2;

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
        double calculated_output[5][5];
        double hidden_bias[128];
        double output_bias[5];
        double w_input_to_hidden[64][128];
        double w_hidden_to_output[128][5];
        double test_input[64];
    };
    struct _64_128_5_str net_64_128_5;

    void advanced_64_128_5_ann_train(   double input[64][5], double desired_output[5][5], double calculated_output[5][5],
                                        double hidden_bias[128], double output_bias[5],
                                        double w_input_to_hidden[64][128],double w_hidden_to_output[128][5],
                                        u32 epoch,double learning_rate);
    void advanced_64_128_5_ann_test(    double input[64],
                                        double hidden_bias[128], double output_bias[5],
                                        double w_input_to_hidden[64][128], double w_hidden_to_output[128][5]);

private:
    Ui::ann *ui;
    MainWindow *mainwindow;
    QTimer* thread_timer;

    double sigmoid_func(double val);
    double derivative_of_sigmoid_func(double val);

    void _2_5_3_1_ann_train(void);
    void _2_5_2_1_ann_train(void);
    void _2_3_2_1_ann_train(void);
    void _2_3_1_ann_train(void);
    void _2_5_1_ann_train(void);
    void _2_5_2_ann_train(void);


    void _64_128_32_5_ann_test( double input[64],
                            double hidden_neuron_bias_1[128], double hidden_neuron_bias_2[32], double output_bias[5],
                            double w_input_to_hidden[64][128], double w_hidden_to_hidden[128][32], double w_hidden_to_output[32][5]);
    void _64_128_32_5_ann_train(double input[64][5], double desired_output[5][5], double calculated_output[5][5],
                            double hidden_neuron_bias_1[128], double hidden_neuron_bias_2[32], double output_bias[5],
                            double w_input_to_hidden[64][128], double w_hidden_to_hidden[128][32], double w_hidden_to_output[32][5],
                            u32 epoch, double learning_rate);
    void _64_128_32_5_ann_tryout(void);
    void _64_128_32_5_ann_show_weights( double hidden_neuron_bias_1[128], double hidden_neuron_bias_2[32], double output_bias[5],
                                    double w_input_to_hidden[64][128], double w_hidden_to_hidden[128][32], double w_hidden_to_output[32][5]);

    void _2_5_3_2_ann_test( double input[2],
                            double hidden_neuron_bias_1[5], double hidden_neuron_bias_2[3], double output_bias[2],
                            double w_input_to_hidden[2][5], double w_hidden_to_hidden[5][3], double w_hidden_to_output[3][2]);
    void _2_5_3_2_ann_train(double input[2][4], double desired_output[2][4],
                            double hidden_neuron_bias_1[5], double hidden_neuron_bias_2[3], double output_bias[2],
                            double w_input_to_hidden[2][5], double w_hidden_to_hidden[5][3], double w_hidden_to_output[3][2],
                            u32 epoch, double learning_rate);
    void _2_5_3_2_ann_tryout(void);
    void _2_5_3_2_ann_show_weights( double hidden_neuron_bias_1[5], double hidden_neuron_bias_2[3], double output_bias[2],
                                    double w_input_to_hidden[2][5], double w_hidden_to_hidden[5][3], double w_hidden_to_output[3][2]);


    void advanced_2_5_2_ann_train(  double input[2][4], double desired_output[2][4],
                                    double hidden_bias[5], double output_bias[2],
                                    double w_input_to_hidden[2][5],double w_hidden_to_output[5][2],
                                    u32 epoch,double learning_rate);
    void advanced_2_5_2_ann_test(   double input[2],
                                    double hidden_bias[5], double output_bias[2],
                                    double w_input_to_hidden[2][5], double w_hidden_to_output[5][2]);
    void advanced_2_5_2_tryout(void);

private slots:
    void thread_handler(void);

};

#endif // ANN_H
