/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>
#include <af/util.h>
#include "mnist_common.h"

#include <math.h>
#include <stdio.h>
#include <string>
#include <vector>

using namespace af;
using std::vector;

std::string toStr(const dtype dt) {
    switch (dt) {
        case f32: return "f32";
        case f16: return "f16";
        default: return "N/A";
    }
}

float accuracy(const array &predicted, const array &target) {
    array val, plabels, tlabels;
    max(val, tlabels, target, 1);
    max(val, plabels, predicted, 1);
    return 100 * count<float>(plabels == tlabels) / tlabels.elements();
}

// activation function
inline array activate(const array &out) { return sigmoid(out); }

// Derivative of the activation function
array deriv(const array &out) { return out * (1 - out); }

// Cost function
inline float error(const array &out, const array &pred) {
    array dif = (out - pred);
    return sqrtf(sum<float>(dif * dif));
}

class ann {
   private:
    int num_layers;
    vector<array> weightsT;
    dtype datatype;

    // Add bias input to the output from previous layer
    inline array add_bias(const array &in) const;
    inline array::array_proxy no_bias(array &in) const;
    inline const array::array_proxy no_bias(const array &in) const;

    const vector<array> forward_propagate(const array &input) const;

    void back_propagate(const vector<array> &signalBiased, const array &pred,
                        const double &alpha);

   public:
    // Create a network with given parameters
    ann(vector<int> layers, double range, dtype dt = f32);

    // Output after single pass of forward propagation
    inline const array predict(const array &input) const;

    // Method to train the neural net
    double train(const array &input, const array &target, double alpha = 1.0,
                 int max_epochs = 300, int batch_size = 100,
                 double maxerr = 1.0, bool verbose = false);
};

inline array ann::add_bias(const array &in) const {
    // Bias input is added on top of given input
    return join(1, constant(1, in.dims(0), 1, datatype), in);
}

inline array::array_proxy ann::no_bias(array &in) const {
    return in(span, seq(1, in.dims().dims[1] - 1));
}

inline const array::array_proxy ann::no_bias(const array &in) const {
    return in(span, seq(1, in.dims().dims[1] - 1));
}

const vector<array> ann::forward_propagate(const array &input) const {
    // Keep the bias for next round, which is the same when in training
    static vector<array> signalsBias(num_layers);

    // Re-initialize signalsBias, when input has a different # of rows
    const dim_t inRows{input.dims().dims[0]};
    if (signalsBias[0].dims().dims[0] != inRows) {
        dim_t outRows{weightsT[0].dims().dims[1]};
        for (int i{0}, end{num_layers - 1}; i < end; ++i) {
            signalsBias[i] = constant(1.0, dim4(inRows, outRows), datatype);
            outRows        = weightsT[i].dims().dims[0] + 1;
        }
        signalsBias.back() = constant(1.0, dim4(inRows, outRows), datatype);
    }

    // Get activations at each layer
    no_bias(signalsBias[0]) = input;
    for (int i{0}, end{num_layers - 1}; i < end; ++i) {
        const array out             = matmulNT(signalsBias[i], weightsT[i]);
        no_bias(signalsBias[i + 1]) = activate(out);
    }
    return signalsBias;
}

#define PR2(str, arr1, arr2) \
    pr("Line:" + std::to_string(__LINE__) + " " + str, arr1, arr2)
#define PR(str, arr1) pr("Line:" + std::to_string(__LINE__) + " " + str, arr1)

void pr(const std::string &str, const array &arr1,
        const array &arr2 = array()) {
    dim4 dim(arr1.dims());
    const dim4 strides(1, dim.dims[0], dim.dims[0] * dim.dims[1],
                       dim.dims[0] * dim.dims[1] * dim.dims[2]);
    double *a1_64{arr1.as(f64).host<double>()};
    double *diff_64{arr2.isempty()
                        ? nullptr
                        : (arr1.as(f64) - arr2.as(f64)).host<double>()};

    std::cout << '\n' << str << " [" << dim << "]\n";
    // std::cout.precision(arr1.type() == f32 ? 5 : 10);
    // std::cout.setf(std::ios::fixed);

    for (int d3{0}; d3 < dim.dims[3]; ++d3) {
        if (d3 != 0) std::cout << '\n';
        for (int d2{0}; d2 < dim.dims[2]; ++d2) {
            if (d2 != 0) std::cout << '\n';
            for (int d0{0}; d0 < dim.dims[0]; ++d0) {
                if (d0 != 0) std::cout << '\n';
                for (int d1{0}; d1 < dim.dims[1]; ++d1) {
                    std::cout.width(24);
                    std::cout << std::hexfloat
                              << a1_64[d0 * strides[0] + d1 * strides[1] +
                                       d2 * strides[2] + d3 * strides[3]]
                              << ' ';
                    if (diff_64 != nullptr) {
                        std::cout << '(';
                        std::cout.width(13);
                        std::cout.precision(10);
                        std::cout << std::defaultfloat << std::fixed
                                  << diff_64[d0 * strides[0] + d1 * strides[1] +
                                             d2 * strides[2] + d3 * strides[3]]
                                  << ")  ";
                    }
                }
            }
        }
    }
    std::cout << '\n' << std::defaultfloat;
    if (a1_64) freeHost(a1_64);
    if (diff_64) freeHost(diff_64);
}

void ann::back_propagate(  // const vector<array> signal, const array &target,
    const vector<array> &signalsBiased, const array &target,
    const double &alpha) {
    int m = target.dims(0);

    // Get error for output layer
    array out = no_bias(signalsBiased.back());
    array err = (out - target);
    for (int i = num_layers - 2; i >= 0; i--) {
        const array deltaT{deriv(out) * err};
        const array grad{matmulTN(deltaT, signalsBiased[i]) * (alpha / m)};
        weightsT[i] -= grad;
        err = no_bias(matmul(deltaT, weightsT[i]));
        out = no_bias(signalsBiased[i]);
    }
}

ann::ann(vector<int> layers, double range, dtype dt)
    : num_layers(layers.size()), weightsT(layers.size() - 1), datatype(dt) {
    for (int i = 0; i < num_layers - 1; i++) {
        weightsT[i] = range * randu(layers[i + 1], layers[i] + 1) - range / 2;
        if (datatype != f32) weightsT[i] = weightsT[i].as(datatype);
    }
}

inline const array ann::predict(const array &input) const {
    return no_bias(forward_propagate(input).back());
}

array train_target, test_target, train_feats, test_feats;

double ann::train(const array &input, const array &target, double alpha,
                  int max_epochs, int batch_size, double maxerr, bool verbose) {
    const int num_samples = input.dims().dims[0];
    const int num_batches = num_samples / batch_size;

    float err = 0;

    // Training the entire network
    for (int i = 0; i < max_epochs; i++) {
        for (int j = 0; j < num_batches - 1; j++) {
            const int st = j * batch_size;
            const int en = st + batch_size - 1;

            const array x = input(seq(st, en), span);
            const array y = target(seq(st, en), span);

            // Propagate the inputs forward
            const vector<array> signalsBiased{forward_propagate(x)};

            // Propagate the error backward
            back_propagate(signalsBiased, y, alpha);
        }

        // Validate with last batch
        const int st{(num_batches - 1) * batch_size};
        const int en{num_samples - 1};
        const array out{predict(input(seq(st, en), span))};
        err = error(out, target(seq(st, en), span));

        // Check if convergence criteria has been met
        if (err < maxerr) {
            printf("Converged on Epoch: %4d\n", i + 1);
            return err;
        }

        if (verbose) {
            if ((i + 1) % 10 == 0) {
                array train_output = predict(train_feats);
                array test_output  = predict(test_feats);
                printf(
                    "Epoch: %4d, Error: %7.4f  | Accuracy training: %7.4f | "
                    "Accuracy test: %7.4f\n",
                    i + 1, err, accuracy(train_output, train_target),
                    accuracy(test_output, test_target));
            }
        }
    }
    return err;
}

int ann_demo(bool console, int perc, const dtype dt) {
    printf("** ArrayFire ANN Demo **\n\n");

    array train_images, test_images;
    int num_classes, num_train, num_test;

    // Load mnist data
    float frac = (float)(perc) / 100.0;
    setup_mnist<true>(&num_classes, &num_train, &num_test, train_images,
                      test_images, train_target, test_target, frac);
    if (dt != f32) {
        train_images = train_images.as(dt);
        test_images  = test_images.as(dt);
        train_target = train_target.as(dt);
    }

    int feature_size = train_images.elements() / num_train;

    // Reshape images into feature vectors
    train_feats = moddims(train_images, feature_size, num_train).T();
    test_feats  = moddims(test_images, feature_size, num_test).T();

    train_target = train_target.T();
    test_target  = test_target.T();

    // Network parameters
    vector<int> layers;
    layers.push_back(train_feats.dims(1));
    layers.push_back(100);
    layers.push_back(50);
    layers.push_back(num_classes);

    const int epochs = 250;
    std::cout << "Epochs:" << epochs << "; Precision:" << toStr(dt) << '\n';
    for (int batchSize :
         {// 2, 3, 5, 7, 11,
          // 15, 16, 17, 18, 19, 20, 23, 24, 27, 32, 40, 43, 47, 48
          32, 40, 48, 52, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 256, 512,
          1024}) {
        // Create network: architecture, range, datatype
        ann network(layers, 0.05, dt);

        // Train network
        timer::start();
        network.train(train_feats, train_target,
                      2.0,        // learning rate / alpha
                      epochs,     // max epochs
                      batchSize,  // WBN 100,    // batch size
                      0.5,        // max error
                      false);     // false);     // verbose
        af::sync();
        double train_time = timer::stop();

        // Run the trained network and test accuracy.
        array train_output = network.predict(train_feats);
        array test_output  = network.predict(test_feats);

        // Benchmark prediction
        // af::sync();
        // timer::start();
        // for (int i = 1000; i > 0; --i) { network.predict(test_feats); }
        // af::sync();
        // double test_time = timer::stop() / 100;

        std::cout << "Batch size: " << batchSize
                  << " | Training time: " << train_time
                  << " s"
                  //<< " - Predic time: " << test_time << " s"
                  << " | Accuracy training: "
                  << accuracy(train_output, train_target)
                  << " | Accuracy test: " << accuracy(test_output, test_target)
                  << "\n\n";
    };

    if (!console) {
        // Get 20 random test images.
        // test_output = test_output.T();
        // display_results<true>(test_images, test_output, test_target.T(),
        // 20);
    }

    return 0;
}

int main(int argc, char **argv) {
    // usage:  neural_network_xxx (device) (console on/off) (percentage
    // training/test set) (f32|f16)
    int device   = argc > 1 ? atoi(argv[1]) : 0;
    bool console = argc > 2 ? argv[2][0] == '-' : false;
    int perc     = argc > 3 ? atoi(argv[3]) : 60;
    if (perc < 0 || perc > 100) {
        std::cerr << "Bad perc arg: " << perc << std::endl;
        return EXIT_FAILURE;
    }
    std::string dts = argc > 4 ? argv[4] : "f32";
    dtype dt        = f32;
    if (dts == "f16")
        dt = f16;
    else if (dts != "f32") {
        std::cerr << "Unsupported datatype " << dts << ". Supported: f32 or f16"
                  << std::endl;
        return EXIT_FAILURE;
    }

    if (dts == "f16" && !af::isHalfAvailable(device)) {
        std::cerr << "Half not available for device " << device << std::endl;
        return EXIT_FAILURE;
    }

    try {
        af::setDevice(device);
        af::info();
        return ann_demo(console, perc, dt);
    } catch (af::exception &ae) { std::cerr << ae.what() << std::endl; }

    return 0;
}
