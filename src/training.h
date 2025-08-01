#ifndef TRAINING_H_
#define TRAINING_H_

#include "data_loader.h"
#include "layers.h"
#include "tensor.h"
#include <random>
#include <string>

// 定义早停的默认参数
const int kDefaultPatience =
    60; // 默认在验证准确率连续10个eval_interval没有改善后停止
const double kDefaultMinDelta = 1e-4; // 认为验证准确率有改善的最小阈值

std::map<std::string, std::vector<double>>
train(GPT *model, DataLoader *X_train_data_loader,
      DataLoader *X_test_data_loader, LogSoftmaxCrossEntropyLoss *loss_func,
      AdamW *optimizer, int epochs, const std::string &model_dir,
      int accumulation_steps = 1, int T_0 = 10, int T_mult = 2,
      double eta_min = 1e-6, int eval_interval = 1,
      int patience = kDefaultPatience, double min_delta = kDefaultMinDelta);

#endif // TRAINING_H_