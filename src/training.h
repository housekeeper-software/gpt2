#ifndef TRAINING_H_
#define TRAINING_H_

#include "data_loader.h"
#include "layers.h"
#include "tensor.h"
#include <random>
#include <string>

// 定义早停的默认参数
const int kDefaultPatience =
    10; // 默认在验证准确率连续10个eval_interval没有改善后停止
const double kDefaultMinDelta = 1e-4; // 认为验证准确率有改善的最小阈值

struct TrainingArguments {
  int epochs;
  int accumulation_steps;
  double max_grad_norm;
  int eval_interval;
  int patience;
  double min_delta;

  TrainingArguments()
      : epochs(0), accumulation_steps(1), max_grad_norm(0.0f), eval_interval(1),
        patience(kDefaultPatience), min_delta(kDefaultMinDelta) {}
};

std::map<std::string, std::vector<double>>
train(GPT *model, const std::string &model_dir, DataLoader *train_data_loader,
      DataLoader *test_data_loader, LogSoftmaxCrossEntropyLoss *loss_func,
      AdamW *optimizer, CosineAnnealingWarmRestarts *scheduler,
      const TrainingArguments &args);

#endif // TRAINING_H_